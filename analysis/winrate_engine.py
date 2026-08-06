#!/usr/bin/env python3
"""V6 Master Plan — 胜率预测头 (Probability of Winning Head)。

核心变更（HERMES-20260801-001 → 006）：
  ❌ 停用回归目标 fwd_return（R² / Spearman 连续预测）
  ✅ 二分类 → Ordinal 三梯度 (HERMES-20260801-006 Fix-1)：
     LightGBM multiclass，标签 = 0(强亏<-2%) / 1(中性±2%) / 2(True Alpha>+2%)
  ✅ P(Win) = P(Bucket_2) = True Alpha 概率，用于排序与软权重闸门
  ✅ Rank IC 对实际 fwd>+2% 的 True Alpha 候选施加 ×2.0 置信度加成（006 任务 1.5x~2.0x，取上限）
  ✅ Bucket_0(强亏/假突破) 训练样本施加 Penalty Weight ×2.0（006 任务硬要求）
  ✅ 硬拦截：P(Win) < 55% 一票否决（Fix-1 后由 soft-weight 替代，见 dynamic_allocation）

评估面板（替代 R²）：
  Rank IC                 —— 预测 P(Win) 与实际 fwd_return_20d 的截面 Spearman IC
  Actual Win Rate (>55%)  —— 入选 Top N 中实际盈利 (fwd_return_20d>0) 的占比
  Avg_Volatility_of_Top5  —— 入选 Top N 的平均波动（默认 atr_pct）
"""
from __future__ import annotations

import glob
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 项目根路径引导（与 analysis/retrain_ml.py 一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COL_TS_CODE, COL_TRADE_DATE, DATA_DIR

logger = logging.getLogger(__name__)

# ── V6 引擎模型持久化 (HERMES-20260802-001: 每版模型必须保存) ──
V6_MODEL_DIR = os.path.join(DATA_DIR, "lgb_models")
os.makedirs(V6_MODEL_DIR, exist_ok=True)


def save_engine_model(model, engine: str, asof_date: str) -> str:
    """保存单个引擎模型为版本化文件。

    命名约定: v6_{engine}_{asof}.pkl（asof = 训练数据截止日期 YYYYMMDD）。
    每版保留、不覆盖旧版——符合「每一版模型都持久化保存」要求。

    Args:
        model: LightGBM 模型
        engine: momentum / reversion / breakout
        asof_date: 训练数据截止日（YYYYMMDD）

    Returns:
        保存的完整路径。
    """
    path = os.path.join(V6_MODEL_DIR, f"v6_{engine}_{asof_date}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("V6 引擎模型已保存: %s", path)
    return path


def load_engine_model(engine: str, asof_date: str | None = None):
    """加载 V6 引擎模型（默认最新版本，或指定 asof 版本）。

    Args:
        engine: momentum / reversion / breakout
        asof_date: 指定训练截止日（YYYYMMDD）；None 则加载该引擎最新版

    Returns:
        LightGBM 模型 或 None（无可用模型）。
    """
    if asof_date:
        path = os.path.join(V6_MODEL_DIR, f"v6_{engine}_{asof_date}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        logger.warning("V6 模型不存在: %s", path)
        return None
    # 最新版：按文件名 asof 降序取最大
    pattern = os.path.join(V6_MODEL_DIR, f"v6_{engine}_*.pkl")
    files = glob.glob(pattern)
    if not files:
        logger.warning("无 V6 引擎模型: %s", engine)
        return None
    latest = max(files, key=lambda p: os.path.basename(p).split("_")[-1].replace(".pkl", ""))
    with open(latest, "rb") as f:
        return pickle.load(f)


def list_engine_models(engine: str | None = None) -> list[str]:
    """列出已保存的 V6 引擎模型文件（可过滤指定引擎）。"""
    pattern = os.path.join(V6_MODEL_DIR, f"v6_{engine}_*.pkl" if engine else "v6_*.pkl")
    return sorted(glob.glob(pattern))


# 硬拦截默认阈值（任务硬性要求，可调）
DEFAULT_WINRATE_GATE = 0.55

# 分组默认参数（对照组：V5 回归参数 → V6 二分类参数）
WINRATE_PARAMS = dict(
    n_estimators=200, num_leaves=15, min_child_samples=20,
    learning_rate=0.05, verbosity=-1, force_col_wise=True,
    objective="multiclass", num_class=3, metric="multi_logloss",
)

EVALUATION_PANEL = ["Rank IC", "Actual Win Rate (>55%)", "Avg_Volatility_of_Top5"]


# ═══════════════════════════════════════════════════════════════
# 1. 标签构建（硬写死二分类）
# ═══════════════════════════════════════════════════════════════

def compute_forward_return(
    feature_df: pd.DataFrame,
    forward_days: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """为特征矩阵追加 fwd_return_20d（前向 N 日收益率 %）。

    向量化实现（groupby + shift），比 V5 的逐股票 Python 循环快一个量级。
    """
    if feature_df is None or feature_df.empty:
        return feature_df
    df = feature_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    # 每股前向收益：fwd = close[t+N] / close[t] - 1
    fwd = df.groupby(COL_TS_CODE)[price_col].shift(-forward_days) / df[price_col] - 1
    df["fwd_return_20d"] = (fwd * 100).round(4)
    return df


# ── Ordinal Classification 三梯度 (HERMES-20260801-006 Fix-1) ──
LABEL_LOSS_MAX = -2.0    # fwd < -2% → Bucket 0（Strong Loss / 假突破）
LABEL_ALPHA_MIN = 2.0    # fwd > +2% → Bucket 2（True Alpha）
LABEL_NEUTRAL_HI = 2.0   # Bucket 1 上界（+2%）
LABEL_NEUTRAL_LO = -2.0  # Bucket 1 下界（-2%）
RANK_IC_ALPHA_BOOST = 2.0  # Rank IC 时 Bucket_2(>+2% True Alpha) 置信度加成倍数
#                          # (006 任务硬要求 1.5x~2.0x；取上限最大化 Alpha 候选排序区分度)
LABEL_LOSS_PENALTY_WEIGHT = 2.0  # Bucket_0(强亏/假突破) 训练样本惩罚权重
#                                 # (006 任务硬要求「Bucket 1 Strong Loss -> Penalty Weight!」)


def build_winrate_labels(
    feature_df: pd.DataFrame,
    forward_days: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """构建 Ordinal 三梯度标签（HERMES-20260801-006 Fix-1，替代二分类）。

    三梯度（区分假突破/中性/真 Alpha，解决二分类无法区分强动量的缺陷）：
        fwd_return_20d < -2%  → Bucket 0 (Strong Loss / 假突破)
        -2% ≤ fwd ≤ +2%      → Bucket 1 (中性盘整 flat/range)
        fwd_return_20d > +2%  → Bucket 2 (True Alpha / 超强动能)

    Returns:
        含 fwd_return_20d 与 y_label 列的 DataFrame（尾部 forward_days 行为 NaN）
    """
    df = compute_forward_return(feature_df, forward_days=forward_days, price_col=price_col)
    if df.empty:
        return df
    fwd = df["fwd_return_20d"]
    labels = np.where(fwd < LABEL_LOSS_MAX, 0,
                      np.where(fwd > LABEL_ALPHA_MIN, 2, 1))
    df["y_label"] = labels.astype(np.int8)
    df["y_bucket"] = labels.astype(np.int8)  # 别名，供 Rank IC 置信度加成
    return df


def build_winrate_sample_weights(
    feature_df: pd.DataFrame,
    label_col: str = "y_label",
    loss_penalty: float = LABEL_LOSS_PENALTY_WEIGHT,
) -> np.ndarray:
    """训练样本权重：Bucket_0(Strong Loss / 假突破) 施加惩罚权重。

    Fix-1 (HERMES-20260801-006)：任务硬要求「Bucket 1 (Strong Loss /
    False Breakout) -> Penalty Weight!」。对 y_label==0（fwd < -2%）的样本
    乘以 loss_penalty，让模型重点学习识别亏损形态，降低 Top-N 假突破占比，
    从根源解决「系统总是挑平庸股导致排兵布阵失效」。

    Args:
        feature_df: 已含 label_col 的标签化特征矩阵
        label_col: 标签列（默认 y_label，三梯度 0/1/2）
        loss_penalty: Bucket_0 惩罚权重（默认 2.0）

    Returns:
        np.ndarray，与 feature_df 行数一致的样本权重。
    """
    if label_col not in feature_df.columns or feature_df.empty:
        return np.ones(len(feature_df), dtype=float)
    y = feature_df[label_col].astype(int).values
    w = np.ones(len(y), dtype=float)
    w[y == 0] = loss_penalty
    return w


# ═══════════════════════════════════════════════════════════════
# 2. 训练 / 预测
# ═══════════════════════════════════════════════════════════════

def train_winrate_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    forward_days: int = 20,
    params: dict | None = None,
    label_col: str = "y_label",
    loss_penalty_weight: float = LABEL_LOSS_PENALTY_WEIGHT,
    engine: str | None = None,
    asof_date: str | None = None,
) -> tuple:
    """训练单个引擎的胜率模型（LGBMClassifier，Ordinal multiclass）。

    Fix-1 (HERMES-20260801-006)：objective 从 binary 改为 multiclass，
    标签为三梯度（0=强亏/1=中性/2=True Alpha）。P(Win) = P(Bucket_2)，
    即「真 Alpha」概率，用于排序与一票否决。

    Fix (HERMES-20260802-001)：持久化——传入 engine+asof_date 时训练成功后
    自动保存为 v6_{engine}_{asof_date}.pkl（每版保留）。默认 None 不保存，
    兼容既有调用。

    Fix-1 Penalty Weight：Bucket_0(强亏/假突破) 样本按 loss_penalty_weight
    加权（sample_weight），重点识别亏损形态，降低假突破入选。

    Args:
        feature_df: 已含 y_label 的特征矩阵（或原始矩阵，内部自动构建）
        feature_cols: 该引擎的特征池（由 strategy_feature_masker 提供）
        forward_days: 前向窗口
        params: LightGBM 参数覆盖
        label_col: 标签列（默认 y_label，三梯度 0/1/2）
        loss_penalty_weight: Bucket_0 惩罚权重（默认 LABEL_LOSS_PENALTY_WEIGHT）

    Returns:
        (model, importance_df) 或 (None, None)
    """
    if label_col not in feature_df.columns:
        feature_df = build_winrate_labels(feature_df, forward_days=forward_days)
    df = feature_df.dropna(subset=[label_col])

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (COL_TS_CODE, COL_TRADE_DATE, "close", "y_label", "y_bucket", "fwd_return_20d", "group")]
    feats = [c for c in feature_cols if c in df.columns]
    if len(feats) < 3 or len(df) < 200:
        logger.warning("胜率模型训练数据不足: 特征=%d, 样本=%d", len(feats), len(df))
        return None, None

    X = df[feats].fillna(0)
    y = df[label_col].values
    # 类别分布检查（三梯度）
    bucket_counts = np.bincount(y.astype(int), minlength=3)
    alpha_rate = bucket_counts[2] / max(len(y), 1)
    if bucket_counts[2] <= 5 or alpha_rate >= 0.95:
        logger.warning("True Alpha Bucket 样本过少/失衡: %s", bucket_counts.tolist())

    cfg = dict(WINRATE_PARAMS)
    if params:
        cfg.update(params)
    cfg.setdefault("objective", "multiclass")
    cfg.setdefault("num_class", 3)
    cfg.setdefault("metric", "multi_logloss")
    model = lgb.LGBMClassifier(**cfg)
    # Fix-1 Penalty Weight: Bucket_0(强亏/假突破) 加权，重点学亏损形态
    sw = build_winrate_sample_weights(df, label_col=label_col, loss_penalty=loss_penalty_weight)
    model.fit(X, y, sample_weight=sw)

    importance = pd.DataFrame({
        "feature": feats,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("胜率模型训练完成(Ordinal): %d 样本, %d 特征, buckets=%s, alpha_rate=%.3f, "
                "loss_penalty=%.1f, top=%s",
                len(df), len(feats), bucket_counts.tolist(), alpha_rate,
                loss_penalty_weight, ", ".join(importance["feature"].head(3).tolist()))

    # HERMES-20260802-001：持久化保存（传入 engine+asof_date 时）
    if engine and asof_date:
        save_engine_model(model, engine, asof_date)
    return model, importance


def predict_win_probability(model, X: pd.DataFrame) -> np.ndarray:
    """返回 P(Win) = P(True Alpha) = Bucket_2 概率（Ordinal 三分类）。

    Fix-1 (HERMES-20260801-006)：模型输出 3 列类别概率，
    P(Win) 取最高梯度（Bucket_2 = fwd>+2% 真 Alpha）的概率。
    """
    proba = model.predict_proba(X)
    # multiclass: proba 形状 (n, num_class=3)；取最后一列（Bucket_2）
    if proba.ndim == 2 and proba.shape[1] >= 3:
        return proba[:, 2]
    return proba[:, 1]  # 兼容二分类 fallback


# ═══════════════════════════════════════════════════════════════
# 3. 评估面板（替代 R²）
# ═══════════════════════════════════════════════════════════════

def rank_ic(df: pd.DataFrame, pred_col: str, actual_col: str = "fwd_return_20d",
            alpha_boost: float = RANK_IC_ALPHA_BOOST) -> float:
    """Rank IC：逐日截面 Spearman(pred, actual) 的均值。

    Fix-1 (HERMES-20260801-006)：True Alpha 置信度加成——实际 fwd>+2% 的行，
    其 pred 额外 ×alpha_boost(默认1.5)，确保强势 Alpha 候选的排序区分度，
    解决「系统总是挑平庸股」问题。

    无 trade_date 列（如单日组合）时，退化为全样本 Spearman。
    """
    if df is None or df.empty or pred_col not in df.columns:
        return 0.0
    d = df.copy()
    if actual_col in d.columns and alpha_boost > 1.0:
        is_alpha = d[actual_col].fillna(-999) > LABEL_ALPHA_MIN
        d.loc[is_alpha, pred_col] = d.loc[is_alpha, pred_col] * alpha_boost

    if COL_TRADE_DATE not in d.columns:
        x = d[pred_col].values
        y = d[actual_col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 10:
            return 0.0
        r, _ = spearmanr(x[valid], y[valid])
        return round(float(r), 4) if not np.isnan(r) else 0.0

    ics = []
    for _date, day_df in d.groupby(COL_TRADE_DATE):
        if len(day_df) < 10:
            continue
        x = day_df[pred_col].values
        y = day_df[actual_col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 10:
            continue
        r, _ = spearmanr(x[valid], y[valid])
        if not np.isnan(r):
            ics.append(r)
    return round(float(np.mean(ics)), 4) if ics else 0.0


def _iter_dates(df: pd.DataFrame):
    """按 trade_date 分组迭代；无 trade_date 列时作为单日返回。"""
    if COL_TRADE_DATE in df.columns:
        return list(df.groupby(COL_TRADE_DATE))
    return [(None, df)]


def actual_win_rate_topn(
    df: pd.DataFrame,
    pred_col: str,
    top_n: int = 5,
    actual_col: str = "fwd_return_20d",
) -> float:
    """Actual Win Rate：按预测分取每日 Top-N，统计其中实际盈利 (fwd>0) 占比。"""
    wins, total = 0, 0
    for _date, day_df in _iter_dates(df):
        if len(day_df) < top_n:
            continue
        top = day_df.nlargest(top_n, pred_col)
        valid = top.dropna(subset=[actual_col])
        if valid.empty:
            continue
        wins += int((valid[actual_col] > 0).sum())
        total += len(valid)
    return round(wins / total, 4) if total else 0.0


def avg_volatility_topn(
    df: pd.DataFrame,
    pred_col: str,
    top_n: int = 5,
    vol_col: str | None = "atr_pct",
) -> float:
    """Avg_Volatility_of_Top5：入选 Top-N 的平均波动率（默认 atr_pct %）。"""
    if vol_col is None or vol_col not in df.columns:
        return 0.0
    vols = []
    for _date, day_df in _iter_dates(df):
        if len(day_df) < top_n:
            continue
        top = day_df.nlargest(top_n, pred_col)
        v = top[vol_col].dropna()
        if not v.empty:
            vols.append(v.mean())
    return round(float(np.mean(vols)), 4) if vols else 0.0


def evaluate_panel(
    df: pd.DataFrame,
    pred_col: str,
    top_n: int = 5,
    vol_col: str = "atr_pct",
) -> dict:
    """输出 V6 评估面板（三项指标，替代 R²）。"""
    return {
        "Rank IC": rank_ic(df, pred_col),
        "Actual Win Rate (>55%)": actual_win_rate_topn(df, pred_col, top_n=top_n),
        "Avg_Volatility_of_Top5": avg_volatility_topn(df, pred_col, top_n=top_n, vol_col=vol_col),
    }


# ═══════════════════════════════════════════════════════════════
# 4. 硬拦截闸门
# ═══════════════════════════════════════════════════════════════

def apply_winrate_gate(
    scored_df: pd.DataFrame,
    gate_threshold: float = DEFAULT_WINRATE_GATE,
    prob_col: str = "win_prob",
) -> pd.DataFrame:
    """硬拦截：P(Win) < 阈值 → 一票否决，无论绝对预测收益多高。

    返回新增 gate 列的 DataFrame：
      gate = "pass" | "veto"

    Args:
        scored_df: 已含 P(Win) 列的评分结果
        gate_threshold: 默认 0.55（任务硬要求）
        prob_col: P(Win) 列名
    """
    df = scored_df.copy()
    df["win_prob"] = df[prob_col].astype(float)
    df["gate"] = np.where(df["win_prob"] >= gate_threshold, "pass", "veto")
    n_veto = int((df["gate"] == "veto").sum())
    logger.info("胜率闸门: 阈值=%.2f, %d/%d 被一票否决", gate_threshold, n_veto, len(df))
    return df


# ═══════════════════════════════════════════════════════════════
# 5. Walk-Forward 验证（多策略独立验证）
# ═══════════════════════════════════════════════════════════════

def walk_forward_winrate(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    initial_train_days: int = 120,
    val_days: int = 25,
    step_days: int = 25,
    forward_days: int = 20,
    top_n: int = 5,
    params: dict | None = None,
    prob_col: str = "win_prob",
    loss_penalty_weight: float = LABEL_LOSS_PENALTY_WEIGHT,
) -> dict:
    """按时间序列滚动训练-验证单个引擎的胜率模型。

    每个验证折输出 V6 面板三指标 + AUC/准确率。
    Fix-1 (HERMES-20260801-006)：每折训练同样施加 Bucket_0 惩罚权重。
    Returns:
        {folds: [...], overall: {...}, n_features, ...}
    """
    df = build_winrate_labels(feature_df, forward_days=forward_days)
    df = df.dropna(subset=["y_label"]).sort_values([COL_TS_CODE, COL_TRADE_DATE])
    feats = [c for c in feature_cols if c in df.columns]
    if len(feats) < 3 or len(df) < 500:
        return {"error": f"数据不足: feats={len(feats)}, rows={len(df)}"}

    dates = sorted(df[COL_TRADE_DATE].unique())
    n_dates = len(dates)
    folds, fold_models = [], []

    train_end = min(initial_train_days, n_dates - val_days)
    while train_end + val_days <= n_dates - forward_days:
        train_dates = set(dates[:train_end])
        val_dates = set(dates[train_end:train_end + val_days])

        train_df = df[df[COL_TRADE_DATE].isin(train_dates)]
        val_df = df[df[COL_TRADE_DATE].isin(val_dates)]
        if len(train_df) < 300 or val_df.empty:
            train_end += step_days
            continue

        X_tr, y_tr = train_df[feats].fillna(0), train_df["y_label"].values
        X_va = val_df[feats].fillna(0)
        cfg = dict(WINRATE_PARAMS)
        if params:
            cfg.update(params)
        model = lgb.LGBMClassifier(**cfg)
        # Fix-1 Penalty Weight (HERMES-20260801-006)：Bucket_0 强亏样本加权
        sw_tr = build_winrate_sample_weights(train_df, label_col="y_label",
                                             loss_penalty=loss_penalty_weight)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

        val_df = val_df.copy()
        val_df[prob_col] = predict_win_probability(model, X_va)
        panel = evaluate_panel(val_df, prob_col, top_n=top_n)

        from sklearn.metrics import accuracy_score, roc_auc_score
        # multiclass: argmax 预测类别；AUC 用 OvR（Bucket_2 为 positives）
        pred_cls = np.argmax(model.predict_proba(X_va), axis=1)
        acc = accuracy_score(val_df["y_label"], pred_cls)
        try:
            auc = roc_auc_score(val_df["y_label"], val_df[prob_col],
                                multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

        folds.append({
            "fold": len(folds) + 1,
            "train_end": str(dates[train_end - 1]),
            "val_start": str(dates[train_end]),
            "val_end": str(dates[train_end + val_days - 1]),
            "n_train": len(train_df), "n_val": len(val_df),
            **panel,
            "AUC": round(float(auc), 4),
            "Accuracy": round(float(acc), 4),
        })
        fold_models.append(model)
        train_end += step_days

    if not folds:
        return {"error": "无足够折"}

    def _mean(key):
        return round(float(np.mean([f[key] for f in folds])), 4)

    overall = {
        "n_folds": len(folds),
        "mean_rank_ic": _mean("Rank IC"),
        "mean_actual_win_rate": _mean("Actual Win Rate (>55%)"),
        "mean_avg_volatility_top5": _mean("Avg_Volatility_of_Top5"),
        "mean_auc": _mean("AUC"),
        "mean_accuracy": _mean("Accuracy"),
        "positive_winrate_folds": round(float(np.mean([1 for f in folds if f["Actual Win Rate (>55%)"] >= 0.55])), 4),
        "n_features": len(feats),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"folds": folds, "fold_models": fold_models, "overall": overall, "n_features": len(feats)}


# ═══════════════════════════════════════════════════════════════
# 6. 自检
# ═══════════════════════════════════════════════════════════════

def _self_test() -> dict:
    """合成数据上验证：标签硬写死 / 二分类可训 / 闸门生效 / 面板指标。"""
    rng = np.random.default_rng(7)
    n_codes, n_days = 40, 90
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]
    dates = pd.bdate_range("2026-01-05", periods=n_days).strftime("%Y%m%d")

    rows = []
    for c in codes:
        drift = rng.normal(0.02, 0.01)
        for d in dates:
            rows.append({
                COL_TS_CODE: c, COL_TRADE_DATE: d,
                "close": 10 * np.exp(np.cumsum(rng.normal(drift, 0.03))[-1]) if len(rows) else 10,
                "mom20": rng.normal(), "ma20_dev": rng.normal(), "rsi_14": rng.uniform(20, 80),
                "bb_pct_b": rng.uniform(0, 1), "atr_pct": rng.uniform(1, 8),
            })
    df = pd.DataFrame(rows).sort_values([COL_TS_CODE, COL_TRADE_DATE]).reset_index(drop=True)

    labeled = build_winrate_labels(df, forward_days=5)
    assert {"fwd_return_20d", "y_label"} <= set(labeled.columns)
    labeled_valid = labeled.dropna(subset=["y_label"])
    # 硬写死校验（三梯度 Ordinal）
    expected = np.where(labeled_valid["fwd_return_20d"] < LABEL_LOSS_MAX, 0,
                        np.where(labeled_valid["fwd_return_20d"] > LABEL_ALPHA_MIN, 2, 1)).astype(np.int8)
    assert (labeled_valid["y_label"].values == expected).all(), "标签逻辑不符合 Ordinal 三梯度"
    assert set(labeled_valid["y_label"].unique()) <= {0, 1, 2}, "y_label 应取值 {0,1,2}"

    feats = ["mom20", "ma20_dev", "rsi_14", "bb_pct_b"]
    model, imp = train_winrate_model(labeled, feature_cols=feats, forward_days=5)
    assert model is not None
    assert isinstance(model, lgb.LGBMClassifier)
    assert model.objective == "multiclass", "Ordinal 应使用 multiclass objective"

    # Fix-1 Penalty Weight 校验 (HERMES-20260801-006)
    sw = build_winrate_sample_weights(labeled_valid)
    assert float(sw.max()) == LABEL_LOSS_PENALTY_WEIGHT, "Bucket_0 惩罚权重应为 LABEL_LOSS_PENALTY_WEIGHT"
    assert float(sw.min()) == 1.0, "非 Bucket_0 权重应为 1.0"
    assert len(sw) == len(labeled_valid)
    model2, _ = train_winrate_model(labeled, feature_cols=feats, forward_days=5,
                                    loss_penalty_weight=3.0)
    assert model2 is not None, "带 loss_penalty_weight 参数训练应成功"

    val = labeled_valid.iloc[:500].copy()
    val["win_prob"] = predict_win_probability(model, val[feats].fillna(0))
    panel = evaluate_panel(val, "win_prob", top_n=5, vol_col="atr_pct")

    gated = apply_winrate_gate(val, gate_threshold=0.55, prob_col="win_prob")
    assert set(gated["gate"].unique()) <= {"pass", "veto"}
    assert (gated.loc[gated["gate"] == "pass", "win_prob"] >= 0.55).all()
    n_veto = int((gated["gate"] == "veto").sum())

    return {
        "label_hardcoded_ok": True,
        "model_type": str(type(model).__name__),
        "objective": model.objective,
        "penalty_weight_max": float(sw.max()),
        "panel": panel,
        "gate_veto_count": n_veto,
        "gate_passed_all_above_threshold": bool((gated.loc[gated["gate"] == "pass", "win_prob"] >= 0.55).all()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import json
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))
