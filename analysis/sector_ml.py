"""强势板块 ML 识别模型 — 从个股特征聚合到 SW L1 板块级，预测相对收益排名。

架构：
  个股特征矩阵 (6.5M × 306) → 板块内聚合（均值） → 板块级特征
  → LightGBM Ranker / Classifier → 与 rule-based module2_persistence.py 对比
"""
import json
import logging
import os
import pickle
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import DATA_DIR, DB_PATH, COL_TS_CODE, COL_TRADE_DATE
import os

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "lgb_models")
SECTOR_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_sector.pkl")


def load_sector_mapping() -> dict:
    """加载 SW L1 行业映射 {ts_code: l1_code}。"""
    from data.stock_industry_mapping import load_stock_industry_mapping
    mapping = load_stock_industry_mapping()
    return {code: info.get("l1_code", "UNKNOWN") for code, info in mapping.items()}


def build_sector_features(
    feature_df: pd.DataFrame,
    sector_mapping: dict,
    use_dispersion: bool = True,
) -> pd.DataFrame:
    """将个股特征聚合到 SW L1 板块级。

    对每个交易日 × 每个板块，计算板块内个股特征的均值。
    可选增加板块内标准差（离散度）。

    Returns:
        sector_df: (trade_date, sector) × feature_cols
    """
    df = feature_df.copy()
    df["sector"] = df[COL_TS_CODE].map(sector_mapping).fillna("UNKNOWN")

    feat_cols = [c for c in df.columns
                 if c not in (COL_TS_CODE, COL_TRADE_DATE, "close", "group", "_ret", "sector")]

    logger.info("聚合 %d 特征到 %d 个 SW L1 板块...",
                len(feat_cols), df["sector"].nunique())

    # 按 (交易日, 板块) 分组聚合均值
    grouped = df.groupby([COL_TRADE_DATE, "sector"])

    means = grouped[feat_cols].mean()
    parts = [means.add_suffix("_mean")]

    if use_dispersion:
        # 选部分关键特征计算标准差
        key_cols = [c for c in feat_cols[:40]]  # 前 40 个
        stds = grouped[key_cols].std()
        parts.append(stds.add_suffix("_std"))

    sector_df = pd.concat(parts, axis=1).reset_index()
    sector_df[COL_TRADE_DATE] = pd.to_datetime(sector_df[COL_TRADE_DATE])

    logger.info("板块级特征: %d 行（板块×交易日）", len(sector_df))
    return sector_df


def add_sector_targets(
    sector_df: pd.DataFrame,
    stock_daily_df: pd.DataFrame,
    sector_mapping: dict,
    forward_days: int = 20,
) -> pd.DataFrame:
    """添加板块未来相对收益标签。

    sector_df: (date, sector) × features
    stock_daily_df: 个股日线（用于合成板块收益）
    """
    # 合成板块日收益（等权）
    sd = stock_daily_df.copy()
    sd["sector"] = sd[COL_TS_CODE].map(sector_mapping).fillna("UNKNOWN")
    sd[COL_TRADE_DATE] = pd.to_datetime(sd[COL_TRADE_DATE])

    # 板块每日等权收益
    sector_ret = sd.groupby([COL_TRADE_DATE, "sector"])["pct_chg"].mean() / 100
    sector_ret = sector_ret.reset_index()
    sector_ret.columns = [COL_TRADE_DATE, "sector", "sector_ret"]

    # 合并到 sector_df
    sector_df = sector_df.merge(sector_ret, on=[COL_TRADE_DATE, "sector"], how="left")

    # 按板块排序并计算前向超额收益
    sector_df = sector_df.sort_values([COL_TRADE_DATE, "sector"])

    # 每个交易日所有板块的收益（用于计算相对排名）
    date_grouped = sector_df.groupby(COL_TRADE_DATE)

    # 前向板块收益
    sector_df["fwd_ret"] = sector_df.groupby("sector")["sector_ret"].transform(
        lambda x: x.shift(-forward_days).rolling(forward_days, min_periods=5).mean()
    )

    # 板块相对排名标签（用 pd.qcut 对每个交易日切片）
    rank_labels = []
    for date, grp in sector_df.groupby(COL_TRADE_DATE):
        fwd_s = grp["fwd_ret"].dropna()
        n = len(grp)
        if len(fwd_s) >= 3:
            try:
                # 用 rank 后 qcut 避免重复值问题
                labels = pd.qcut(fwd_s.rank(method="first"), q=3, labels=[0, 1, 2])
                rank_map = dict(zip(fwd_s.index, labels))
            except Exception:
                rank_map = {}
        elif len(fwd_s) > 0:
            med = fwd_s.median()
            rank_map = {idx: (2 if v > med else 0) for idx, v in fwd_s.items()}
        else:
            rank_map = {}
        for idx in grp.index:
            rank_labels.append(rank_map.get(idx, 1))

    sector_df["fwd_rank"] = rank_labels
    sector_df["fwd_outperform"] = (sector_df["fwd_ret"] > sector_df.groupby(COL_TRADE_DATE)["fwd_ret"].transform(
        lambda x: x.median() if hasattr(x, 'median') else np.nanmedian(x))).astype(int)

    logger.info("板块标签已添加")
    return sector_df


def add_rule_scores(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """计算当前 rule-based 板块评分（简化版 module2_persistence 模拟）。"""
    # 从特征中提取规则评分所需的信号
    # module2 使用: RSI, MACD, BB, 收益斜率, 换手率, 相对强度
    sector_df = sector_df.sort_values([COL_TRADE_DATE, "sector"])

    # 板块动量 score（模拟 RSI）
    if "rsi_14_mean" in sector_df.columns:
        sector_df["rule_rsi"] = sector_df["rsi_14_mean"] / 100  # 0-1 normalize

    # 趋势强度
    if "mom20_mean" in sector_df.columns:
        sector_df["rule_mom"] = (sector_df["mom20_mean"] - sector_df["mom20_mean"].min()) / \
                                 max(1, sector_df["mom20_mean"].max() - sector_df["mom20_mean"].min())

    # 稳定性 (atr 越低越稳定)
    if "atr_pct_mean" in sector_df.columns:
        sector_df["rule_stability"] = 1 - (sector_df["atr_pct_mean"] -
                                            sector_df["atr_pct_mean"].min()) / \
                                       max(1, sector_df["atr_pct_mean"].max() -
                                           sector_df["atr_pct_mean"].min())

    # 综合规则评分
    rule_cols = [c for c in ["rule_rsi", "rule_mom", "rule_stability"] if c in sector_df.columns]
    if rule_cols:
        sector_df["rule_score"] = sector_df[rule_cols].mean(axis=1)
        # 排名
        sector_df["rule_rank_3"] = sector_df.groupby(COL_TRADE_DATE)["rule_score"].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=3, labels=[0, 1, 2],
                              duplicates="drop") if x.nunique() > 2 else 1
        )
    else:
        sector_df["rule_score"] = 0
        sector_df["rule_rank_3"] = 1

    return sector_df


def train_sector_model(
    sector_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "fwd_outperform",
    initial_train_days: int = 500,
    val_days: int = 60,
    step_days: int = 60,
) -> dict:
    """Walk-Forward 训练板块评估模型。"""
    df = sector_df.sort_values([COL_TRADE_DATE, "sector"]).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c.endswith(("_mean", "_std"))]
        exclude = {"fwd_ret", "fwd_rank", "fwd_outperform",
                    "sector_ret", "rule_score", "rule_rank_3",
                    "rule_rsi", "rule_mom", "rule_stability"}
        feature_cols = [c for c in feature_cols if c not in exclude]

    logger.info("Walk-Forward 板块: %d 行, %d 特征, 目标=%s",
                len(df), len(feature_cols), target_col)

    dates = sorted(df[COL_TRADE_DATE].unique())
    n = len(dates)

    params = dict(
        n_estimators=100, num_leaves=12, min_child_samples=20,
        learning_rate=0.05, verbosity=-1, force_col_wise=True,
        objective="binary", metric="binary_logloss",
    )

    folds = []
    fold_models = []
    train_end = initial_train_days

    fold_idx = 0
    while train_end + val_days < n:
        train_dates = dates[:train_end]
        val_dates = dates[train_end:train_end + val_days]

        train_df = df[df[COL_TRADE_DATE].isin(train_dates)].dropna(
            subset=feature_cols + [target_col])
        val_df = df[df[COL_TRADE_DATE].isin(val_dates)].dropna(
            subset=feature_cols + [target_col])

        if len(train_df) < 200 or len(val_df) < 30:
            train_end += step_days
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(0)])

        # 验证: ML 准确率
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except Exception:
            auc = 0

        # 规则评分在验证集的准确率
        rule_acc = 0
        if "rule_rank_3" in val_df.columns:
            rule_outperform = (val_df["rule_rank_3"] >= 2).astype(int)
            rule_acc = accuracy_score(y_val, rule_outperform)

        fold_result = {
            "fold": fold_idx + 1,
            "train_end": str(dates[train_end - 1]),
            "val_start": str(val_dates[0]),
            "val_end": str(val_dates[-1]),
            "accuracy": round(float(acc), 4),
            "auc": round(float(auc), 4),
            "rule_accuracy": round(float(rule_acc), 4),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_sectors": val_df["sector"].nunique(),
            "ml_beat_rule": round(float(acc - rule_acc), 4),
        }
        folds.append(fold_result)
        fold_models.append(model)

        logger.info(
            "  折%d: 训练~%s | val %s~%s | acc=%.4f auc=%.4f | 规则=%.4f | beat=%+.4f",
            fold_idx + 1, fold_result["train_end"][:10],
            fold_result["val_start"][:10], fold_result["val_end"][:10],
            acc, auc, rule_acc, acc - rule_acc,
        )

        fold_idx += 1
        train_end += step_days

    # 统计
    accs = [f["accuracy"] for f in folds]
    aucs = [f["auc"] for f in folds]
    rule_accs = [f["rule_accuracy"] for f in folds]

    overall = {
        "n_folds": len(folds),
        "mean_accuracy": float(np.mean(accs)) if accs else 0,
        "mean_auc": float(np.mean(aucs)) if aucs else 0,
        "mean_rule_accuracy": float(np.mean(rule_accs)) if rule_accs else 0,
        "accuracy_improvement": float(np.mean([f["ml_beat_rule"] for f in folds])),
        "beat_rule_ratio": float(np.mean([1 for f in folds if f["ml_beat_rule"] > 0])),
        "n_features": len(feature_cols),
        "target": target_col,
        "feature_cols": feature_cols,
    }

    # 最终模型
    logger.info("训练最终板块模型（全量数据）...")
    full_df = df.dropna(subset=feature_cols + [target_col])
    final_model = lgb.LGBMClassifier(**params)
    final_model.set_params(n_estimators=200)
    final_model.fit(full_df[feature_cols].values, full_df[target_col].values)

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    result = {
        "model_type": "sector",
        "target": target_col,
        "folds": folds,
        "fold_models": fold_models,
        "overall": overall,
        "final_model": final_model,
        "final_importance": imp,
        "n_dates": n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info("板块 Walk-Forward 完成: %d 折", len(folds))
    logger.info("  ML 平均准确率: %.4f", overall["mean_accuracy"])
    logger.info("  规则平均准确率: %.4f", overall["mean_rule_accuracy"])
    logger.info("  ML 超越规则: %.4f", overall["accuracy_improvement"])
    logger.info("  击败规则比例: %.0f%%", overall["beat_rule_ratio"] * 100)

    return result


def save_sector_model(result: dict, path: str = SECTOR_MODEL_PATH) -> None:
    """保存板块模型和报告。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    final_model = result.pop("final_model", None)
    fold_models = result.pop("fold_models", [])
    with open(path, "wb") as f:
        pickle.dump(final_model, f)

    report_path = path.replace(".pkl", "_report.json")
    report = {k: v for k, v in result.items() if k != "final_model"}
    report["model_path"] = path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    result["final_model"] = final_model
    result["fold_models"] = fold_models
    logger.info("板块模型已保存: %s", path)


def load_sector_model(path: str = SECTOR_MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_sector_training_date(report_path: str = None) -> str | None:
    if report_path is None:
        report_path = SECTOR_MODEL_PATH.replace(".pkl", "_report.json")
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path) as f:
            return json.load(f).get("timestamp", "").split(" ")[0]
    except Exception:
        return None


def _load_feature_matrix(feature_df=None) -> pd.DataFrame:
    if feature_df is not None and not feature_df.empty and len(feature_df.columns) > 50:
        return feature_df
    # 复用 market_ml 的进程内缓存（避免重复加载 16GB）
    from analysis.market_ml import _FEATURE_MATRIX_CACHE
    if _FEATURE_MATRIX_CACHE.get("df") is not None:
        return _FEATURE_MATRIX_CACHE["df"]
    cache_path = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
    if os.path.exists(cache_path):
        logger.info("加载特征矩阵缓存用于板块...")
        df = pd.read_parquet(cache_path)
        _FEATURE_MATRIX_CACHE["df"] = df
        return df
    raise FileNotFoundError("特征矩阵缓存不存在")


def ensure_sector_model_fresh(
    feature_df: pd.DataFrame | None = None,
    stock_daily_df: pd.DataFrame | None = None,
    sector_mapping: dict | None = None,
    max_stale_days: int = 3,
) -> None:
    """检查板块模型新鲜度，过期则增量训练。"""
    # 权威最新日期来自 SQLite stock_daily（parquet 可能滞后，见 ml_common 说明）
    from analysis.ml_common import get_latest_stock_daily_date, get_parquet_latest_date
    latest_data = get_latest_stock_daily_date()
    if latest_data is None:
        latest_data = get_parquet_latest_date(os.path.join(DATA_DIR, "feature_matrix_v4.parquet"))
    if latest_data is None:
        feat = _load_feature_matrix(feature_df)
        latest_data = pd.to_datetime(feat[COL_TRADE_DATE].max())

    train_date_str = get_sector_training_date()
    needs_retrain = True
    if train_date_str and latest_data is not None:
        train_date = pd.to_datetime(train_date_str)
        if (latest_data - train_date).days <= max_stale_days:
            logger.info("板块模型新鲜，跳过重训练")
            needs_retrain = False
    if needs_retrain:
        logger.info("板块模型过期（最新 %s），增量训练...",
                    latest_data.date() if latest_data is not None else "?")
        feat = _load_feature_matrix(feature_df)
        sector_df = build_sector_features(feat, sector_mapping)
        sector_df = add_sector_targets(sector_df, stock_daily_df, sector_mapping)
        sector_df = add_rule_scores(sector_df)
        result = train_sector_model(sector_df, initial_train_days=500, val_days=60, step_days=60)
        save_sector_model(result)


def predict_sector_today(
    feature_df: pd.DataFrame,
    stock_daily_df: pd.DataFrame,
    sector_mapping: dict,
    top_n: int = 5,
) -> dict:
    """用最新数据预测强势板块。

    Returns:
        {
            "ml_sector_rankings": [(sector, prob), ...],  # 按 ML 评分排序
            "top_sectors": [...],
            "details": {...}
        }
    """
    model = load_sector_model()
    if model is None:
        return {"error": "no_model", "top_sectors": [], "ml_sector_rankings": []}

    report_path = SECTOR_MODEL_PATH.replace(".pkl", "_report.json")
    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception:
        report = {}
    feature_cols = report.get("overall", {}).get("feature_cols", [])

    # 特征矩阵不足时从 parquet 加载
    if feature_df is None or feature_df.empty or len(feature_df.columns) < 50:
        cache_path = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
        if os.path.exists(cache_path):
            logger.info("加载特征矩阵缓存用于板块推理...")
            feature_df = pd.read_parquet(cache_path)
        else:
            return {"error": "no_feature_matrix", "top_sectors": [], "ml_sector_rankings": []}

    # 推理时只用最近 60 天（更快）
    latest_dates = sorted(feature_df[COL_TRADE_DATE].unique())[-60:]
    feature_subset = feature_df[feature_df[COL_TRADE_DATE].isin(latest_dates)]

    if not feature_cols:
        sector_df = build_sector_features(feature_subset, sector_mapping)
        feature_cols = [c for c in sector_df.columns if c.endswith(("_mean", "_std"))]
    else:
        sector_df = build_sector_features(feature_subset, sector_mapping)

    # 取最新交易日
    latest_date = sector_df[COL_TRADE_DATE].max()
    latest = sector_df[sector_df[COL_TRADE_DATE] == latest_date].copy()

    if latest.empty:
        return {"error": "no_data", "top_sectors": [], "ml_sector_rankings": []}

    available = [c for c in feature_cols if c in latest.columns]
    X = latest[available].values

    if X.shape[1] < len(feature_cols) * 0.5:
        logger.warning("板块预测: 仅有 %d/%d 特征", X.shape[1], len(feature_cols))
        return {"error": "too_few_features", "top_sectors": []}

    proba = model.predict_proba(X)[:, 1]

    # 板块评分排名
    rankings = sorted(zip(latest["sector"].values, proba),
                       key=lambda x: x[1], reverse=True)

    # 分 3 档
    n = len(rankings)
    strong = [s for s, p in rankings[:max(3, n // 3)] if p > 0.5]
    weak = [s for s, p in rankings[-max(3, n // 3):] if p < 0.5]
    neutral = [s for s, p in rankings if s not in strong and s not in weak]

    result_dict = {
        "ml_sector_rankings": [(s, round(float(p), 4)) for s, p in rankings],
        "top_sectors": strong[:top_n],
        "bottom_sectors": weak[:3],
        "details": {
            "model_accuracy": round(report.get("overall", {}).get("mean_accuracy", 0), 4),
            "model_auc": round(report.get("overall", {}).get("mean_auc", 0), 4),
            "training_date": report.get("timestamp", "unknown"),
            "strong_sectors": strong,
            "weak_sectors": weak,
            "n_sectors": len(rankings),
        },
        "strong_sectors_list": strong,
    }
    logger.info("板块预测: top=%s, bottom=%s", strong[:3], weak[:3])
    return result_dict
