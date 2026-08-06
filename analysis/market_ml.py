"""大盘评估 ML 模型 — 从个股特征聚合到市场级，训练 LightGBM 预测方向。

架构：
  个股特征矩阵 (6.5M × 306) → 截面聚合（每日均值/标准差） → 市场级特征 (~2600 天)
  → LightGBM Walk-Forward → 与 rule-based market_regime.py 对比
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

from config import DATA_DIR, COL_TS_CODE, COL_TRADE_DATE

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "lgb_models")
MARKET_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_market.pkl")

# ── 需要聚类的聚合统计量 ──
AGG_STATS = ["mean", "std", "q25", "q75"]

# 模型 1 中最重要的 20 个特征（用于计算市场离散度）
KEY_DISPERSION_FEATURES = [
    "mom20", "ma20_dev", "vol_ratio", "atr_pct", "sharpe_20d",
    "bb_pct_b", "rsi_14", "streak", "amihud_illiq_20d", "turnover_ratio_5d_20d",
    "autocorr_1d", "variance_ratio_5_1", "gain_loss_consistency",
]


def build_market_features(
    feature_df: pd.DataFrame,
    market_index_df: pd.DataFrame | None = None,
    stock_daily_df: pd.DataFrame | None = None,
    use_dispersion: bool = True,
) -> pd.DataFrame:
    """将个股级特征聚合成市场级特征。

    对每个交易日，所有股票截面的均值作为市场级特征。
    可选增加标准差/分位数作为离散度特征。
    大盘目标收益从 market_index_df 获取，若无则从 stock_daily_df 合成。
    """
    feat_cols = [c for c in feature_df.columns
                 if c not in (COL_TS_CODE, COL_TRADE_DATE, "close", "group", "_ret")]
    # 只保留数值列
    numeric_cols = feature_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()

    logger.info("聚合 %d 个特征到市场级...", len(numeric_cols))

    # 按交易日分组聚合
    grouped = feature_df.groupby(COL_TRADE_DATE)

    # 均值（全部特征）
    means = grouped[numeric_cols].mean()
    means.columns = [f"{c}_mean" for c in means.columns]

    # 可选离散度
    parts = [means]

    if use_dispersion:
        # 所有特征的 std（计算量较大，用采样或部分特征）
        disp_cols = [c for c in numeric_cols if c in KEY_DISPERSION_FEATURES or
                     any(k in c for k in ["mom", "ma", "vol", "atr", "rsi", "bb_", "streak"])]
        # 最多 60 个离散度特征
        disp_cols = disp_cols[:60]

        stds = grouped[disp_cols].std()
        stds.columns = [f"{c}_std" for c in stds.columns]
        parts.append(stds)

        # 分位数（关键特征）
        for q, label in [(0.25, "q25"), (0.75, "q75")]:
            qdf = grouped[disp_cols[:20]].quantile(q)
            qdf.columns = [f"{c}_{label}" for c in qdf.columns]
            parts.append(qdf)

    # 合并
    market_df = pd.concat(parts, axis=1)
    market_df.index.name = COL_TRADE_DATE
    market_df = market_df.reset_index()
    market_df[COL_TRADE_DATE] = pd.to_datetime(market_df[COL_TRADE_DATE])

    # ── 添加指数特征 + 标签 ──
    # 优先用外部大盘指数，回退到个股等权合成
    mkt_source = None
    if market_index_df is not None and len(market_index_df) > 10:
        mkt_source = market_index_df.copy()
        logger.info("使用外部大盘指数: %d 行", len(mkt_source))
    elif stock_daily_df is not None:
        logger.info("从 stock_daily 合成等权市场指数...")
        sd = stock_daily_df.copy()
        sd[COL_TRADE_DATE] = pd.to_datetime(sd[COL_TRADE_DATE])
        mkt_ret = sd.groupby(COL_TRADE_DATE)["pct_chg"].mean() / 100
        mkt_index = (1 + mkt_ret).cumprod()
        mkt_price = mkt_index * 1000 / mkt_index.iloc[0]
        # 用价格均值近似 high/low（regime 计算需要这些列）
        if "high" in sd.columns and "low" in sd.columns:
            hi = sd.groupby(COL_TRADE_DATE)["high"].mean()
            lo = sd.groupby(COL_TRADE_DATE)["low"].mean()
        else:
            hi, lo = mkt_price, mkt_price
        mkt_source = pd.DataFrame({
            COL_TRADE_DATE: mkt_ret.index,
            "close": mkt_price.values,
            "high": hi.values,
            "low": lo.values,
            "open": mkt_price.values,
        })
        logger.info("合成市场指数: %d 天, %s ~ %s",
                    len(mkt_source), mkt_source[COL_TRADE_DATE].min(), mkt_source[COL_TRADE_DATE].max())

    if mkt_source is not None:
        mkt_source[COL_TRADE_DATE] = pd.to_datetime(mkt_source[COL_TRADE_DATE])
        mkt_source = mkt_source.sort_values(COL_TRADE_DATE)

        # 指数日收益
        mkt_source["idx_ret"] = mkt_source["close"].pct_change()
        mkt_source["idx_ret_5d"] = mkt_source["close"].pct_change(5)
        mkt_source["idx_ret_20d"] = mkt_source["close"].pct_change(20)

        # 指数技术特征
        mkt_source["idx_ma5"] = mkt_source["close"].rolling(5).mean()
        mkt_source["idx_ma20"] = mkt_source["close"].rolling(20).mean()
        mkt_source["idx_ma60"] = mkt_source["close"].rolling(60).mean()
        mkt_source["idx_vol_20d"] = mkt_source["idx_ret"].rolling(20).std()

        # 当前 rule-based regime（用于对比）
        try:
            from analysis.market_regime import determine_regime
            regime_results = []
            for i in range(len(mkt_source)):
                day_df = mkt_source.iloc[:i + 1]
                try:
                    r = determine_regime(day_df)
                    regime_results.append(r.get("v2_label", "unknown"))
                except Exception:
                    regime_results.append("unknown")
            mkt_source["regime_label"] = regime_results
        except Exception:
            mkt_source["regime_label"] = "unknown"

        # 标签: 前向收益（回归）和前向方向（分类）
        mkt_source["fwd_ret_5"] = mkt_source["close"].shift(-5) / mkt_source["close"] - 1
        mkt_source["fwd_ret_20"] = mkt_source["close"].shift(-20) / mkt_source["close"] - 1
        mkt_source["fwd_dir_5"] = (mkt_source["fwd_ret_5"] > 0).astype(int)
        mkt_source["fwd_dir_20"] = (mkt_source["fwd_ret_20"] > 0).astype(int)

        # 合并到 market_df
        idx_cols = [COL_TRADE_DATE, "idx_ret", "idx_ret_5d", "idx_ret_20d",
                    "idx_ma5", "idx_ma20", "idx_ma60", "idx_vol_20d",
                    "fwd_ret_5", "fwd_ret_20", "fwd_dir_5", "fwd_dir_20",
                    "regime_label"]
        market_df = market_df.merge(
            mkt_source[[c for c in idx_cols if c in mkt_source.columns]],
            on=COL_TRADE_DATE, how="left"
        )

    # 去掉最开始缺少指数特征的日期
    market_df = market_df.dropna(subset=["idx_ret"])

    logger.info("市场级特征: %d 天 × %d 列", len(market_df), len(market_df.columns))
    return market_df


def walk_forward_market(
    market_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "fwd_dir_20",
    initial_train_days: int = 500,
    val_days: int = 60,
    step_days: int = 60,
    forward_days: int = 20,
    task: str = "binary",
) -> dict:
    """Walk-Forward 训练大盘评估模型。

    Args:
        market_df: 市场级特征（行=交易日）
        feature_cols: 特征列
        target_col: 目标列
        initial_train_days: 初始训练天数
        val_days: 验证天数
        step_days: 步进天数
        task: 'binary'（方向分类）或 'regression'（收益回归）

    Returns:
        results dict
    """
    df = market_df.sort_values(COL_TRADE_DATE).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c.endswith(("_mean", "_std", "_q25", "_q75"))
                        or c.startswith("idx_")]
        # 排除目标相关列
        exclude = {"idx_ret", "idx_ret_5d", "idx_ret_20d",
                    "fwd_ret_5", "fwd_ret_20", "fwd_dir_5", "fwd_dir_20",
                    "regime_label"}
        feature_cols = [c for c in feature_cols if c not in exclude]

    logger.info("Walk-Forward 大盘: %d 样本, %d 特征", len(df), len(feature_cols))

    dates = df[COL_TRADE_DATE].values
    n = len(dates)
    folds = []
    fold_models = []
    train_end = initial_train_days

    if task == "binary":
        params = dict(
            n_estimators=100, num_leaves=8, min_child_samples=30,
            learning_rate=0.05, verbosity=-1, force_col_wise=True,
            objective="binary", metric="binary_logloss",
        )
    else:
        params = dict(
            n_estimators=100, num_leaves=8, min_child_samples=30,
            learning_rate=0.05, verbosity=-1, force_col_wise=True,
        )

    fold_idx = 0
    while train_end + val_days <= n - forward_days:
        train_df = df.iloc[:train_end].dropna(subset=feature_cols + [target_col])
        val_start = train_end
        val_end = train_end + val_days
        val_df = df.iloc[val_start:val_end].dropna(subset=feature_cols + [target_col])

        if len(train_df) < 100 or len(val_df) < 10:
            train_end += step_days
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        model = lgb.LGBMClassifier(**params) if task == "binary" else lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )

        # 验证
        y_pred = model.predict(X_val) if task == "binary" else model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if task == "binary" else y_pred

        # 准确率
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y_val, y_pred) if task == "binary" else 0
        try:
            auc = roc_auc_score(y_val, y_prob) if task == "binary" else 0
        except Exception:
            auc = 0

        # rule-based regime accuracy in same period
        regime_acc = 0
        if "regime_label" in val_df.columns:
            regime_map = {"bull": 1, "early_bull": 1, "rebound": 1,
                          "range_up": 1, "pullback": 0, "range_down": 0, "bear": 0,
                          "range": 0}
            regime_pred = val_df["regime_label"].map(
                lambda x: regime_map.get(str(x).lower(), 0) if isinstance(x, str) else 0
            )
            if len(regime_pred) > 0:
                regime_acc = accuracy_score(y_val, regime_pred)

        fold_result = {
            "fold": fold_idx + 1,
            "train_end": str(dates[min(train_end - 1, len(dates) - 1)]),
            "val_start": str(dates[min(val_start, len(dates) - 1)]),
            "val_end": str(dates[min(val_end - 1, len(dates) - 1)]),
            "accuracy": round(float(acc), 4),
            "auc": round(float(auc), 4),
            "regime_accuracy": round(float(regime_acc), 4),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "regime_beat": round(float(acc - regime_acc), 4),
        }
        folds.append(fold_result)
        fold_models.append(model)

        logger.info(
            "  折%d: 训练~%s | val %s~%s | acc=%.4f auc=%.4f | 规则=%.4f | beat=%+.4f | train=%d val=%d",
            fold_idx + 1,
            fold_result["train_end"][:10],
            fold_result["val_start"][:10],
            fold_result["val_end"][:10],
            acc, auc, regime_acc, acc - regime_acc,
            len(train_df), len(val_df),
        )

        fold_idx += 1
        train_end += step_days

    # 统计
    accs = [f["accuracy"] for f in folds]
    aucs = [f["auc"] for f in folds]
    reg_accs = [f["regime_accuracy"] for f in folds]

    overall = {
        "n_folds": len(folds),
        "mean_accuracy": float(np.mean(accs)) if accs else 0,
        "mean_auc": float(np.mean(aucs)) if aucs else 0,
        "mean_regime_accuracy": float(np.mean(reg_accs)) if reg_accs else 0,
        "accuracy_improvement": float(np.mean([f["regime_beat"] for f in folds])),
        "positive_ratio": float(np.mean([1 for a in accs if a > 0.5])) if accs else 0,
        "beat_regime_ratio": float(np.mean([1 for f in folds if f["regime_beat"] > 0])),
        "n_features": len(feature_cols),
        "target": target_col,
        "feature_cols": feature_cols,
    }

    # 最终模型（全量训练）
    logger.info("训练最终大盘模型（全量数据）...")
    full_df = df.dropna(subset=feature_cols + [target_col])
    X_full = full_df[feature_cols].values
    y_full = full_df[target_col].values

    if task == "binary":
        final_model = lgb.LGBMClassifier(**params)
        final_model.set_params(**{"n_estimators": 200})
    else:
        final_model = lgb.LGBMRegressor(**params)
        final_model.set_params(**{"n_estimators": 200})

    final_model.fit(X_full, y_full)

    # 特征重要性
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    result = {
        "model_type": "market",
        "task": task,
        "target": target_col,
        "folds": folds,
        "fold_models": fold_models,
        "overall": overall,
        "final_model": final_model,
        "final_importance": imp,
        "n_dates": n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info("大盘 Walk-Forward 完成: %d 折", len(folds))
    logger.info("  ML 平均准确率: %.4f", overall["mean_accuracy"])
    logger.info("  规则平均准确率: %.4f", overall["mean_regime_accuracy"])
    logger.info("  ML 超越规则: %.4f", overall["accuracy_improvement"])
    logger.info("  击败规则的比例: %.0f%%", overall["beat_regime_ratio"] * 100)

    return result


def predict_market(
    market_df: pd.DataFrame,
    feature_cols: list[str],
    model_path: str = MARKET_MODEL_PATH,
) -> pd.Series:
    """用训练好的大盘模型预测市场方向。"""
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = market_df[feature_cols].values
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=market_df.index, name="market_up_prob")


def save_market_model(result: dict, path: str = MARKET_MODEL_PATH) -> None:
    """保存大盘模型和报告。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    final_model = result.pop("final_model", None)
    fold_models = result.pop("fold_models", [])

    with open(path, "wb") as f:
        pickle.dump(final_model, f)

    # 保存报告
    report_path = path.replace(".pkl", "_report.json")
    report = {k: v for k, v in result.items() if k != "final_model"}
    report["model_path"] = path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 恢复
    result["final_model"] = final_model
    result["fold_models"] = fold_models

    logger.info("大盘模型已保存: %s", path)
    logger.info("报告: %s", report_path)


def load_market_model(path: str = MARKET_MODEL_PATH):
    """加载训练好的大盘模型。"""
    if not os.path.exists(path):
        logger.warning("大盘模型不存在: %s", path)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_model_training_date(report_path: str = None) -> str | None:
    """从报告 JSON 读取模型训练日期。"""
    if report_path is None:
        report_path = MARKET_MODEL_PATH.replace(".pkl", "_report.json")
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path) as f:
            report = json.load(f)
        return report.get("timestamp", "").split(" ")[0]
    except Exception:
        return None


_FEATURE_MATRIX_CACHE = {}


def _load_feature_matrix(feature_df=None) -> pd.DataFrame:
    """辅助：从参数或缓存加载特征矩阵（进程内缓存避免重复加载 16GB）。"""
    if feature_df is not None and not feature_df.empty and len(feature_df.columns) > 50:
        return feature_df
    if _FEATURE_MATRIX_CACHE.get("df") is not None:
        return _FEATURE_MATRIX_CACHE["df"]
    cache_path = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
    if os.path.exists(cache_path):
        logger.info("加载特征矩阵缓存...")
        df = pd.read_parquet(cache_path)
        _FEATURE_MATRIX_CACHE["df"] = df
        return df
    raise FileNotFoundError("特征矩阵缓存不存在")


def _get_latest_feature_date() -> pd.Timestamp | None:
    """获取最新数据日期（权威来源 = SQLite stock_daily，非 parquet 缓存）。

    2026-07-31 修复：原实现读 feature_matrix_v4.parquet 的 trade_date，
    但 parquet 无 cron 刷新会滞后，导致「数据日期 vs 模型日期」恒差 0~1 天，
    增量训练永远不触发。现改为读 DB，parquet 仅作回退。
    """
    from analysis.ml_common import get_latest_stock_daily_date, get_parquet_latest_date
    d = get_latest_stock_daily_date()
    if d is not None:
        return d
    # 回退：DB 不可用时才读 parquet（标注可能滞后）
    return get_parquet_latest_date(os.path.join(DATA_DIR, "feature_matrix_v4.parquet"))


def ensure_market_model_fresh(
    feature_df: pd.DataFrame | None = None,
    stock_daily_df: pd.DataFrame | None = None,
    max_stale_days: int = 3,
) -> None:
    """检查大盘模型新鲜度，落后最新数据超过阈值则增量训练。"""
    latest_data_date = _get_latest_feature_date()
    training_date_str = get_model_training_date()
    needs_retrain = True

    if training_date_str and latest_data_date is not None:
        training_date = pd.to_datetime(training_date_str)
        days_stale = (latest_data_date - training_date).days
        if days_stale <= max_stale_days:
            logger.info("大盘模型新鲜（差 %d 天），跳过重训练", days_stale)
            needs_retrain = False

    if needs_retrain:
        logger.info("大盘模型过期（最新数据 %s），增量训练...",
                    latest_data_date.date() if latest_data_date is not None else "?")
        feat = _load_feature_matrix(feature_df)
        market_df = build_market_features(feat, stock_daily_df=stock_daily_df)
        result = walk_forward_market(market_df, initial_train_days=500, val_days=60, step_days=60)
        save_market_model(result)
        logger.info("大盘模型增量训练完成")


def predict_market_today(
    feature_df: pd.DataFrame,
    stock_daily_df: pd.DataFrame,
) -> tuple[float, dict]:
    """用最新数据预测大盘方向。

    Returns:
        (up_probability, details_dict)
    """
    model = load_market_model()
    if model is None:
        logger.warning("大盘模型未训练，跳过预测")
        return 0.5, {"error": "no_model"}

    report_path = MARKET_MODEL_PATH.replace(".pkl", "_report.json")
    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception:
        report = {}

    feature_cols = report.get("feature_cols", []) or report.get("overall", {}).get("feature_cols", [])
    if not feature_cols:
        logger.warning("大盘模型无特征列信息")
        return 0.5, {"error": "no_features"}

    # 如果 feature_df 为空或只有几列，从 parquet 加载
    if feature_df is None or feature_df.empty or len(feature_df.columns) < 50:
        cache_path = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
        if os.path.exists(cache_path):
            logger.info("加载特征矩阵缓存用于大盘推理...")
            feature_df = pd.read_parquet(cache_path)
        else:
            logger.error("无特征矩阵可用")
            return 0.5, {"error": "no_feature_matrix"}

    market_df = build_market_features(feature_df, stock_daily_df=stock_daily_df)
    if market_df.empty:
        return 0.5, {"error": "empty_market_features"}
    latest = market_df.sort_values(COL_TRADE_DATE).iloc[-1:]

    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        logger.warning("大盘预测: 缺少 %d/%d 特征", len(missing), len(feature_cols))
        return 0.5, {"error": f"missing_{len(missing)}_features"}

    X = latest[feature_cols].values
    proba = model.predict_proba(X)[:, 1][0]

    # 构建详情
    details = {
        "ml_up_prob": round(float(proba), 4),
        "ml_direction": "看涨" if proba > 0.5 else "看跌",
        "ml_confidence": "高" if abs(proba - 0.5) > 0.15 else ("中" if abs(proba - 0.5) > 0.05 else "低"),
        "model_accuracy": round(report.get("overall", {}).get("mean_accuracy", 0), 4),
        "model_auc": round(report.get("overall", {}).get("mean_auc", 0), 4),
        "training_date": report.get("timestamp", "unknown"),
    }
    logger.info("大盘预测: %s (p=%.1f%%)", details["ml_direction"], proba * 100)
    return proba, details
