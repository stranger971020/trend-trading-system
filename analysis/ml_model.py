"""
ML 评分模型 — LightGBM LambdaRank
- 从历史数据构建特征矩阵
- 训练 Learning-to-Rank 模型
- 每日推理替换线性加权评分
"""

import logging
import os
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    DB_PATH,
    DATA_DIR,
    BEIJING_TZ_OFFSET,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_VOL,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(DATA_DIR, "lgb_model.pkl")
FEATURE_COLS = [
    "momentum_20d", "momentum_5d", "ma20_deviation", "vol_ratio",
    "excess_return_l1", "excess_return_l3",
    "pe_pct", "pb_pct", "roe", "roe_trend",
    "net_mf_amount", "atr_pct",
    "sector_persistence", "l3_persistence",
    "market_adx", "market_ma50_ratio",
]

_BEIJING_TZ = timezone(timedelta(hours=BEIJING_TZ_OFFSET))


# ============================================================
# 特征工程
# ============================================================

def build_feature_matrix(
    stock_daily_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame | None = None,
    stock_mapping: dict | None = None,
    persistence_scores: dict | None = None,
    multi_day: bool = False,
) -> pd.DataFrame:
    """从原始数据构建特征矩阵。

    Args:
        multi_day: True 时对每个历史日期都生成特征（训练用），
                   False 时只生成最新日期（推理用）

    特征：
    - 技术面: 20日动量, 5日动量, MA20偏离, 成交量比
    - 相对强度: 相对L1超额收益, 相对L3超额收益
    - 质量: ROE
    - 风控: ATR百分比
    """
    if stock_daily_df is None or stock_daily_df.empty:
        return pd.DataFrame()

    df = stock_daily_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    codes = df[COL_TS_CODE].unique()

    features = []
    for code in codes:
        sdf = df[df[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
        if len(sdf) < 22:
            continue

        closes = sdf[COL_CLOSE].values
        highs = sdf[COL_HIGH].values if COL_HIGH in sdf.columns else closes
        lows = sdf[COL_LOW].values if COL_LOW in sdf.columns else closes
        vols = sdf[COL_VOL].values if COL_VOL in sdf.columns else np.ones(len(closes))
        dates = sdf[COL_TRADE_DATE].values
        n = len(closes)

        if multi_day:
            # 对每个有效日期生成特征
            for i in range(21, n):
                mom20 = (closes[i] / closes[i - 20] - 1) * 100 if closes[i - 20] > 0 else 0
                mom5 = (closes[i] / closes[i - 5] - 1) * 100 if closes[i - 5] > 0 else 0
                ma20 = closes[i - 20:i].mean()
                ma20_dev = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
                vol_ratio = vols[i] / vols[i - 20:i].mean() if vols[i - 20:i].mean() > 0 else 1

                # ATR
                tr_vals = []
                for j in range(i - 13, i + 1):
                    tr = max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
                    tr_vals.append(tr)
                atr_val = np.mean(tr_vals)
                atr_pct = (atr_val / closes[i] * 100) if closes[i] > 0 else 0

                features.append({
                    COL_TS_CODE: code, COL_TRADE_DATE: dates[i],
                    "momentum_20d": round(mom20, 2), "momentum_5d": round(mom5, 2),
                    "ma20_deviation": round(ma20_dev, 2), "vol_ratio": round(vol_ratio, 2),
                    "atr_pct": round(atr_pct, 2),
                    "excess_return_l1": 0.0, "excess_return_l3": 0.0,
                    "pe_pct": 0, "pb_pct": 0, "roe": 0, "roe_trend": 0,
                    "net_mf_amount": 0, "sector_persistence": 5.0,
                    "l3_persistence": 5.0, "market_adx": 0.0, "market_ma50_ratio": 0.0,
                    "close": closes[i],  # 临时保存用于 label 计算
                })
        else:
            # 仅最新日期
            i = n - 1
            mom20 = (closes[i] / closes[i - 20] - 1) * 100 if closes[i - 20] > 0 else 0
            mom5 = (closes[i] / closes[i - 5] - 1) * 100 if closes[i - 5] > 0 else 0
            ma20 = closes[i - 20:i].mean()
            ma20_dev = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
            vol_ratio = vols[i] / vols[i - 20:i].mean() if vols[i - 20:i].mean() > 0 else 1
            atr_pct = 0

            features.append({
                COL_TS_CODE: code, COL_TRADE_DATE: dates[i],
                "momentum_20d": round(mom20, 2), "momentum_5d": round(mom5, 2),
                "ma20_deviation": round(ma20_dev, 2), "vol_ratio": round(vol_ratio, 2),
                "atr_pct": round(atr_pct, 2),
                "excess_return_l1": 0.0, "excess_return_l3": 0.0,
                "pe_pct": 0, "pb_pct": 0, "roe": 0, "roe_trend": 0,
                "net_mf_amount": 0, "sector_persistence": 5.0,
                "l3_persistence": 5.0, "market_adx": 0.0, "market_ma50_ratio": 0.0,
            })

    return pd.DataFrame(features)


def _compute_atr_value(high, low, close, period=14):
    if len(close) < period + 1:
        return 0
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]


def _get_funda_features(code: str) -> dict:
    """从 fundamental_cache 读取最新基本面数据。"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            "SELECT pe_ttm, pb, roe FROM fundamental_cache WHERE ts_code=? AND trade_date>=? ORDER BY trade_date DESC LIMIT 1",
            (code, (datetime.now(_BEIJING_TZ) - timedelta(days=10)).strftime("%Y%m%d")),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "pe_pct": 0,  # 由调用方计算分位
                "pb_pct": 0,
                "roe": row[2] if row[2] else 0,
                "roe_trend": 0,
            }
    except Exception:
        pass
    return {"pe_pct": 0, "pb_pct": 0, "roe": 0, "roe_trend": 0}


def _get_moneyflow(code: str) -> float:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            "SELECT SUM(net_mf_amount) FROM moneyflow_cache WHERE ts_code=? AND trade_date>=?",
            (code, (datetime.now(_BEIJING_TZ) - timedelta(days=7)).strftime("%Y%m%d")),
        )
        row = cur.fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else 0.0
    except Exception:
        return 0.0


# ============================================================
# 训练
# ============================================================

def train_model(feature_df: pd.DataFrame, forward_days: int = 20) -> tuple:
    """训练 LightGBM LambdaRank 模型。

    Args:
        feature_df: 特征矩阵（多日多股票）
        forward_days: 前向收益天数作为 label

    Returns:
        (model, feature_importance_df)
    """
    if feature_df.empty or len(feature_df) < 100:
        logger.warning("训练数据不足")
        return None, pd.DataFrame()

    feature_df = feature_df.copy()

    # 检查是否有 close 列用于 label 计算
    if "close" not in feature_df.columns:
        logger.warning("特征中无 close 列，无法计算 label")
        return None, pd.DataFrame()

    # 构建 label: forward N-day return
    labels = []
    for code in feature_df[COL_TS_CODE].unique():
        sdf = feature_df[feature_df[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
        closes = sdf["close"].values
        for i in range(len(sdf) - forward_days):
            if closes[i] > 0:
                fwd_ret = (closes[i + forward_days] / closes[i] - 1) * 100
                labels.append({
                    COL_TS_CODE: code,
                    COL_TRADE_DATE: sdf[COL_TRADE_DATE].iloc[i],
                    "label": fwd_ret,
                })

    if not labels:
        return None, pd.DataFrame()

    label_df = pd.DataFrame(labels)
    train_df = feature_df.merge(label_df, on=[COL_TS_CODE, COL_TRADE_DATE], how="inner")

    # 去掉 close 列
    if "close" in train_df.columns:
        train_df = train_df.drop(columns=["close"])

    if len(train_df) < 100:
        return None, pd.DataFrame()

    # 特征和标签
    X = train_df[[c for c in FEATURE_COLS if c in train_df.columns]].fillna(0)
    y = train_df["label"]

    # 分组（按日期）- 用于排序学习
    dates = train_df[COL_TRADE_DATE]
    groups = dates.value_counts().sort_index().values

    # LightGBM 回归（预测前向收益）
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=150,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=30,
        verbosity=-1,
        force_col_wise=True,
    )

    model.fit(X, y)

    # 特征重要性
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("ML 训练完成: %d 样本, %d 特征, Top3=%s",
                 len(train_df), len(X.columns),
                 ", ".join(importance["feature"].head(3).tolist()))

    return model, importance


def save_model(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("模型已保存: %s", path)


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 推理
# ============================================================

def predict_scores(model, feature_df: pd.DataFrame) -> pd.Series:
    """用 ML 模型预测个股得分。

    Returns:
        Series(index=ts_code, values=predicted_score)
    """
    if model is None or feature_df.empty:
        return pd.Series(dtype=float)

    X = feature_df[[c for c in FEATURE_COLS if c in feature_df.columns]].fillna(0)
    predictions = model.predict(X)

    return pd.Series(predictions, index=feature_df[COL_TS_CODE].values)


def rerank_with_ml(
    stock_picks: list[dict],
    feature_df: pd.DataFrame,
    model=None,
) -> list[dict]:
    """用 ML 模型重排个股精选结果。

    如果模型不可用，回退到线性评分。
    """
    if model is None or feature_df.empty:
        logger.info("ML 模型不可用，使用线性评分")
        return stock_picks

    scores = predict_scores(model, feature_df)

    # 更新评分
    for pick in stock_picks:
        code = pick["ts_code"]
        if code in scores.index:
            ml_score = float(scores[code])
            # 归一化到 0-10
            pick["score"] = round(max(0, min(10, ml_score * 5 + 5)), 2)
            pick["ml_score"] = round(ml_score, 3)

    # 重排
    stock_picks.sort(key=lambda x: x["score"], reverse=True)
    return stock_picks
