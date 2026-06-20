"""
模块2: 板块持续性评分
- 对每个 SW L1 行业计算 0-10 持续性得分
- 4 因子加权：技术动量分 + 收益斜率 + 换手率分 + 相对强度
- 输出高/中/低持续性分类列表
"""

import logging

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_LOOKBACK,
    RETURN_SLOPE_LOOKBACK,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    RSI_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    PERSISTENCE_WEIGHTS,
    HIGH_PERSISTENCE,
    MEDIUM_PERSISTENCE,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_VOL,
    COL_NAME,
)

logger = logging.getLogger(__name__)


# ============================================================
# 子指标计算
# ============================================================

def compute_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> float:
    """计算 RSI(14) 最新值。

    Returns:
        float: RSI 值 (0-100)
    """
    if len(prices) < period + 1:
        return 50.0

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

    # 使用 Wilder 平滑（与参考代码一致）
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0 or np.isnan(avg_loss):
        return 100.0
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0, 100))


def compute_macd_signal_label(prices: pd.Series) -> str:
    """判断 MACD 金叉/死叉状态。

    Returns:
        "golden_cross" | "death_cross" | "neutral"
    """
    if len(prices) < MACD_SLOW + MACD_SIGNAL:
        return "neutral"

    ema_fast = prices.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = prices.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()

    current_dif = dif.iloc[-1]
    current_dea = dea.iloc[-1]
    prev_dif = dif.iloc[-2]
    prev_dea = dea.iloc[-2]

    if current_dif > current_dea and prev_dif <= prev_dea:
        return "golden_cross"
    elif current_dif < current_dea and prev_dif >= prev_dea:
        return "death_cross"
    else:
        return "neutral"


def compute_bollinger_position(prices: pd.Series) -> float:
    """计算布林带位置 (0-100)。

    位置 = (close - lower) / (upper - lower) * 100
    """
    if len(prices) < BOLLINGER_PERIOD:
        return 50.0

    ma = prices.rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).mean()
    std = prices.rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).std()
    upper = ma + BOLLINGER_STD * std
    lower = ma - BOLLINGER_STD * std

    close = prices.iloc[-1]
    upper_val = upper.iloc[-1]
    lower_val = lower.iloc[-1]

    if upper_val - lower_val < 1e-10:
        return 50.0
    return float(np.clip((close - lower_val) / (upper_val - lower_val) * 100, 0, 100))


def compute_trend_deviation(prices: pd.Series) -> float:
    """计算趋势偏离度（百分比）。

    td = (close - ma20) / ma20 * 100
    """
    if len(prices) < MOMENTUM_LOOKBACK:
        return 0.0
    ma20 = prices.rolling(window=MOMENTUM_LOOKBACK, min_periods=MOMENTUM_LOOKBACK).mean()
    close = prices.iloc[-1]
    ma20_val = ma20.iloc[-1]
    if ma20_val <= 0:
        return 0.0
    return float((close - ma20_val) / ma20_val * 100)


# ============================================================
# 四因子评分
# ============================================================

def compute_momentum_sub_score(prices: pd.Series) -> float:
    """计算技术动量综合分 (0-10)。

    四个子因子各 0-10，等权重：
    - RSI 分：极端方向得分高
    - MACD 分：金叉 8，死叉 2，中性 5
    - 布林位置分：30-70 中间区域 8，极端 5
    - 趋势偏离分：>2% 得 8，-2~2% 得 5，<-2% 得 3
    """
    rsi = compute_rsi(prices)
    macd_label = compute_macd_signal_label(prices)
    bb_pos = compute_bollinger_position(prices)
    td = compute_trend_deviation(prices)

    # RSI 分 (0-10)
    rsi_score = np.clip(5.0 + (rsi - 50.0) / 5.0, 0, 10)

    # MACD 分
    macd_score_map = {"golden_cross": 8.0, "death_cross": 2.0, "neutral": 5.0}
    macd_score = macd_score_map.get(macd_label, 5.0)

    # 布林位置分
    bb_score = 8.0 if 30 <= bb_pos <= 70 else 5.0

    # 趋势偏离分
    if td > 2.0:
        trend_score = 8.0
    elif td < -2.0:
        trend_score = 3.0
    else:
        trend_score = 5.0

    return rsi_score * 0.25 + macd_score * 0.25 + bb_score * 0.25 + trend_score * 0.25


def compute_momentum_stability(prices: pd.Series, lookback: int = MOMENTUM_LOOKBACK) -> float:
    """计算动量稳定性（20日回归R²，0-10）。"""
    if len(prices) < lookback + 1:
        return 5.0
    y = prices.iloc[-lookback:].values
    x = np.arange(len(y))
    if np.std(y) > 0 and len(y) > 1:
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    else:
        r2 = 0
    return r2 * 10


def compute_return_slope(prices: pd.Series) -> float:
    """计算收益斜率（百分比），回看窗口由 RETURN_SLOPE_LOOKBACK 配置。"""
    if len(prices) < RETURN_SLOPE_LOOKBACK + 1:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[-(RETURN_SLOPE_LOOKBACK + 1)] - 1) * 100)


def compute_turnover_ratio(volumes: pd.Series, lookback: int = MOMENTUM_LOOKBACK) -> float:
    """计算换手率比 = 今日成交量 / 20日均量。"""
    if len(volumes) < lookback + 1:
        return 1.0
    vol_today = volumes.iloc[-1]
    vol_ma = volumes.iloc[-(lookback + 1):-1].mean()
    if vol_ma <= 0:
        return 1.0
    return float(vol_today / vol_ma)


def _minmax_normalize(series: pd.Series, target_min: float = 0.0, target_max: float = 10.0) -> pd.Series:
    """Min-max 归一化到 [target_min, target_max]。"""
    s_min, s_max = series.min(), series.max()
    if s_max - s_min < 1e-10:
        return pd.Series(5.0, index=series.index)
    return target_min + (series - s_min) / (s_max - s_min) * (target_max - target_min)


# ============================================================
# 主分析函数
# ============================================================

def compute_persistence(
    daily_df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """计算所有行业的板块持续性得分。

    Args:
        daily_df: 日线数据
        mapping: 行业代码→名称映射

    Returns:
        DataFrame with columns:
            rank, ts_code, name, persistence_score, label,
            momentum_score, return_slope, turnover_score, relative_strength
        按 persistence_score 降序排列
    """
    if daily_df is None or daily_df.empty:
        logger.warning("日线数据为空，无法计算持续性")
        return pd.DataFrame()

    codes = daily_df[COL_TS_CODE].unique()
    records = []

    # 先计算所有行业的原始值（用于归一化）
    raw_data: dict[str, dict] = {}

    for ts_code in codes:
        group = daily_df[daily_df[COL_TS_CODE] == ts_code].sort_values(COL_TRADE_DATE)

        if len(group) < BOLLINGER_PERIOD + 1:
            logger.debug("%s 数据不足 %d 条，跳过", ts_code, len(group))
            continue

        prices = group[COL_CLOSE]
        volumes = group[COL_VOL] if COL_VOL in group.columns else pd.Series(dtype=float)

        name = group[COL_NAME].iloc[0] if COL_NAME in group.columns else ts_code
        if mapping and ts_code in mapping:
            name = mapping[ts_code]

        mom_score = compute_momentum_sub_score(prices)
        ret_slope = compute_return_slope(prices)
        stability = compute_momentum_stability(prices)

        if not volumes.empty and len(volumes) >= MOMENTUM_LOOKBACK + 1:
            turnover = compute_turnover_ratio(volumes)
        else:
            turnover = 1.0

        raw_data[ts_code] = {
            "name": name,
            "momentum_score": mom_score,
            "return_slope": ret_slope,
            "turnover": turnover,
            "stability_score": stability,
        }

    if not raw_data:
        return pd.DataFrame()

    codes_list = list(raw_data.keys())

    # 收益斜率 → 归一化到 0-10
    slopes = pd.Series([raw_data[c]["return_slope"] for c in codes_list], index=codes_list)
    slope_scores = _minmax_normalize(slopes, 0, 10)

    # 换手率 → 归一化到 0-10
    turnovers = pd.Series([raw_data[c]["turnover"] for c in codes_list], index=codes_list)
    turnover_scores = _minmax_normalize(turnovers, 0, 10)

    # 相对强度 = 行业收益 / 全市场均值（归一化到 0-10）
    avg_slope = slopes.mean()
    rel_strengths = pd.Series([
        slopes[c] / (avg_slope + 1e-10) for c in codes_list
    ], index=codes_list)
    rel_scores = _minmax_normalize(rel_strengths, 0, 10)

    # 加权合成（含稳定性因子）
    w = PERSISTENCE_WEIGHTS
    for ts_code in codes_list:
        persistence = (
            raw_data[ts_code]["momentum_score"] * w["momentum_score"]
            + slope_scores[ts_code] * w["return_slope"]
            + turnover_scores[ts_code] * w["turnover_score"]
            + rel_scores[ts_code] * w["relative_strength"]
            + raw_data[ts_code]["stability_score"] * w["stability_score"]
        )

        if persistence >= HIGH_PERSISTENCE:
            label = "🔥高持续性"
        elif persistence >= MEDIUM_PERSISTENCE:
            label = "⚡中等持续性"
        else:
            label = "⚠️低持续性"

        records.append({
            "ts_code": ts_code,
            "name": raw_data[ts_code]["name"],
            "persistence_score": round(persistence, 2),
            "label": label,
            "momentum_score": round(raw_data[ts_code]["momentum_score"], 2),
            "return_slope": round(slope_scores[ts_code], 2),
            "turnover_score": round(turnover_scores[ts_code], 2),
            "relative_strength": round(rel_scores[ts_code], 2),
            "stability_score": round(raw_data[ts_code]["stability_score"], 2),
            "return_20d_pct": round(raw_data[ts_code]["return_slope"], 2),
            "rsi": None,  # 可选：后续填充
        })

    df = pd.DataFrame(records).sort_values("persistence_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df = df.reset_index(drop=True)

    high_count = (df["persistence_score"] >= HIGH_PERSISTENCE).sum()
    medium_count = ((df["persistence_score"] >= MEDIUM_PERSISTENCE) & (df["persistence_score"] < HIGH_PERSISTENCE)).sum()
    low_count = (df["persistence_score"] < MEDIUM_PERSISTENCE).sum()
    logger.info(
        "模块2: %d 个行业评分完成 | 高:%d 中:%d 低:%d",
        len(df), high_count, medium_count, low_count,
    )

    return df


def analyze_persistence(
    daily_df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> dict:
    """模块2 包装函数，提供统一的返回格式。

    Returns:
        dict: {
            "status": "success" | "failed",
            "df": pd.DataFrame (行业排名),
            "high_persistence": [行业名称列表],
            "medium_persistence": [...],
            "low_persistence": [...],
            "error": None | str,
        }
    """
    result = {
        "status": "failed",
        "df": pd.DataFrame(),
        "high_persistence": [],
        "medium_persistence": [],
        "low_persistence": [],
        "error": None,
    }

    try:
        df = compute_persistence(daily_df, mapping)

        if df.empty:
            result["status"] = "degraded"
            result["error"] = "没有足够数据计算持续性"
            return result

        result["df"] = df
        result["high_persistence"] = df[df["label"] == "🔥高持续性"]["name"].tolist()
        result["medium_persistence"] = df[df["label"] == "⚡中等持续性"]["name"].tolist()
        result["low_persistence"] = df[df["label"] == "⚠️低持续性"]["name"].tolist()
        result["status"] = "success"

    except Exception as e:
        logger.error("模块2分析失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def analyze_l3_persistence(l3_daily_df: pd.DataFrame) -> dict:
    """模块2 L3 扩展：分析三级行业持续性。

    Returns 结构与 analyze_persistence 相同，额外多 l3 相关字段。
    """
    result = {
        "status": "failed",
        "df": pd.DataFrame(),
        "high_persistence": [],
        "medium_persistence": [],
        "low_persistence": [],
        "error": None,
    }

    try:
        df = compute_persistence(l3_daily_df, mapping=None)

        if df.empty:
            result["status"] = "degraded"
            result["error"] = "L3 数据不足"
            return result

        result["df"] = df
        result["high_persistence"] = df[df["label"] == "🔥高持续性"]["name"].tolist()
        result["medium_persistence"] = df[df["label"] == "⚡中等持续性"]["name"].tolist()
        result["low_persistence"] = df[df["label"] == "⚠️低持续性"]["name"].tolist()
        result["status"] = "success"

    except Exception as e:
        logger.error("模块2 L3 分析失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result
