from __future__ import annotations
"""
模块1: 市场情绪与择时
- 计算 31 个 SW L1 行业的平均动量
- 判定市场情绪（Bullish / Neutral / Bearish）
- 检测 MACD 顶背离（价格新高但 DIF 未新高 → 见顶预警）
"""

import logging

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_LOOKBACK,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    DIVERGENCE_LOOKBACK,
    PEAK_RADIUS,
    SENTIMENT_BULLISH,
    SENTIMENT_BEARISH,
    DIVERGENCE_WARNING_COUNT,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_NAME,
)

logger = logging.getLogger(__name__)


# ============================================================
# MACD 计算
# ============================================================

def compute_macd(prices: pd.Series) -> pd.DataFrame:
    """计算 MACD 指标。

    Args:
        prices: 按时间升序排列的收盘价序列

    Returns:
        DataFrame with columns: dif, dea, histogram
    """
    ema_fast = prices.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = prices.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = dif - dea
    return pd.DataFrame({"dif": dif, "dea": dea, "histogram": histogram})


# ============================================================
# 动量计算
# ============================================================

def compute_industry_momentum(prices: pd.Series, lookback: int = MOMENTUM_LOOKBACK) -> float:
    """计算单个行业的动量（百分比）。

    momentum = (close_today / close_N_days_ago - 1) * 100
    """
    if len(prices) < lookback + 1:
        return 0.0
    close_today = prices.iloc[-1]
    close_N_ago = prices.iloc[-(lookback + 1)]
    if close_N_ago <= 0:
        return 0.0
    return (close_today / close_N_ago - 1) * 100


# ============================================================
# MACD 顶背离检测
# ============================================================

def _find_local_peaks(series: pd.Series, radius: int = PEAK_RADIUS) -> list[int]:
    """在序列中查找局部极大值的索引位置。

    一个点被认为是局部峰，当它在 [i-radius, i+radius] 范围内是最大值。
    """
    peaks = []
    n = len(series)
    for i in range(radius, n - radius):
        window = series.iloc[i - radius : i + radius + 1]
        if series.iloc[i] == window.max():
            # 确保不是平台（相邻点不等）
            if series.iloc[i] > series.iloc[i - 1] and series.iloc[i] > series.iloc[i + 1]:
                peaks.append(i)
    return peaks


def detect_macd_top_divergence(
    prices: pd.Series, lookback: int = DIVERGENCE_LOOKBACK
) -> bool:
    """检测 MACD 顶背离。

    在 lookback 窗口内：
    1. 找到至少 2 个价格局部峰
    2. 比较最近的两个峰：价格新高但 DIF 值更低 → 顶背离

    Args:
        prices: 按时间升序排列的收盘价序列
        lookback: 回看窗口长度（数据点）

    Returns:
        True 如果检测到顶背离
    """
    if len(prices) < lookback:
        return False

    # 取最近 lookback 个数据点
    recent_prices = prices.iloc[-lookback:]
    macd_df = compute_macd(recent_prices)
    dif = macd_df["dif"]

    price_peaks = _find_local_peaks(recent_prices)
    if len(price_peaks) < 2:
        return False

    # 取最近两个价格峰
    prev_peak_idx = price_peaks[-2]
    latest_peak_idx = price_peaks[-1]

    price_prev = recent_prices.iloc[prev_peak_idx]
    price_latest = recent_prices.iloc[latest_peak_idx]
    dif_prev = dif.iloc[prev_peak_idx]
    dif_latest = dif.iloc[latest_peak_idx]

    # 顶背离条件：价格创新高，但 DIF 没有创新高
    if price_latest > price_prev and dif_latest < dif_prev:
        return True

    return False


# ============================================================
# 主分析函数
# ============================================================

def analyze_sentiment(
    daily_df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> dict:
    """分析市场情绪与择时信号。

    Args:
        daily_df: 日线数据，含 trade_date, ts_code, name, close 列
        mapping: 行业代码→名称映射（可选，用于输出中文名）

    Returns:
        dict: {
            "status": "success" | "degraded" | "failed",
            "date": "20260613",
            "sentiment": "Bullish" | "Neutral" | "Bearish",
            "avg_momentum": 4.72,
            "bullish_count": 22,
            "bearish_count": 7,
            "neutral_count": 2,
            "divergence_warnings": ["食品饮料", "银行"],
            "position_advice": "建议7-8成仓位",
        }
    """
    result = {
        "status": "failed",
        "date": None,
        "sentiment": "N/A",
        "avg_momentum": 0.0,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "divergence_warnings": [],
        "position_advice": "无法判断",
        "error": None,
    }

    try:
        if daily_df is None or daily_df.empty:
            result["status"] = "degraded"
            result["error"] = "日线数据为空"
            return result

        # 获取最新交易日
        latest_date = daily_df[COL_TRADE_DATE].max()
        result["date"] = str(latest_date)

        # 获取最新日期的所有行业数据
        latest_df = daily_df[daily_df[COL_TRADE_DATE] == latest_date]

        # 按行业分组计算动量
        industry_momentums = {}
        industry_names = {}
        divergence_list = []

        for ts_code, group in daily_df.groupby(COL_TS_CODE):
            if len(group) < MOMENTUM_LOOKBACK + 1:
                continue

            # 确保按日期排序
            group = group.sort_values(COL_TRADE_DATE)
            prices = group[COL_CLOSE]

            # 计算动量
            mom = compute_industry_momentum(prices)
            industry_momentums[ts_code] = mom

            # 行业名称
            name = group[COL_NAME].iloc[0] if COL_NAME in group.columns else ts_code
            if mapping and ts_code in mapping:
                name = mapping[ts_code]
            industry_names[ts_code] = name

            # 检测 MACD 顶背离
            if detect_macd_top_divergence(prices):
                divergence_list.append(name)

        if not industry_momentums:
            result["status"] = "degraded"
            result["error"] = "没有足够数据计算动量"
            return result

        # 统计
        momentums = list(industry_momentums.values())
        avg_momentum = np.mean(momentums)
        result["avg_momentum"] = round(float(avg_momentum), 2)

        bullish_count = sum(1 for m in momentums if m > SENTIMENT_BULLISH)
        bearish_count = sum(1 for m in momentums if m < SENTIMENT_BEARISH)
        neutral_count = len(momentums) - bullish_count - bearish_count

        result["bullish_count"] = bullish_count
        result["bearish_count"] = bearish_count
        result["neutral_count"] = neutral_count

        # 情绪判定
        if avg_momentum > SENTIMENT_BULLISH:
            result["sentiment"] = "Bullish"
        elif avg_momentum < SENTIMENT_BEARISH:
            result["sentiment"] = "Bearish"
        else:
            result["sentiment"] = "Neutral"

        # 顶背离预警
        result["divergence_warnings"] = divergence_list

        # 仓位建议
        extra_warning = ""
        if len(divergence_list) >= DIVERGENCE_WARNING_COUNT:
            extra_warning = "（注意: 多行业顶背离，适当降低仓位）"

        if result["sentiment"] == "Bullish":
            result["position_advice"] = f"建议7-8成仓位{extra_warning}"
        elif result["sentiment"] == "Neutral":
            result["position_advice"] = f"建议5成仓位{extra_warning}"
        else:
            result["position_advice"] = f"建议2-3成仓位{extra_warning}"

        result["status"] = "success"
        logger.info(
            "模块1: 情绪=%s, 平均动量=%.2f%%, 上涨=%d, 下跌=%d, 背离=%d",
            result["sentiment"],
            avg_momentum,
            bullish_count,
            bearish_count,
            len(divergence_list),
        )

    except Exception as e:
        logger.error("模块1分析失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result
