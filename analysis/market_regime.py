"""
宏观状态机
- 基于行业等权合成指数的 MA50/MA200 + ADX(14) 判定市场状态
- 输出: BULL / RANGE / BEAR + 建议仓位上限
"""

import logging

import numpy as np
import pandas as pd

from config import (
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
)

logger = logging.getLogger(__name__)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算 ADX 指标。

    Args:
        high, low, close: 价格序列（按时间升序）
        period: ADX 周期，默认 14

    Returns:
        ADX 序列
    """
    n = len(close)
    if n < period + 1:
        return pd.Series(np.full(n, np.nan), index=close.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    # Wilder 平滑
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx


def determine_regime(
    daily_df: pd.DataFrame,
    ma_fast: int = 50,
    ma_slow: int = 200,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> dict:
    """判定当前市场状态。

    使用所有 L1 行业等权合成市场指数，计算 MA 和 ADX。

    Args:
        daily_df: 行业日线数据

    Returns:
        dict: {
            "regime": "BEAR" | "RANGE" | "BULL",
            "adx": float,
            "ma50": float, "ma200": float,
            "price": float,
            "position_advice": "建议X成仓位",
            "details": str,
        }
    """
    result = {
        "regime": "RANGE",
        "adx": 0.0,
        "ma50": 0.0,
        "ma200": 0.0,
        "price": 0.0,
        "position_advice": "建议5成仓位",
        "details": "",
    }

    try:
        if daily_df is None or daily_df.empty:
            return result

        # 等权合成市场指数：每日取所有行业收盘价均值
        daily_df = daily_df.sort_values(COL_TRADE_DATE).copy()
        market = daily_df.groupby(COL_TRADE_DATE)[COL_CLOSE].mean()

        if len(market) < ma_slow + 1:
            logger.warning("数据不足，无法判定 regime（需 >%d 天）", ma_slow)
            return result

        price = market.iloc[-1]
        ma50 = market.rolling(ma_fast, min_periods=ma_fast).mean().iloc[-1]
        ma200 = market.rolling(ma_slow, min_periods=ma_slow).mean().iloc[-1]

        # ADX 需 high/low 数据，简单用 close 近似
        # 取所有行业的 high/low 均值
        avg_high = daily_df.groupby(COL_TRADE_DATE)[COL_HIGH].mean()
        avg_low = daily_df.groupby(COL_TRADE_DATE)[COL_LOW].mean()
        adx_series = compute_adx(avg_high, avg_low, market, adx_period)
        adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0

        # 状态判定
        above_ma200 = price > ma200
        trend_strong = adx >= adx_threshold

        if trend_strong and above_ma200:
            regime = "BULL"
            position = "建议7-8成仓位"
            detail = f"ADX={adx:.1f}(强趋势) | 价格>{ma200:.0f}(MA200之上)"
        elif trend_strong and not above_ma200:
            regime = "BEAR"
            position = "建议2-3成仓位"
            detail = f"ADX={adx:.1f}(强趋势) | 价格<{ma200:.0f}(MA200之下)"
        else:
            regime = "RANGE"
            position = "建议5成仓位"
            detail = f"ADX={adx:.1f}(弱趋势) | 价格{'>{:.0f}'.format(ma200) if above_ma200 else '<{:.0f}'.format(ma200)}"

        result.update({
            "regime": regime,
            "adx": round(adx, 1),
            "ma50": round(float(ma50), 0),
            "ma200": round(float(ma200), 0),
            "price": round(float(price), 0),
            "position_advice": position,
            "details": detail,
        })

        logger.info("Regime: %s | ADX=%.1f MA50=%.0f MA200=%.0f Price=%.0f",
                     regime, adx, ma50, ma200, price)

    except Exception as e:
        logger.error("Regime 判定失败: %s", e)

    return result
