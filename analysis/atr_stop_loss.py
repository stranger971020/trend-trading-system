"""
ATR 动态止损
- 计算个股 ATR(14)，输出建议止损价
- 止损价 = 最新收盘价 - 2×ATR
"""

import logging

import numpy as np
import pandas as pd

from config import (
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
)

logger = logging.getLogger(__name__)

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # 止损倍数


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """计算 ATR（Average True Range）。

    使用标准 Wilder 平滑公式。

    Args:
        high, low, close: 价格序列
        period: ATR 周期

    Returns:
        ATR 序列
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_stop_loss(
    stock_daily_df: pd.DataFrame,
    stock_picks: list[dict],
    multiplier: float = ATR_MULTIPLIER,
) -> list[dict]:
    """为精选个股计算 ATR 止损价。

    Args:
        stock_daily_df: 个股日线数据
        stock_picks: 精选个股列表（来自模块3）
        multiplier: ATR 倍数

    Returns:
        附加了 stop_loss_price 和 atr_pct 的个股列表
    """
    if stock_daily_df is None or stock_daily_df.empty:
        return stock_picks

    picks_with_stops = []
    for pick in stock_picks:
        ts_code = pick["ts_code"]
        sdf = stock_daily_df[stock_daily_df[COL_TS_CODE] == ts_code].sort_values(COL_TRADE_DATE)

        if sdf.empty or len(sdf) < ATR_PERIOD + 1:
            pick["stop_loss_price"] = None
            pick["atr_pct"] = None
            picks_with_stops.append(pick)
            continue

        atr_series = compute_atr(sdf[COL_HIGH], sdf[COL_LOW], sdf[COL_CLOSE])
        latest_atr = atr_series.iloc[-1]
        latest_close = sdf[COL_CLOSE].iloc[-1]

        if pd.isna(latest_atr) or latest_close <= 0:
            pick["stop_loss_price"] = None
            pick["atr_pct"] = None
        else:
            stop_price = latest_close - multiplier * latest_atr
            atr_pct = (latest_atr / latest_close) * 100
            pick["stop_loss_price"] = round(float(stop_price), 2)
            pick["atr_pct"] = round(float(atr_pct), 2)

        picks_with_stops.append(pick)

    valid = sum(1 for p in picks_with_stops if p.get("stop_loss_price") is not None)
    logger.info("ATR 止损: %d/%d 只个股计算完成", valid, len(picks_with_stops))
    return picks_with_stops
