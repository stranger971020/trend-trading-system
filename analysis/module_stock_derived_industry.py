from __future__ import annotations
"""
个股推算行业指标模块（含反转动能检测）
- 基于个股日线数据 + 行业-个股映射，自下而上整合 L2 行业指标
- 新增：5 日动量计算 → 反转强度 = 短期 - 中期，标记反转候选
"""

import logging

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_LOOKBACK,
    STOCK_DERIVED_MA_PERIOD,
    STOCK_DERIVED_MIN_STOCKS,
    STOCK_DERIVED_TOP_N,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
)

logger = logging.getLogger(__name__)

# 反转检测参数
REVERSAL_LOOKBACK_SHORT = 5   # 短期动量窗口
REVERSAL_STRENGTH_MIN = 5.0   # 反转强度下限

_SHORT_WINDOW = REVERSAL_LOOKBACK_SHORT


def _compute_stock_momentum(prices: pd.Series, period: int = MOMENTUM_LOOKBACK) -> float:
    """计算单只个股的 N 日动量。"""
    if len(prices) < period + 1:
        return np.nan
    close_today = prices.iloc[-1]
    close_N_ago = prices.iloc[-(period + 1)]
    if close_N_ago <= 0:
        return np.nan
    return (close_today / close_N_ago - 1) * 100


def _is_above_ma(prices: pd.Series, period: int = 0) -> bool:
    """判断最新收盘价是否站上 MA(N)。"""
    if period <= 0:
        period = STOCK_DERIVED_MA_PERIOD
    if len(prices) < period:
        return False
    ma = prices.iloc[-period:].mean()
    return prices.iloc[-1] > ma


def analyze_stock_derived_industry(
    stock_daily_df: pd.DataFrame,
    stock_mapping: dict,
) -> dict:
    """基于个股数据推算 L2 行业指标。

    Args:
        stock_daily_df: 个股日线数据，列含 trade_date, ts_code, close
        stock_mapping: {stock_code: {l2_code, l2_name, ...}}

    Returns:
        dict: {
            "status": "success" | "degraded" | "failed",
            "df": pd.DataFrame (columns: l2_code, l2_name, stock_count,
                  avg_return_20d, median_return_20d, pct_positive, pct_above_ma20),
            "l2_count": int,
            "error": str | None,
        }
    """
    result = {
        "status": "failed",
        "df": pd.DataFrame(),
        "l2_count": 0,
        "error": None,
    }

    try:
        if stock_daily_df is None or stock_daily_df.empty:
            result["status"] = "degraded"
            result["error"] = "个股日线数据为空"
            return result

        if not stock_mapping:
            result["status"] = "degraded"
            result["error"] = "个股行业映射为空"
            return result

        # 构建 stock_code → l2_code 快速查找
        stock_to_l2 = {}
        l2_names = {}
        for code, info in stock_mapping.items():
            l2_code = info.get("l2_code", "")
            if l2_code:
                stock_to_l2[code] = l2_code
                l2_names[l2_code] = info.get("l2_name", l2_code)

        # 按个股分组，计算每只个股的动量
        stock_metrics = []
        for ts_code, group in stock_daily_df.groupby(COL_TS_CODE):
            if ts_code not in stock_to_l2:
                continue
            if len(group) < MOMENTUM_LOOKBACK + 1:
                continue

            group = group.sort_values(COL_TRADE_DATE)
            prices = group[COL_CLOSE]

            mom20d = _compute_stock_momentum(prices, period=MOMENTUM_LOOKBACK)
            mom5d = _compute_stock_momentum(prices, period=_SHORT_WINDOW)
            above_ma20 = _is_above_ma(prices, period=MOMENTUM_LOOKBACK)
            above_ma5 = _is_above_ma(prices, period=_SHORT_WINDOW)

            if not np.isnan(mom20d):
                stock_metrics.append({
                    "ts_code": ts_code,
                    "l2_code": stock_to_l2[ts_code],
                    "return_20d": mom20d,
                    "return_5d": mom5d,
                    "above_ma20": above_ma20,
                    "above_ma5": above_ma5,
                })

        if not stock_metrics:
            result["status"] = "degraded"
            result["error"] = "没有足够个股数据计算行业指标"
            return result

        metrics_df = pd.DataFrame(stock_metrics)

        # 按 L2 行业聚合
        l2_records = []
        for l2_code, group in metrics_df.groupby("l2_code"):
            ret20 = group["return_20d"].dropna()
            ret5 = group["return_5d"].dropna()
            above_ma20_s = group["above_ma20"].dropna()
            above_ma5_s = group["above_ma5"].dropna()

            if len(ret20) < STOCK_DERIVED_MIN_STOCKS:
                continue

            avg_20d = float(ret20.mean())
            avg_5d = float(ret5.mean()) if len(ret5) > 0 else 0.0
            reversal_strength = round(avg_5d - avg_20d, 2)

            # 反转候选：20日动量为负 & 短期明显强于中期
            is_reversal = bool(avg_20d < 0 and reversal_strength > REVERSAL_STRENGTH_MIN)

            l2_records.append({
                "l2_code": l2_code,
                "l2_name": l2_names.get(l2_code, l2_code),
                "stock_count": len(ret20),
                "avg_return_20d": round(avg_20d, 2),
                "median_return_20d": round(float(ret20.median()), 2),
                "avg_return_5d": round(avg_5d, 2),
                "reversal_strength": reversal_strength,
                "is_reversal": is_reversal,
                "pct_positive": round(float((ret20 > 0).sum() / len(ret20) * 100), 1),
                "pct_above_ma20": round(float(above_ma20_s.sum() / len(above_ma20_s) * 100), 1)
                    if len(above_ma20_s) > 0 else 0,
                "pct_above_ma5": round(float(above_ma5_s.sum() / len(above_ma5_s) * 100), 1)
                    if len(above_ma5_s) > 0 else 0,
            })

        if not l2_records:
            result["status"] = "degraded"
            result["error"] = "L2 行业成分股不足（均<5只）"
            return result

        df = pd.DataFrame(l2_records)
        df = df.sort_values("avg_return_20d", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        result["status"] = "success"
        result["df"] = df
        result["l2_count"] = len(df)
        result["reversal_count"] = int(df["is_reversal"].sum())
        result["reversal_df"] = df[df["is_reversal"]].sort_values(
            "reversal_strength", ascending=False
        ).reset_index(drop=True) if result["reversal_count"] > 0 else pd.DataFrame()

        logger.info(
            "个股推算行业指标: %d 个 L2 行业, avg_return [%.1f, %.1f], 反转候选 %d 个",
            len(df),
            df["avg_return_20d"].min(),
            df["avg_return_20d"].max(),
            result["reversal_count"],
        )

    except Exception as e:
        logger.error("个股推算行业指标失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result
