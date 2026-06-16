"""
个股推算行业指标模块
- 基于个股日线数据 + 行业-个股映射，自下而上整合 L2 行业指标
- 行业指标与个股数据时效性一致（不同于独立获取的官方行业指数）
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


def _compute_stock_momentum(prices: pd.Series) -> float:
    """计算单只个股的 20 日动量。"""
    if len(prices) < MOMENTUM_LOOKBACK + 1:
        return np.nan
    close_today = prices.iloc[-1]
    close_N_ago = prices.iloc[-(MOMENTUM_LOOKBACK + 1)]
    if close_N_ago <= 0:
        return np.nan
    return (close_today / close_N_ago - 1) * 100


def _is_above_ma(prices: pd.Series, period: int | None = None) -> bool | None:
    """判断最新收盘价是否站上 MA(N)。"""
    if period is None:
        period = STOCK_DERIVED_MA_PERIOD
    if len(prices) < period:
        return None
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

            mom = _compute_stock_momentum(prices)
            above_ma = _is_above_ma(prices)

            if not np.isnan(mom):
                stock_metrics.append({
                    "ts_code": ts_code,
                    "l2_code": stock_to_l2[ts_code],
                    "return_20d": mom,
                    "above_ma20": above_ma,
                })

        if not stock_metrics:
            result["status"] = "degraded"
            result["error"] = "没有足够个股数据计算行业指标"
            return result

        metrics_df = pd.DataFrame(stock_metrics)

        # 按 L2 行业聚合
        l2_records = []
        for l2_code, group in metrics_df.groupby("l2_code"):
            returns = group["return_20d"].dropna()
            above_ma_series = group["above_ma20"].dropna()

            if len(returns) < STOCK_DERIVED_MIN_STOCKS:
                continue

            l2_records.append({
                "l2_code": l2_code,
                "l2_name": l2_names.get(l2_code, l2_code),
                "stock_count": len(returns),
                "avg_return_20d": round(float(returns.mean()), 2),
                "median_return_20d": round(float(returns.median()), 2),
                "pct_positive": round(float((returns > 0).sum() / len(returns) * 100), 1),
                "pct_above_ma20": round(float(above_ma_series.sum() / len(above_ma_series) * 100), 1)
                    if len(above_ma_series) > 0 else 0,
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

        logger.info(
            "个股推算行业指标: %d 个 L2 行业, avg_return 范围 [%.1f, %.1f]",
            len(df),
            df["avg_return_20d"].min(),
            df["avg_return_20d"].max(),
        )

    except Exception as e:
        logger.error("个股推算行业指标失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result
