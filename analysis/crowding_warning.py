"""
行业拥挤度预警
- 计算行业成交额占全市场比例
- 滚动 80 日分位数
- 拥挤 + 动量回落 → 预警
"""

import logging

import numpy as np
import pandas as pd

from config import (
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_AMOUNT,
)

logger = logging.getLogger(__name__)

CROWDING_PERCENTILE = 90       # 拥挤分位数阈值
CROWDING_LOOKBACK = 80          # 回看天数


def detect_crowding(
    daily_df: pd.DataFrame,
    percentile_threshold: int = CROWDING_PERCENTILE,
    lookback: int = CROWDING_LOOKBACK,
) -> dict:
    """检测行业拥挤度。

    Args:
        daily_df: 行业日线数据（含 amount 列）

    Returns:
        dict: {
            "crowded_industries": [行业名称列表],
            "details": [{"name": "电子", "percentile": 95, "momentum_5d": -2.1}, ...],
        }
    """
    result = {
        "crowded_industries": [],
        "details": [],
    }

    try:
        if daily_df is None or daily_df.empty:
            return result

        if COL_AMOUNT not in daily_df.columns:
            logger.warning("缺少成交额数据，跳过拥挤度检测")
            return result

        daily_df = daily_df.sort_values(COL_TRADE_DATE).copy()

        # 每日全市场总成交额
        market_amount = daily_df.groupby(COL_TRADE_DATE)[COL_AMOUNT].sum()

        # 每个行业每日成交额占比
        daily_df["market_total"] = daily_df[COL_TRADE_DATE].map(market_amount)
        daily_df["amount_ratio"] = daily_df[COL_AMOUNT] / (daily_df["market_total"] + 1e-10)

        # 滚动分位数
        codes = daily_df[COL_TS_CODE].unique()
        for code in codes:
            grp = daily_df[daily_df[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
            if len(grp) < lookback:
                continue

            ratios = grp["amount_ratio"]
            latest_ratio = ratios.iloc[-1]

            # 近 80 日分位数
            hist = ratios.iloc[-(lookback + 1):-1]
            percentile = (hist < latest_ratio).sum() / len(hist) * 100

            # 5日动量
            prices = grp[COL_CLOSE]
            if len(prices) >= 6:
                mom_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) * 100
            else:
                mom_5d = 0

            name_col = grp["name"].iloc[0] if "name" in grp.columns else code

            if percentile >= percentile_threshold and mom_5d < 0:
                result["crowded_industries"].append(str(name_col))
                result["details"].append({
                    "name": str(name_col),
                    "code": code,
                    "percentile": round(percentile, 1),
                    "momentum_5d": round(mom_5d, 2),
                })

        if result["crowded_industries"]:
            logger.info("拥挤度预警: %d 个行业 — %s",
                         len(result["crowded_industries"]),
                         ", ".join(result["crowded_industries"][:5]))
        else:
            logger.info("拥挤度: 无预警")

    except Exception as e:
        logger.error("拥挤度检测失败: %s", e)

    return result
