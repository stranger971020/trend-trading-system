"""
模块0: 三级行业领先信号
- 计算每个三级行业相对其所属一级行业的超额动量
- 输出 "三级领先 → 一级确认" 信号强度排名
- 用途：发现微观行业层面的领先信号，增强模块1的择时判断
"""

import logging

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_LOOKBACK,
    L3_LEADING_THRESHOLD,
    L3_STRONG_LEADING,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_NAME,
)

logger = logging.getLogger(__name__)


def compute_excess_momentum(
    l3_prices: pd.Series,
    l1_prices: pd.Series,
    lookback: int = MOMENTUM_LOOKBACK,
) -> float:
    """计算三级行业相对一级行业的超额动量（百分比）。"""
    if len(l3_prices) < lookback + 1 or len(l1_prices) < lookback + 1:
        return 0.0

    l3_ret = (l3_prices.iloc[-1] / l3_prices.iloc[-(lookback + 1)] - 1) * 100
    l1_ret = (l1_prices.iloc[-1] / l1_prices.iloc[-(lookback + 1)] - 1) * 100
    return l3_ret - l1_ret


def analyze_l3_leading(
    l3_daily_df: pd.DataFrame,
    l1_daily_df: pd.DataFrame,
) -> dict:
    """分析三级行业领先信号。

    对每个有数据的三级行业，计算其相对父级一级行业的超额动量。

    Args:
        l3_daily_df: 三级行业日线数据（含 parent_l1 列）
        l1_daily_df: 一级行业日线数据

    Returns:
        dict: {
            "status": "success" | "degraded" | "failed",
            "df": pd.DataFrame (L3 ranking by excess momentum),
            "leading_count": N,
            "by_l1": {parent_name: [L3列表], ...},
        }
    """
    result = {
        "status": "failed",
        "df": pd.DataFrame(),
        "leading_count": 0,
        "strong_leading": [],
        "by_l1": {},
        "error": None,
    }

    try:
        if l3_daily_df is None or l3_daily_df.empty:
            result["status"] = "degraded"
            result["error"] = "三级行业数据为空"
            return result

        if l1_daily_df is None or l1_daily_df.empty:
            result["status"] = "degraded"
            result["error"] = "一级行业数据为空"
            return result

        # 构建 L1 价格索引
        l1_prices_map = {}
        for l1_code, group in l1_daily_df.groupby(COL_TS_CODE):
            group = group.sort_values(COL_TRADE_DATE)
            if len(group) >= MOMENTUM_LOOKBACK + 1:
                l1_prices_map[l1_code] = group[COL_CLOSE]

        # 计算每个 L3 的超额动量
        records = []
        for l3_code, group in l3_daily_df.groupby(COL_TS_CODE):
            group = group.sort_values(COL_TRADE_DATE)
            if len(group) < MOMENTUM_LOOKBACK + 1:
                continue

            parent_l1 = group["parent_l1"].iloc[0] if "parent_l1" in group.columns else ""
            parent_name = group["parent_name"].iloc[0] if "parent_name" in group.columns else ""
            l3_name = group[COL_NAME].iloc[0] if COL_NAME in group.columns else l3_code

            if not parent_l1 or parent_l1 not in l1_prices_map:
                continue

            l3_prices = group[COL_CLOSE]
            l1_prices = l1_prices_map[parent_l1]

            excess = compute_excess_momentum(l3_prices, l1_prices)
            l3_return = (l3_prices.iloc[-1] / l3_prices.iloc[-(MOMENTUM_LOOKBACK + 1)] - 1) * 100

            if excess >= L3_STRONG_LEADING:
                label = "🔥强烈领先"
            elif excess >= L3_LEADING_THRESHOLD:
                label = "⚡领先"
            elif excess >= 0:
                label = "同步"
            else:
                label = "落后"

            records.append({
                "l3_code": l3_code,
                "l3_name": l3_name,
                "parent_l1": parent_l1,
                "parent_name": parent_name,
                "l3_return_20d": round(l3_return, 2),
                "excess_momentum": round(excess, 2),
                "label": label,
            })

        if not records:
            result["status"] = "degraded"
            result["error"] = "无有效三级行业数据"
            return result

        df = pd.DataFrame(records).sort_values("excess_momentum", ascending=False)
        df["rank"] = range(1, len(df) + 1)

        # 统计
        leading = df[df["excess_momentum"] >= L3_LEADING_THRESHOLD]
        strong = df[df["excess_momentum"] >= L3_STRONG_LEADING]

        # 按 L1 分组
        by_l1 = {}
        for _, row in leading.iterrows():
            pname = row["parent_name"] or row["parent_l1"]
            if pname not in by_l1:
                by_l1[pname] = []
            by_l1[pname].append({
                "l3_name": row["l3_name"],
                "excess": row["excess_momentum"],
                "l3_return": row["l3_return_20d"],
            })

        result["df"] = df
        result["leading_count"] = len(leading)
        result["strong_leading"] = [
            {"name": r["l3_name"], "excess": r["excess_momentum"], "parent": r["parent_name"]}
            for _, r in strong.iterrows()
        ]
        result["by_l1"] = by_l1
        result["status"] = "success"

        logger.info(
            "模块0: %d 个 L3 行业, %d 领先 (%d 强烈)",
            len(df), len(leading), len(strong),
        )

    except Exception as e:
        logger.error("模块0分析失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result
