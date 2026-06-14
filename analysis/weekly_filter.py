"""
周线滤波器
- 从日线合成周线（周五收盘）
- 计算周线动量得分，过滤短期噪音
- 日线信号需获得周线确认
"""

import logging

import pandas as pd

from config import (
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
)

logger = logging.getLogger(__name__)


def daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线（取每周最后一个交易日）。

    Returns:
        DataFrame with trade_date, ts_code, close (weekly)
    """
    daily_df = daily_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    daily_df[COL_TRADE_DATE] = pd.to_datetime(daily_df[COL_TRADE_DATE], format="%Y%m%d")

    # 按 ISO 周分组，取每周最后一天
    daily_df["week"] = daily_df[COL_TRADE_DATE].dt.isocalendar().week
    daily_df["year"] = daily_df[COL_TRADE_DATE].dt.isocalendar().year

    weekly = daily_df.groupby([COL_TS_CODE, "year", "week"]).last().reset_index()
    weekly[COL_TRADE_DATE] = weekly[COL_TRADE_DATE].dt.strftime("%Y%m%d")
    return weekly


def compute_weekly_momentum(
    weekly_df: pd.DataFrame,
    window: int = 4,
) -> pd.DataFrame:
    """计算每周的周线动量得分（简化版：4周=约1个月）。

    Returns:
        DataFrame with ts_code, momentum_score, sustained_weeks
    """
    if weekly_df is None or weekly_df.empty:
        return pd.DataFrame()

    codes = weekly_df[COL_TS_CODE].unique()
    records = []

    for code in codes:
        grp = weekly_df[weekly_df[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
        if len(grp) < window + 1:
            continue

        closes = grp[COL_CLOSE]

        # 4周动量 (%)
        mom_4w = (closes.iloc[-1] / closes.iloc[-(window + 1)] - 1) * 100

        # 转换为 0-10 得分
        score = 5.0 + mom_4w  # 0% 动量 → 5 分，每 1% → +1 分
        score = max(0, min(10, score))

        # 持续周数: 连续 score >= 6 的周数
        scores = pd.Series([
            max(0, min(10, 5.0 + (closes.iloc[i] / closes.iloc[i - window] - 1) * 100))
            for i in range(window, len(closes))
        ], index=closes.index[window:])

        sustained = 0
        for s in scores.iloc[::-1]:
            if s >= 6:
                sustained += 1
            else:
                break

        records.append({
            COL_TS_CODE: code,
            "weekly_momentum": round(score, 1),
            "sustained_weeks": sustained,
            "weekly_confirmed": score >= 6 and sustained >= 2,
        })

    df = pd.DataFrame(records)
    logger.info(
        "周线过滤: %d 行业, %d 周线确认",
        len(df), df["weekly_confirmed"].sum(),
    )
    return df


def apply_weekly_filter(
    persistence_df: pd.DataFrame,
    weekly_scores: pd.DataFrame,
) -> pd.DataFrame:
    """对板块持续性得分应用周线过滤。

    未通过周线确认的行业，持续性分降 1.5 分（惩罚），而非直接剔除。

    Returns:
        调整后的 persistence_df
    """
    if weekly_scores is None or weekly_scores.empty:
        return persistence_df

    df = persistence_df.copy()
    wmap = weekly_scores.set_index(COL_TS_CODE)

    penalty_count = 0
    for idx, row in df.iterrows():
        code = row[COL_TS_CODE]
        if code in wmap.index and not wmap.loc[code, "weekly_confirmed"]:
            df.at[idx, "persistence_score"] = max(0, row["persistence_score"] - 1.5)
            df.at[idx, "label"] = _relabel(df.at[idx, "persistence_score"])
            penalty_count += 1

    logger.info("周线惩罚: %d 个行业降权", penalty_count)
    return df


def _relabel(score: float) -> str:
    if score >= 7:
        return "🔥高持续性"
    elif score >= 5:
        return "⚡中等持续性"
    else:
        return "⚠️低持续性"
