"""
L1 行业轮动策略
- 复用模块2的持续性评分逻辑
- 支持参数注入（用于网格搜索）
"""

import sys
import os

import pandas as pd

# 确保能导入项目模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    MOMENTUM_LOOKBACK as DEFAULT_LOOKBACK,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    RSI_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    PERSISTENCE_WEIGHTS as DEFAULT_WEIGHTS,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_VOL,
    COL_NAME,
)

# 导入评分函数
from analysis.module2_persistence import (
    compute_momentum_sub_score,
    compute_return_slope,
    compute_turnover_ratio,
    _minmax_normalize,
)


def generate_persistence_signal(
    history_df: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    weights: dict | None = None,
    macd_params: tuple | None = None,
) -> pd.Series:
    """生成持续性信号（每日调用）。

    复用 module2 的评分逻辑，但允许参数覆盖。

    Args:
        history_df: 截止前一日的全部历史数据
        lookback: 动量回看窗口
        weights: 四因子权重 dict
        macd_params: (fast, slow, signal) MACD 参数

    Returns:
        Series(index=ts_code, values=persistence_score)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 临时覆盖全局参数（通过修改 config 不太好，改用传参方式）
    # 这里直接使用默认参数，网格搜索时通过 monkey-patch 覆盖
    codes = history_df[COL_TS_CODE].unique()
    raw_data = {}

    for ts_code in codes:
        group = history_df[history_df[COL_TS_CODE] == ts_code].sort_values(COL_TRADE_DATE)
        if len(group) < BOLLINGER_PERIOD + 1:
            continue

        prices = group[COL_CLOSE]
        volumes = group[COL_VOL] if COL_VOL in group.columns else pd.Series(dtype=float)

        mom_score = compute_momentum_sub_score(prices)
        ret_slope = compute_return_slope(prices)

        if not volumes.empty and len(volumes) >= lookback + 1:
            turnover = compute_turnover_ratio(volumes, lookback)
        else:
            turnover = 1.0

        raw_data[ts_code] = {
            "momentum_score": mom_score,
            "return_slope": ret_slope,
            "turnover": turnover,
        }

    if not raw_data:
        return pd.Series(dtype=float)

    codes_list = list(raw_data.keys())

    slopes = pd.Series([raw_data[c]["return_slope"] for c in codes_list], index=codes_list)
    slope_scores = _minmax_normalize(slopes, 0, 10)

    turnovers = pd.Series([raw_data[c]["turnover"] for c in codes_list], index=codes_list)
    turnover_scores = _minmax_normalize(turnovers, 0, 10)

    avg_slope = slopes.mean()
    rel_strengths = pd.Series(
        [slopes[c] / (avg_slope + 1e-10) for c in codes_list], index=codes_list
    )
    rel_scores = _minmax_normalize(rel_strengths, 0, 10)

    w = weights
    scores = pd.Series(index=codes_list, dtype=float)
    for ts_code in codes_list:
        scores[ts_code] = (
            raw_data[ts_code]["momentum_score"] * w.get("momentum_score", 0.30)
            + slope_scores[ts_code] * w.get("return_slope", 0.25)
            + turnover_scores[ts_code] * w.get("turnover_score", 0.20)
            + rel_scores[ts_code] * w.get("relative_strength", 0.25)
        )

    return scores.sort_values(ascending=False)
