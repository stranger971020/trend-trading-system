"""
通用回测引擎 (v2 — 向量化)
- 预计算每日每行业的得分矩阵
- 组合模拟极快（纯 numpy）
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果数据结构。"""
    nav: pd.Series
    benchmark_nav: pd.Series
    daily_returns: pd.Series
    benchmark_returns: pd.Series

    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0
    win_rate_daily: float = 0.0
    win_rate_monthly: float = 0.0
    profit_loss_ratio: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    params: dict = field(default_factory=dict)


def run_backtest_from_scores(
    daily_df: pd.DataFrame,
    score_df: pd.DataFrame,
    top_n: int = 5,
    start_date: str = "20180101",
) -> BacktestResult:
    """从预计算得分矩阵执行回测。

    Args:
        daily_df: 行业日线数据 (trade_date, ts_code, close)
        score_df: 预计算的得分矩阵 (index=trade_date, columns=ts_code, values=score)
        top_n: 每日选取行业数
        start_date: 回测起始日期

    Returns:
        BacktestResult
    """
    # 构建收益率矩阵
    pivoted = daily_df.pivot_table(
        index="trade_date", columns="ts_code", values="close", aggfunc="last"
    )
    pivoted = pivoted.sort_index()
    pivoted = pivoted.ffill()
    returns = pivoted.pct_change()

    # 对齐得分和收益率
    common_dates = sorted(set(score_df.index) & set(returns.index))
    if not common_dates:
        raise ValueError("得分矩阵和收益率矩阵无共同日期")

    score_df = score_df.loc[common_dates]
    returns = returns.loc[common_dates]

    # 只取回测窗口
    mask = [d >= start_date for d in common_dates]
    bt_dates = [d for d, m in zip(common_dates, mask) if m]
    bt_start_idx = mask.index(True) if True in mask else 0

    if bt_start_idx >= len(common_dates) - 1:
        raise ValueError("回测窗口无数据")

    nav = 1.0
    bench_nav = 1.0
    nav_list = [1.0]
    bench_list = [1.0]
    daily_rets = []
    bench_rets = []

    for i in range(bt_start_idx, len(common_dates)):
        date = common_dates[i]
        scores = score_df.loc[date].dropna()

        if len(scores) == 0:
            nav_list.append(nav)
            bench_list.append(bench_nav)
            daily_rets.append(0.0)
            bench_rets.append(0.0)
            continue

        # Top-N 等权
        top_codes = scores.nlargest(min(top_n, len(scores))).index.tolist()
        w = 1.0 / len(top_codes)

        # 当日收益
        day_ret = 0.0
        valid = 0
        for code in top_codes:
            if code in returns.columns:
                r = returns.loc[date, code]
                if not np.isnan(r):
                    day_ret += w * r
                    valid += 1

        if valid == 0:
            day_ret = 0.0
        elif valid < len(top_codes):
            day_ret = day_ret * (len(top_codes) / valid)

        # 基准（等权）
        bench_ret = returns.loc[date].mean()

        nav *= (1 + day_ret)
        bench_nav *= (1 + bench_ret)

        nav_list.append(nav)
        bench_list.append(bench_nav)
        daily_rets.append(day_ret)
        bench_rets.append(bench_ret)

    result = BacktestResult(
        nav=pd.Series(nav_list, index=pd.to_datetime(common_dates[bt_start_idx - 1:])),
        benchmark_nav=pd.Series(bench_list, index=pd.to_datetime(common_dates[bt_start_idx - 1:])),
        daily_returns=pd.Series(daily_rets, index=pd.to_datetime(bt_dates)),
        benchmark_returns=pd.Series(bench_rets, index=pd.to_datetime(bt_dates)),
    )

    _compute_metrics(result)
    return result


def compute_rolling_persistence_scores(
    daily_df: pd.DataFrame,
    window: int = 20,
    weights: dict | None = None,
    min_days: int = 60,
) -> pd.DataFrame:
    """预计算每日每行业的持续性得分（滚动窗口）。

    对每个日期，用前 window 日的价格数据计算每个行业的持续性得分。

    Args:
        daily_df: 日线数据
        window: 动量回看窗口
        weights: 权重字典（简化为只用动量分）
        min_days: 最少需要的历史天数

    Returns:
        DataFrame (index=trade_date, columns=ts_code, values=persistence_score)
    """
    if weights is None:
        weights = {"momentum": 0.35, "return_slope": 0.25, "turnover": 0.20, "relative": 0.20}

    daily_df = daily_df.sort_values(["ts_code", "trade_date"]).copy()

    # 用简化版快速计算：20日动量 + 5日动量
    # 这是原始持续性评分的最主要成分
    codes = sorted(daily_df["ts_code"].unique())
    dates = sorted(daily_df["trade_date"].unique())

    # 预计算每个 code 的 close price series
    price_dict = {}
    for c in codes:
        grp = daily_df[daily_df["ts_code"] == c].set_index("trade_date")["close"]
        price_dict[c] = grp

    # 对每个日期计算得分
    scores_dict = {}
    total = len(dates[min_days:])

    for idx, date in enumerate(dates[min_days:], 1):
        date_scores = {}
        for code in codes:
            prices = price_dict.get(code, pd.Series())
            if len(prices) == 0:
                continue

            # 取截止当前日期的价格（严格历史：不含当日）
            hist = prices[prices.index < date]
            if len(hist) < window + 1:
                continue

            prices_hist = hist.iloc[-(window + 1):]

            # 20日动量
            mom20 = (prices_hist.iloc[-1] / prices_hist.iloc[0] - 1) * 100 if prices_hist.iloc[0] > 0 else 0

            # 5日动量
            mom5 = 0.0
            if len(prices_hist) >= 6:
                mom5 = (prices_hist.iloc[-1] / prices_hist.iloc[-6] - 1) * 100

            # 简化持续性分 = 动量分 + 趋势确认
            ma20 = prices_hist.mean()
            trend = (prices_hist.iloc[-1] / ma20 - 1) * 100 if ma20 > 0 else 0

            # Normalize to 0-10
            score = 5.0 + (mom20 * 0.4 + mom5 * 0.3 + trend * 0.3) / 3
            score = max(0, min(10, score))
            date_scores[code] = score

        if date_scores:
            scores_dict[date] = date_scores

        if idx % 500 == 0:
            logger.info("  得分计算进度: %d/%d", idx, total)

    score_df = pd.DataFrame(scores_dict).T
    score_df.index.name = "trade_date"
    return score_df


def _compute_metrics(result: BacktestResult) -> None:
    """计算绩效指标。"""
    rets = result.daily_returns.dropna()
    if len(rets) == 0:
        return

    result.total_return = float(result.nav.iloc[-1] - 1)
    result.benchmark_return = float(result.benchmark_nav.iloc[-1] - 1)
    result.excess_return = result.total_return - result.benchmark_return

    n_years = len(rets) / 252
    if n_years > 0:
        result.annual_return = float((1 + result.total_return) ** (1 / n_years) - 1)
    else:
        result.annual_return = 0.0

    result.annual_volatility = float(rets.std() * np.sqrt(252))

    rf_daily = 0.02 / 252
    excess = rets - rf_daily
    if excess.std() > 0:
        result.sharpe_ratio = float(excess.mean() / excess.std() * np.sqrt(252))

    peak = result.nav.expanding().max()
    drawdown = (result.nav - peak) / peak
    result.max_drawdown = float(drawdown.min())

    dd_start = None
    max_dd_days = 0
    for i, dd in enumerate(drawdown):
        if dd < 0 and dd_start is None:
            dd_start = i
        elif dd >= 0 and dd_start is not None:
            max_dd_days = max(max_dd_days, i - dd_start)
            dd_start = None
    if dd_start is not None:
        max_dd_days = max(max_dd_days, len(drawdown) - dd_start)
    result.max_drawdown_days = max_dd_days

    if abs(result.max_drawdown) > 0.001:
        result.calmar_ratio = result.annual_return / abs(result.max_drawdown)

    result.win_rate_daily = float((rets > 0).mean())
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    result.win_rate_monthly = float((monthly > 0).mean())

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    if len(losses) > 0 and losses.mean() != 0:
        result.profit_loss_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) > 0 else 0.0
