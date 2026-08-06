"""
个股周涨幅预测与回测验证模块
================================
核心逻辑：
  1. 向量化预计算所有因子（一次扫描，全量计算）
  2. 基于多因子评分对全市场个股排序
  3. 滚动回测验证预测准确性

评分因子（选股逻辑 = 找"一周内可能大涨"的股票）：
  - momentum_5d:   近 5 日涨幅 —— 短期加速信号
  - momentum_20d:  近 20 日涨幅 —— 中期趋势
  - volatility_20d: 近 20 日波动率 —— 波幅越大约可能大涨
  - volume_ratio:  近 5 日均量 / 近 20 日均量 —— 放量信号
  - ma20_deviation: 价格偏离 MA20 程度 —— 突破确认
  - rel_strength:  相对全市场平均的 20 日动量 —— 选强弃弱

用法（独立运行）:
  python3 analysis/backtest_predictor.py              # 完整回测（120 天）
  python3 analysis/backtest_predictor.py --quick       # 快速回测（60 天）
  python3 analysis/backtest_predictor.py --predict latest  # 预测今天
"""

import argparse
import logging
import os
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DB_PATH,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_PCT_CHG,
    COL_VOL,
    COL_OPEN,
    COL_AMOUNT,
    COL_HIGH,
    COL_LOW,
    MOMENTUM_LOOKBACK,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)

# ============================================================
# 可调参数
# ============================================================
PREDICT_HORIZON = 5          # 预测未来 N 个交易日
TOP_N_DEFAULT = 10           # 每次选入 TOP_N 只股票
MIN_HISTORY = 30             # 个股最少历史天数
MIN_PRICE = 3.0              # 最低股价（过滤低价股/ST）

# 回测默认时间范围（按数据库实际数据自动调整）
BACKTEST_DAYS_DEFAULT = 120   # 默认回看天数
BACKTEST_DAYS_QUICK = 60      # 快速模式

# 评分权重
WEIGHTS = {
    "momentum_5d": 0.25,
    "momentum_20d": 0.20,
    "volatility_20d": 0.15,
    "volume_ratio": 0.15,
    "ma20_deviation": 0.10,
    "rel_strength": 0.15,
}


# ============================================================
# 数据加载
# ============================================================

def load_stock_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """从数据库加载个股日线数据。

    返回按 ts_code + trade_date 排序的完整 DataFrame。
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT trade_date, ts_code, open, high, low, close, "
        "pct_chg, vol, amount FROM stock_daily ORDER BY trade_date",
        conn,
    )
    conn.close()
    df[COL_TRADE_DATE] = df[COL_TRADE_DATE].astype(str)
    return df


def get_trading_dates(df: pd.DataFrame) -> list:
    """获取所有交易日（升序）。"""
    return sorted(df[COL_TRADE_DATE].unique())


# ============================================================
# 向量化因子预计算
# ============================================================

def precompute_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """向量化预计算所有股票的所有因子和未来收益。

    用 pandas groupby + rolling / shift 替代逐只循环，
    全量计算复杂度 = O(总行数 * 因子数)，与股票数无关。

    Returns:
        原 DataFrame + 新增因子列 + forward_return 列
    """
    df = df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    close_col = COL_CLOSE
    vol_col = COL_VOL

    # ---- 动量因子 ----
    for period, label in [(5, "momentum_5d"), (10, "momentum_10d"), (20, "momentum_20d")]:
        df[label] = df.groupby(COL_TS_CODE)[close_col].transform(
            lambda s: (s / s.shift(period) - 1) * 100
        )

    # ---- 波动率（年化） ----
    def _vol(s, w):
        ret = s.pct_change()
        return ret.rolling(w, min_periods=5).std() * np.sqrt(252) * 100

    df["volatility_5d"] = df.groupby(COL_TS_CODE)[close_col].transform(
        lambda s: _vol(s, 5)
    )
    df["volatility_20d"] = df.groupby(COL_TS_CODE)[close_col].transform(
        lambda s: _vol(s, 20)
    )

    # ---- 量比 ----
    df["vol_ma5"] = df.groupby(COL_TS_CODE)[vol_col].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df["vol_ma20"] = df.groupby(COL_TS_CODE)[vol_col].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df["volume_ratio"] = df["vol_ma5"] / df["vol_ma20"].replace(0, np.nan)

    # ---- MA20 偏离度 ----
    df["ma20"] = df.groupby(COL_TS_CODE)[close_col].transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )
    df["ma20_deviation"] = (df[close_col] / df["ma20"] - 1) * 100

    # ---- ATR 比率 ----
    prev_close = df.groupby(COL_TS_CODE)[close_col].shift(1)
    tr = pd.concat([
        df[COL_HIGH] - df[COL_LOW],
        (df[COL_HIGH] - prev_close).abs(),
        (df[COL_LOW] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["_tr_"] = tr
    df["atr"] = df.groupby(COL_TS_CODE)["_tr_"].transform(
        lambda s: s.rolling(14, min_periods=7).mean()
    )
    df["atr_ratio"] = (df["atr"] / df[close_col].replace(0, np.nan)) * 100
    df = df.drop(columns=["_tr_", "vol_ma5", "vol_ma20", "ma20"], errors="ignore")

    # ---- 未来 5 日收益（回测标签） ----
    df["forward_return"] = df.groupby(COL_TS_CODE)[close_col].transform(
        lambda s: (s.shift(-PREDICT_HORIZON) / s - 1) * 100
    )

    return df


# ============================================================
# 评分与排名
# ============================================================

def _score_day(df_day: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """对单日数据评分排序，返回 TOP-N。

    Args:
        df_day: 某一天的因子 DataFrame
        top_n: 选取数量

    Returns:
        DataFrame with columns: ts_code, score, forward_return, 因子列
    """
    if df_day.empty:
        return pd.DataFrame()

    # 过滤
    mask = (
        (df_day[COL_CLOSE] >= MIN_PRICE)
        & (df_day[COL_VOL] > 0)
        & (df_day["momentum_5d"].notna())
    )
    df_valid = df_day[mask].copy()
    if df_valid.empty:
        return pd.DataFrame()

    # 相对强度
    market_avg_20d = df_valid["momentum_20d"].mean()
    df_valid["rel_strength"] = df_valid["momentum_20d"] - market_avg_20d

    # 综合评分
    w = WEIGHTS
    df_valid["score"] = 0.0
    for key, weight in w.items():
        vals = df_valid[key]
        df_valid.loc[vals.notna(), "score"] += vals * weight

    df_valid = df_valid.dropna(subset=["score"])

    if df_valid.empty:
        return pd.DataFrame()

    return df_valid.nlargest(top_n, "score")


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(
    db_path: str = DB_PATH,
    lookback_days: int = BACKTEST_DAYS_DEFAULT,
    top_n: int = TOP_N_DEFAULT,
    verbose: bool = True,
) -> dict:
    """主回测流程（向量化版本）。

    流程：
      1. 加载全量数据
      2. 一次性预计算所有股票的因子 + 未来收益
      3. 逐日评分（DataFrame 过滤，无逐股循环）
      4. 统计汇总

    Args:
        db_path: 数据库路径
        lookback_days: 回测涵盖的天数（从最新交易日往前数）
        top_n: 每次选入 TOP N 只股票
        verbose: 是否打印进度

    Returns:
        dict 回测报告
    """
    t0 = time.time()

    # ---- 加载 + 预计算 ----
    df = load_stock_data(db_path)
    logger.info("加载 %d 只股票, %d 个交易日",
                df[COL_TS_CODE].nunique(), df[COL_TRADE_DATE].nunique())
    logger.info("预计算因子...")
    df = precompute_features_vectorized(df)
    all_dates = get_trading_dates(df)
    logger.info("预计算完成，开始回测...")

    if len(all_dates) < lookback_days + MIN_HISTORY:
        lookback_days = len(all_dates) - MIN_HISTORY - 1

    # 确定回测日期范围
    test_end_idx = len(all_dates) - PREDICT_HORIZON
    test_start_idx = max(MIN_HISTORY, test_end_idx - lookback_days)
    test_dates = all_dates[test_start_idx:test_end_idx]
    logger.info("回测区间: %s -> %s (%d 天)",
                test_dates[0], test_dates[-1], len(test_dates))

    # ---- 逐日评分 ----
    all_predictions = []

    for i, date in enumerate(test_dates):
        df_today = df[df[COL_TRADE_DATE] == date]
        if df_today.empty:
            continue

        candidates = _score_day(df_today, top_n)
        if candidates.empty:
            continue

        for rank, (_, row) in enumerate(candidates.iterrows(), 1):
            fwd_ret = row.get("forward_return")
            if pd.isna(fwd_ret):
                continue

            all_predictions.append({
                "date": date,
                "ts_code": row[COL_TS_CODE],
                "rank": rank,
                "score": row["score"],
                "forward_return": round(float(fwd_ret), 2),
                "momentum_5d": row.get("momentum_5d"),
                "momentum_20d": row.get("momentum_20d"),
                "volatility_20d": row.get("volatility_20d"),
                "volume_ratio": row.get("volume_ratio"),
                "ma20_deviation": row.get("ma20_deviation"),
                "atr_ratio": row.get("atr_ratio"),
            })

        if verbose and (i + 1) % 20 == 0:
            logger.info("回测进度: %d / %d", i + 1, len(test_dates))

    elapsed = time.time() - t0
    logger.info("回测完成: %d 次预测, 耗时 %.1f 秒",
                len(all_predictions), elapsed)

    # ---- 统计汇总 ----
    if not all_predictions:
        return {"status": "no_data", "error": "回测无有效预测"}

    pred_df = pd.DataFrame(all_predictions)
    fwd_arr = pred_df["forward_return"].values

    # 各阈值胜率
    thresholds = [0, 5, 10, 15, 20, 25, 30]
    win_rates = {}
    for t in thresholds:
        win_rates[f">{t}%"] = {
            "count": int(np.sum(fwd_arr > t)),
            "rate": round(float(np.mean(fwd_arr > t) * 100), 2),
        }

    # 分布百分位
    percentiles_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    # 因子效果分析
    factor_analysis = {}
    for factor in ["momentum_5d", "momentum_20d", "volatility_20d",
                   "volume_ratio", "atr_ratio"]:
        vals = pred_df[factor].dropna()
        if len(vals) < 50:
            continue
        try:
            pred_df["_bin"] = pd.qcut(vals, 4, labels=["Q1", "Q2", "Q3", "Q4"],
                                       duplicates="drop")
            fa = {}
            for name, grp in pred_df.groupby("_bin", observed=False):
                fa[str(name)] = {
                    "count": len(grp),
                    "avg_return": round(float(grp["forward_return"].mean()), 2),
                    "win_rate_gt0": round(float((grp["forward_return"] > 0).mean() * 100), 2),
                    "win_rate_gt10": round(float((grp["forward_return"] > 10).mean() * 100), 2),
                }
            factor_analysis[factor] = fa
        except Exception:
            continue
    pred_df.drop(columns=["_bin"], errors="ignore", inplace=True)

    report = {
        "status": "success",
        "metadata": {
            "backtest_period": f"{test_dates[0]} -> {test_dates[-1]}",
            "trading_days": len(test_dates),
            "total_predictions": len(all_predictions),
            "stocks_per_day": top_n,
            "predict_horizon": f"{PREDICT_HORIZON} 交易日",
            "computation_time_sec": round(elapsed, 2),
        },
        "summary": {
            "avg_return": round(float(np.mean(fwd_arr)), 2),
            "median_return": round(float(np.median(fwd_arr)), 2),
            "max_return": round(float(np.max(fwd_arr)), 2),
            "min_return": round(float(np.min(fwd_arr)), 2),
            "std_return": round(float(np.std(fwd_arr)), 2),
            "positive_rate": round(float(np.mean(fwd_arr > 0) * 100), 2),
            "avg_top5_return": round(float(
                pred_df[pred_df["rank"] <= 5]["forward_return"].mean()
            ), 2),
            "avg_top3_return": round(float(
                pred_df[pred_df["rank"] <= 3]["forward_return"].mean()
            ), 2),
        },
        "win_rates": win_rates,
        "percentiles": {str(p): round(float(np.percentile(fwd_arr, p)), 2)
                        for p in percentiles_list},
        "best_predictions": pred_df.nlargest(10, "forward_return")[
            ["date", "ts_code", "forward_return", "score", "momentum_5d"]
        ].to_dict("records") if len(pred_df) > 0 else [],
        "worst_predictions": pred_df.nsmallest(10, "forward_return")[
            ["date", "ts_code", "forward_return", "score", "momentum_5d"]
        ].to_dict("records") if len(pred_df) > 0 else [],
        "factor_analysis": factor_analysis,
    }

    report["_raw_predictions"] = pred_df

    # ---- 止损分析 ----
    report["stop_loss_analysis"] = None
    if not pred_df.empty and df is not None and COL_CLOSE in df.columns:
        try:
            report["stop_loss_analysis"] = simulate_stop_losses(pred_df, df)
            logger.info("止损分析完成")
        except Exception as ex:
            logger.warning("止损分析失败: %s", ex)

    # 缓存回测摘要供报告使用（不含原始数据以节省内存）
    _cache_backtest_summary(report)

    return report


# ============================================================
# 回测摘要缓存（供交易参考报告调用，避免重复计算）
# ============================================================
_BACKTEST_SUMMARY_CACHE = None


def _cache_backtest_summary(report: dict):
    """从完整回测报告中提取摘要并缓存。"""
    global _BACKTEST_SUMMARY_CACHE
    s = report.get("summary", {})
    wr = report.get("win_rates", {})
    _BACKTEST_SUMMARY_CACHE = {
        "positive_rate": s.get("positive_rate"),
        "avg_return": s.get("avg_return"),
        "avg_top3_return": s.get("avg_top3_return"),
        "avg_top5_return": s.get("avg_top5_return"),
        "win_rate_gt5": wr.get(">5%", {}).get("rate") if wr else None,
        "win_rate_gt10": wr.get(">10%", {}).get("rate") if wr else None,
        "total_predictions": report.get("metadata", {}).get("total_predictions"),
    }


def get_backtest_summary() -> dict:
    """返回最后一次回测的摘要统计，供报告使用。"""
    return _BACKTEST_SUMMARY_CACHE


# ============================================================
# 对外预测接口
# ============================================================

def predict_top_stocks(
    db_path: str = DB_PATH,
    date: str = "latest",
    top_n: int = TOP_N_DEFAULT,
) -> list:
    """对外预测接口：给定日期，返回最可能大涨的 TOP-N 股票。

    Args:
        db_path: 数据库路径
        date: 基准日期 YYYYMMDD，或 "latest"
        top_n: 返回数量

    Returns:
        [{"ts_code": "...", "score": 85.5, "features": {...}}, ...]
    """
    df = load_stock_data(db_path)
    all_dates = get_trading_dates(df)

    if date == "latest":
        date = all_dates[-1]

    df = precompute_features_vectorized(df)
    df_today = df[df[COL_TRADE_DATE] == date]

    if df_today.empty:
        logger.warning("日期 %s 无数据", date)
        return []

    candidates = _score_day(df_today, top_n)
    if candidates.empty:
        return []

    feature_keys = ["momentum_5d", "momentum_20d", "volatility_20d",
                    "volume_ratio", "ma20_deviation", "atr_ratio",
                    "rel_strength"]
    results = []
    for _, row in candidates.iterrows():
        feats = {k: round(float(row[k]), 2) if pd.notna(row.get(k)) else None
                 for k in feature_keys}
        results.append({
            "ts_code": row[COL_TS_CODE],
            "score": round(float(row["score"]), 1),
            "features": feats,
        })

    return results


# ============================================================
# 止损策略模拟
# ============================================================

def simulate_stop_losses(
    predictions_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    stop_levels: list = None,
) -> dict:
    """对回测的每笔交易模拟止损策略。

    每笔交易：在日期 date 买入 ts_code，持有 PREDICT_HORIZON 天。
    止损逻辑：每日检查收盘价，若跌破 entry_price * (1 - SL%) 则当日离场。

    Args:
        predictions_df: 回测预测记录（含 date, ts_code, forward_return）
        stock_df: 全量日线数据（需要 close, trade_date, ts_code）
        stop_levels: 止损线列表（%），默认 [3, 5, 7, 10]

    Returns:
        dict: {stop_level: {avg_return, win_rate, max_loss, hit_rate, ...}}
    """
    if stop_levels is None:
        stop_levels = [3, 5, 7, 10]

    horizon = PREDICT_HORIZON
    results = {"baseline": {}, "fixed_stops": {}, "atr_stops": {}}

    # ---- 基准（不止损） ----
    fwd = predictions_df["forward_return"].values
    results["baseline"] = {
        "avg_return": round(float(np.mean(fwd)), 2),
        "median_return": round(float(np.median(fwd)), 2),
        "win_rate": round(float(np.mean(fwd > 0) * 100), 2),
        "win_rate_gt10": round(float(np.mean(fwd > 10) * 100), 2),
        "win_rate_gt20": round(float(np.mean(fwd > 20) * 100), 2),
        "max_loss": round(float(np.min(fwd)), 2),
        "std_return": round(float(np.std(fwd)), 2),
    }

    # ---- 按股票分组加速查找（价格 + ATR） ----
    has_atr = "atr_ratio" in stock_df.columns
    stock_data = {}
    for ts_code, grp in stock_df.groupby(COL_TS_CODE):
        grp = grp.sort_values(COL_TRADE_DATE)
        d = {"dates": grp[COL_TRADE_DATE].tolist(), "prices": grp[COL_CLOSE].tolist()}
        if has_atr:
            d["atr"] = grp["atr_ratio"].tolist()
        stock_data[ts_code] = d

    # 对每个固定%止损水平模拟
    for sl in stop_levels:
        sl_returns = []
        sl_hold_days = []
        sl_triggered = 0

        for _, pred in predictions_df.iterrows():
            ts_code = pred["ts_code"]
            entry_date = pred["date"]

            sd = stock_data.get(ts_code)
            if sd is None:
                continue
            dates_list, prices_list = sd["dates"], sd["prices"]
            try:
                entry_idx = dates_list.index(entry_date)
            except ValueError:
                continue

            if entry_idx + horizon >= len(prices_list):
                continue

            entry_price = prices_list[entry_idx]
            if entry_price <= 0:
                continue

            stop_price = entry_price * (1 - sl / 100.0)
            exited_early = False

            for day in range(1, horizon + 1):
                day_close = prices_list[entry_idx + day]
                if day_close < stop_price:
                    # 止损触发
                    trade_return = (day_close / entry_price - 1) * 100
                    sl_returns.append(trade_return)
                    sl_hold_days.append(day)
                    sl_triggered += 1
                    exited_early = True
                    break

            if not exited_early:
                exit_price = prices_list[entry_idx + horizon]
                trade_return = (exit_price / entry_price - 1) * 100
                sl_returns.append(trade_return)
                sl_hold_days.append(horizon)

        if not sl_returns:
            continue

        arr = np.array(sl_returns)
        results["fixed_stops"][f"{sl}%"] = {
            "avg_return": round(float(np.mean(arr)), 2),
            "median_return": round(float(np.median(arr)), 2),
            "win_rate": round(float(np.mean(arr > 0) * 100), 2),
            "win_rate_gt10": round(float(np.mean(arr > 10) * 100), 2),
            "win_rate_gt20": round(float(np.mean(arr > 20) * 100), 2),
            "max_loss": round(float(np.min(arr)), 2),
            "std_return": round(float(np.std(arr)), 2),
            "avg_hold_days": round(float(np.mean(sl_hold_days)), 1),
            "trigger_rate": round(sl_triggered / len(sl_returns) * 100, 1),
        }

    # ---- ATR 动态止损 ----
    if has_atr:
        atr_multipliers = [1.5, 2, 3]
        for mult in atr_multipliers:
            sl_returns = []
            sl_hold_days = []
            sl_triggered = 0

            for _, pred in predictions_df.iterrows():
                ts_code = pred["ts_code"]
                entry_date = pred["date"]

                sd = stock_data.get(ts_code)
                if sd is None or "atr" not in sd:
                    continue

                try:
                    entry_idx = sd["dates"].index(entry_date)
                except ValueError:
                    continue

                if entry_idx + horizon >= len(sd["prices"]):
                    continue

                entry_price = sd["prices"][entry_idx]
                if entry_price <= 0:
                    continue

                atr_val = sd["atr"][entry_idx]
                if atr_val is None or np.isnan(atr_val):
                    exit_price = sd["prices"][entry_idx + horizon]
                    if exit_price > 0:
                        ret = (exit_price / entry_price - 1) * 100
                        sl_returns.append(ret)
                        sl_hold_days.append(horizon)
                    continue

                stop_price = entry_price * (1 - mult * atr_val / 100.0)
                exited = False

                for day in range(1, horizon + 1):
                    day_close = sd["prices"][entry_idx + day]
                    if day_close < stop_price:
                        ret = (day_close / entry_price - 1) * 100
                        sl_returns.append(ret)
                        sl_hold_days.append(day)
                        sl_triggered += 1
                        exited = True
                        break

                if not exited:
                    exit_price = sd["prices"][entry_idx + horizon]
                    ret = (exit_price / entry_price - 1) * 100
                    sl_returns.append(ret)
                    sl_hold_days.append(horizon)

            if sl_returns:
                arr = np.array(sl_returns)
                results["atr_stops"][f"ATR {mult}x"] = {
                    "avg_return": round(float(np.mean(arr)), 2),
                    "median_return": round(float(np.median(arr)), 2),
                    "win_rate": round(float(np.mean(arr > 0) * 100), 2),
                    "win_rate_gt10": round(float(np.mean(arr > 10) * 100), 2),
                    "win_rate_gt20": round(float(np.mean(arr > 20) * 100), 2),
                    "max_loss": round(float(np.min(arr)), 2),
                    "std_return": round(float(np.std(arr)), 2),
                    "avg_hold_days": round(float(np.mean(sl_hold_days)), 1),
                    "trigger_rate": round(sl_triggered / len(sl_returns) * 100, 1),
                }

    return results


# ============================================================
# 结果展示
# ============================================================

def print_backtest_report(report: dict) -> None:
    """格式化打印回测报告到控制台。"""
    if report.get("status") != "success":
        print(f"\n❌ 回测失败: {report.get('error', '未知错误')}")
        return

    meta = report["metadata"]
    summ = report["summary"]
    wrs = report["win_rates"]
    pcts = report["percentiles"]

    print("\n" + "=" * 65)
    print("  个股周涨幅预测 . 回测报告")
    print("=" * 65)
    print(f"  回测区间:    {meta['backtest_period']}")
    print(f"  交易日数:    {meta['trading_days']} 天")
    print(f"  总预测次数:  {meta['total_predictions']} 次 (每日 TOP-{meta['stocks_per_day']})")
    print(f"  预测窗口:    未来 {meta['predict_horizon']}")
    print(f"  计算耗时:    {meta['computation_time_sec']} 秒")
    print("-" * 65)
    print(f"  \U0001f4ca 收益统计")
    print(f"     平均收益:  {summ['avg_return']:+.2f}%")
    print(f"     中位收益:  {summ['median_return']:+.2f}%")
    print(f"     最大收益:  {summ['max_return']:+.2f}%")
    print(f"     最小收益:  {summ['min_return']:+.2f}%")
    print(f"     标准差:    {summ['std_return']:.2f}%")
    print(f"     胜率 (>0%): {summ['positive_rate']:.1f}%")
    print(f"     TOP-3 平均: {summ['avg_top3_return']:+.2f}%")
    print(f"     TOP-5 平均: {summ['avg_top5_return']:+.2f}%")
    print("-" * 65)
    print(f"  \U0001f3af 各阈值胜率")
    for threshold, data in wrs.items():
        bar = "█" * int(data["rate"] / 5)
        print(f"     {threshold:>5}: {data['rate']:6.2f}% ({data['count']}次) {bar}")
    print("-" * 65)
    print(f"  分布百分位")
    for label in ["1", "5", "10", "25", "50", "75", "90", "95", "99"]:
        val = pcts.get(label, 0)
        print(f"     P{label:>2}: {val:+.2f}%")
    print("=" * 65)

    print("\n  \U0001f4c8 最佳预测 TOP-5:")
    for i, pred in enumerate(report.get("best_predictions", [])[:5], 1):
        print(f"    {i}. {pred['ts_code']} @ {pred['date']}"
              f" -> +{pred['forward_return']:.1f}%"
              f" (评分 {pred.get('score', '?'):.1f},"
              f" 5日动 {pred.get('momentum_5d', '?'):.1f})")

    print("\n  \U0001f4c9 最差预测 TOP-5:")
    for i, pred in enumerate(report.get("worst_predictions", [])[:5], 1):
        print(f"    {i}. {pred['ts_code']} @ {pred['date']}"
              f" -> {pred['forward_return']:.1f}%"
              f" (评分 {pred.get('score', '?'):.1f},"
              f" 5日动 {pred.get('momentum_5d', '?'):.1f})")

    fa = report.get("factor_analysis", {})
    if fa:
        print("\n  \U0001f52c 因子效果分析 (各因子分箱 -> 平均收益)")
        for factor, bins in fa.items():
            print(f"    {factor}:")
            for bin_name, stats in bins.items():
                print(f"      {bin_name}: 平均 {stats['avg_return']:+.1f}%, "
                      f"胜率>0 {stats['win_rate_gt0']:.0f}%, "
                      f"胜率>10% {stats['win_rate_gt10']:.0f}%"
                      f" ({stats['count']}次)")

    sla = report.get("stop_loss_analysis")
    if sla and (sla.get("fixed_stops") or sla.get("atr_stops")):
        print(f"\n  \U0001f6d1 止损策略对比")
        baseline = sla["baseline"]
        print(f"      {'='*80}")
        print(f"      {'策略':>10} | {'平均收益':>8} | {'胜率>0':>8} | {'胜率>20%':>9} | {'最大亏损':>8} | {'标准差':>7} | {'持仓天数':>8} | {'触发率':>7}")
        print(f"      {'-'*80}")
        print(f"      {'不止损':>10} | {baseline['avg_return']:>+7.2f}% | {baseline['win_rate']:>7.1f}% | {baseline['win_rate_gt20']:>8.1f}% | {baseline['max_loss']:>+7.2f}% | {baseline['std_return']:>6.2f} | {'5.0天':>8} | {'0.0%':>7}")

        for sl_name, stats in sla["fixed_stops"].items():
            print(f"      {sl_name:>10} | {stats['avg_return']:>+7.2f}% | {stats['win_rate']:>7.1f}% | {stats['win_rate_gt20']:>8.1f}% | {stats['max_loss']:>+7.2f}% | {stats['std_return']:>6.2f} | {stats['avg_hold_days']:>5.1f}天 | {stats['trigger_rate']:>6.1f}%")

        atr_stops = sla.get("atr_stops", {})
        if atr_stops:
            print(f"      {'-'*80}")
            for sl_name, stats in atr_stops.items():
                print(f"      {sl_name:>10} | {stats['avg_return']:>+7.2f}% | {stats['win_rate']:>7.1f}% | {stats['win_rate_gt20']:>8.1f}% | {stats['max_loss']:>+7.2f}% | {stats['std_return']:>6.2f} | {stats['avg_hold_days']:>5.1f}天 | {stats['trigger_rate']:>6.1f}%")
    print()


# ============================================================
# 独立运行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="个股一周涨幅预测回测")
    parser.add_argument("--quick", action="store_true", help="快速模式 (60 天)")
    parser.add_argument("--days", type=int, default=None, help="回看天数")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT, help="每日选股数")
    parser.add_argument("--predict", type=str, default=None,
                        help="预测模式: 指定日期 YYYYMMDD, 或 latest")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.predict:
        results = predict_top_stocks(date=args.predict, top_n=args.top)
        dt = "最新交易日" if args.predict == "latest" else args.predict
        print(f"\n\U0001f4cb 预测 TOP-{len(results)} (基准日: {dt})")
        print("=" * 60)
        for i, r in enumerate(results, 1):
            feats = r["features"]
            print(f"  {i:2d}. {r['ts_code']}"
                  f"  评分 {r['score']:>5.1f}"
                  f" | 5日动 {feats.get('momentum_5d', 'N/A')}"
                  f" | 20日动 {feats.get('momentum_20d', 'N/A')}"
                  f" | 波幅 {feats.get('atr_ratio', 'N/A')}%"
                  f" | 量比 {feats.get('volume_ratio', 'N/A')}")
        return

    lookback = args.days or (BACKTEST_DAYS_QUICK if args.quick else BACKTEST_DAYS_DEFAULT)
    report = run_backtest(lookback_days=lookback, top_n=args.top)
    print_backtest_report(report)

    raw = report.get("_raw_predictions")
    if raw is not None and not raw.empty:
        csv_path = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        raw.to_csv(csv_path, index=False)
        print(f"  详细数据已保存: {csv_path}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════
# ML 模型回测 — 使用真实 LightGBM 模型（V5, 306特征）
# ═══════════════════════════════════════════════════════════════

def run_ml_backtest(
    feature_df: pd.DataFrame,
    stock_daily_df: pd.DataFrame,
    models: dict,
    top_n: int = 5,
    backtest_days: int = 175,
    forward_horizon: int = 20,
    stop_loss: float = 0.05,
    max_hold: int = 20,
) -> dict:
    """用真实 ML 模型对历史数据进行回测。

    策略: 每日 ML Top-N 买入 → 5% 硬止损 / 最长持仓 max_hold 天 / 到期卖出
    Args:
        feature_df: 特征矩阵（含 ts_code, trade_date, close, amount, 特征列）
        stock_daily_df: 个股日线（用于价格跟踪）
        models: {"micro": LGBM, "mid": LGBM} 或 "micro"/"mid" 模型 dict
        top_n: 每日选股数
        backtest_days: 回测天数（从最新往前数）
        forward_horizon: 前向持有期（与训练一致，默认20天）
        stop_loss: 硬止损阈值（默认5%）
        max_hold: 最长持仓天数

    Returns:
        dict 回测摘要（total_return, sharpe, max_drawdown, win_rate, trades...）
    """
    from analysis.ml_model import ALL_FEATURES, load_model, _compute_amount_thresholds, assign_group

    df = feature_df.copy()
    sd = stock_daily_df.copy()
    sd[COL_TRADE_DATE] = pd.to_datetime(sd[COL_TRADE_DATE])

    # 三分组阈值
    thresholds = _compute_amount_thresholds(sd)

    # 所有交易日
    dates = sorted(df[COL_TRADE_DATE].unique())
    if len(dates) < backtest_days + 2:
        backtest_days = len(dates) - 2
    test_end = len(dates) - forward_horizon  # 需要未来收益
    test_start = max(0, test_end - backtest_days)
    test_dates = dates[test_start:test_end]

    # 预计算分组
    amt_map = sd.groupby(COL_TS_CODE)[COL_AMOUNT].mean().to_dict()

    def _grp(code):
        return assign_group(amt_map.get(code, 0), thresholds)

    # 可用特征列
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    logger.info("ML 回测: %d 特征, %d 天 (%s ~ %s)",
                len(feat_cols), len(test_dates), test_dates[0], test_dates[-1])

    # 逐日评分
    daily_scores = []  # (date, ts_code, score)
    for date in test_dates:
        day = df[df[COL_TRADE_DATE] == date]
        if day.empty:
            continue
        for grp_name in ["micro", "mid"]:
            model = models.get(grp_name) if models else load_model(grp_name)
            if model is None:
                continue
            grp_df = day[day[COL_TS_CODE].map(_grp) == grp_name]
            if grp_df.empty:
                continue
            X = grp_df[feat_cols].fillna(0).values
            scores = model.predict(X)
            for code, sc in zip(grp_df[COL_TS_CODE].values, scores):
                daily_scores.append((date, code, float(sc)))

    score_df = pd.DataFrame(daily_scores, columns=["date", COL_TS_CODE, "score"])

    # 价格表: (ts_code, date) → close（日期统一为 YYYYMMDD 字符串，与特征矩阵一致）
    price_pivot = sd.pivot_table(index=COL_TRADE_DATE, columns=COL_TS_CODE, values=COL_CLOSE)
    price_pivot.index = [str(pd.Timestamp(d).strftime("%Y%m%d")) for d in price_pivot.index]
    date_positions = {str(d): i for i, d in enumerate(price_pivot.index)}

    # ════════════════════════════════════════════════════════
    # 组合模拟: top_n 个仓位槽，每槽 = 1/top_n 资金（始终满仓）
    # 每日: 更新持仓 → 有槽空出则用当日 ML Top-N 补仓
    # 平仓: 5% 硬止损 / 最长 max_hold 天
    # ════════════════════════════════════════════════════════
    positions = []  # 已平仓 {entry_date, exit_date, ret, held_days, stop_hit}
    slots = [None] * top_n  # 每个槽位持有一个 position dict 或 None
    slot_capital = [1.0 / top_n] * top_n  # 每槽独立资金（初始 1/top_n）
    equity_curve = []  # 每日组合净值（0-1）

    all_test_dates = [d for d in test_dates if d in date_positions]
    daily_scores_by_date = score_df.groupby("date")

    def _next_pick(date, today_idx, excluded_codes):
        """取当日 ML 评分最高的未持仓股票。"""
        day = score_df[score_df["date"] == date]
        for _, row in day.sort_values("score", ascending=False).iterrows():
            code = row[COL_TS_CODE]
            if code not in excluded_codes:
                px = price_pivot.iloc[today_idx].get(code)
                if not pd.isna(px):
                    return code, float(px)
        return None, None

    for date in all_test_dates:
        today_idx = date_positions[date]

        # 1) 更新持仓，触发平仓则空出槽位（资金复利结算）
        for s in range(top_n):
            pos = slots[s]
            if pos is None:
                continue
            pos["held_days"] += 1
            px = price_pivot.iloc[today_idx].get(pos["code"])
            if pd.isna(px):
                continue  # 停牌，暂不平仓
            ret = px / pos["entry_price"] - 1
            if ret <= -stop_loss or pos["held_days"] >= max_hold:
                slot_capital[s] *= (1 + ret)  # 复利结算到槽资金
                positions.append({
                    "code": pos["code"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "return": float(ret),
                    "held_days": pos["held_days"],
                    "stop_hit": ret <= -stop_loss,
                })
                slots[s] = None  # 空出槽位

        # 2) 用当日 ML Top-N 补仓空槽
        for s in range(top_n):
            if slots[s] is not None:
                continue
            excluded = {p["code"] for p in slots if p is not None}
            code, px = _next_pick(date, today_idx, excluded)
            if code is None:
                continue
            slots[s] = {
                "code": code, "entry_date": date,
                "entry_price": px, "held_days": 0,
            }

        # 3) 组合净值 = 各槽位资金 × 当前持仓浮盈（满仓跟踪）
        day_value = 0.0
        for s in range(top_n):
            pos = slots[s]
            if pos is None:
                day_value += slot_capital[s]  # 空槽资金按原值
                continue
            px = price_pivot.iloc[today_idx].get(pos["code"])
            pos_val = (px / pos["entry_price"]) if not pd.isna(px) else 1.0
            day_value += slot_capital[s] * pos_val
        equity_curve.append(day_value)

    # 期末平仓（结算最后持仓到槽资金）
    if all_test_dates:
        final_idx = date_positions[all_test_dates[-1]]
        for s in range(top_n):
            pos = slots[s]
            if pos is None:
                continue
            px = price_pivot.iloc[final_idx].get(pos["code"])
            ret = (px / pos["entry_price"] - 1) if not pd.isna(px) else 0.0
            slot_capital[s] *= (1 + ret)
            positions.append({
                "code": pos["code"], "entry_date": pos["entry_date"],
                "exit_date": str(price_pivot.index[final_idx]),
                "return": float(ret), "held_days": pos["held_days"],
                "stop_hit": False,
            })
        # 最终净值 = 5 槽资金总和
        equity_curve.append(sum(slot_capital))

    if not positions or not equity_curve:
        return {"status": "no_trades", "total_return": 0, "sharpe": 0,
                "max_drawdown": 0, "n_trades": 0}

    # 组合统计
    eq = np.array(equity_curve)
    total_ret = eq[-1] / eq[0] - 1
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1))
    daily_rets = np.diff(eq) / eq[:-1]
    mean_dr = np.mean(daily_rets)
    std_dr = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 0
    sharpe = (mean_dr / std_dr * np.sqrt(252)) if std_dr > 0 else 0

    rets = [p["return"] for p in positions]
    win_rate = np.mean([r > 0 for r in rets]) * 100
    stop_count = sum(1 for p in positions if p["stop_hit"])
    win_ratio = np.mean([r for r in rets if r > 0]) if any(r > 0 for r in rets) else 0
    loss_ratio = np.mean([r for r in rets if r < 0]) if any(r < 0 for r in rets) else 0

    result = {
        "status": "success",
        "backtest_period": f"{test_dates[0]} ~ {test_dates[-1]}",
        "n_days": len(all_test_dates),
        "n_trades": len(positions),
        "total_return": round(float(total_ret) * 100, 1),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd) * 100, 1),
        "win_rate": round(float(win_rate), 1),
        "stop_loss_rate": round(float(stop_count / len(positions)) * 100, 1),
        "avg_win": round(float(win_ratio) * 100, 1),
        "avg_loss": round(float(loss_ratio) * 100, 1),
        "profit_loss_ratio": round(float(abs(win_ratio / loss_ratio)) if loss_ratio else 0, 2),
        "strategy": f"ML Top{top_n} | 5%硬止损 | 最长{max_hold}天 | 满仓",
        "_positions": positions,
    }
    logger.info("ML 回测完成: 总收益 %+.1f%% | Sharpe %.2f | MaxDD %.1f%% | %d 笔 | %d 天",
                result["total_return"], result["sharpe"],
                result["max_drawdown"], result["n_trades"], result["n_days"])
    return result



def run_ml_backtest_for_report(
    feature_df: pd.DataFrame | None = None,
    stock_daily_df: pd.DataFrame | None = None,
    top_n: int = 5,
    backtest_days: int = 1000,
) -> dict:
    """为报告运行 ML 回测（懒加载数据）。

    Returns:
        dict 回测摘要（无 _positions，便于 JSON 序列化）
    """
    from config import DATA_DIR, DB_PATH
    import os, sqlite3

    # 懒加载特征矩阵（复用 market_ml 的进程内缓存）
    if feature_df is None:
        try:
            from analysis.market_ml import _load_feature_matrix as _load_fm
            feature_df = _load_fm()
        except Exception:
            cache = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
            if not os.path.exists(cache):
                return {"status": "no_data", "error": "无特征矩阵缓存"}
            feature_df = pd.read_parquet(cache)

    # 懒加载个股日线
    if stock_daily_df is None:
        conn = sqlite3.connect(DB_PATH)
        stock_daily_df = pd.read_sql_query(
            "SELECT ts_code, trade_date, close, amount FROM stock_daily ORDER BY ts_code, trade_date",
            conn,
        )
        conn.close()

    # 加载模型
    from analysis.ml_model import load_model
    models = {"micro": load_model("micro"), "mid": load_model("mid")}
    if models["micro"] is None and models["mid"] is None:
        return {"status": "no_model", "error": "无 ML 模型"}

    result = run_ml_backtest(feature_df, stock_daily_df, models,
                              top_n=top_n, backtest_days=backtest_days)

    # 去掉 _positions（避免大对象）
    if isinstance(result, dict):
        result.pop("_positions", None)
    return result
