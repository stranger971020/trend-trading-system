from __future__ import annotations
"""
参数网格搜索
- 对 L1 行业轮动策略的关键参数进行网格搜索
- 优化目标：夏普比率最大化
- 输出最优参数组合
"""

import json
import logging
import os
import sys
import itertools
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DB_PATH,
    MOMENTUM_LOOKBACK,
    PERSISTENCE_WEIGHTS,
)
from data.industry_daily_updater import load_daily_data, get_db_connection
from backtest.backtest_engine import run_backtest, BacktestResult
from backtest.l1_rotation_strategy import generate_persistence_signal

logger = logging.getLogger(__name__)

# 参数搜索空间
SEARCH_SPACE = {
    "lookback": [10, 15, 20, 30, 60],
    "top_n": [3, 5, 10],
    "w_momentum": [0.20, 0.25, 0.30, 0.35, 0.40],
    "w_slope": [0.15, 0.20, 0.25, 0.30, 0.35],
}


def _weights_from_momentum_slope(w_mom: float, w_slope: float) -> dict:
    """从动量分和斜率权重推导完整四因子权重。

    保持 turn_over 和 relative_strength 的比例不变 (0.20:0.25 → 4:5)。
    """
    remaining = 1.0 - w_mom - w_slope
    if remaining <= 0:
        w_mom = w_mom / (w_mom + w_slope) * 0.55
        w_slope = w_slope / (w_mom + w_slope) * 0.55
        remaining = 0.45
    w_turn = remaining * 4 / 9
    w_rel = remaining * 5 / 9
    return {
        "momentum_score": round(w_mom, 2),
        "return_slope": round(w_slope, 2),
        "turnover_score": round(w_turn, 2),
        "relative_strength": round(w_rel, 2),
    }


def grid_search(
    daily_df: pd.DataFrame,
    start_date: str = "20180101",
    end_date: str = "20260612",
    output_path: str | None = None,
) -> dict:
    """执行网格搜索。

    Args:
        daily_df: 全量 L1 日线数据
        start_date: 回测起始日期
        end_date: 回测截止日期
        output_path: params.json 输出路径

    Returns:
        dict: {"best_params": {...}, "best_sharpe": 1.23, "all_results": [...]}
    """
    # 筛选回测窗口
    daily_df = daily_df.copy()
    daily_df["trade_date_str"] = daily_df["trade_date"].astype(str)
    mask = (daily_df["trade_date_str"] >= start_date) & (daily_df["trade_date_str"] <= end_date)
    bt_df = daily_df[mask].copy()
    # 同时保留起始日前 120 天用于信号计算
    pre_mask = (daily_df["trade_date_str"] >= "20170901") & (daily_df["trade_date_str"] < start_date)
    pre_df = daily_df[pre_mask].copy()
    bt_df = pd.concat([pre_df, bt_df], ignore_index=True)

    logger.info("回测窗口: %s ~ %s, %d 条数据", start_date, end_date, len(bt_df))

    # 生成搜索组合
    keys = SEARCH_SPACE.keys()
    combinations = list(itertools.product(*SEARCH_SPACE.values()))
    total = len(combinations)
    logger.info("网格搜索: %d 个参数组合", total)

    results = []
    best_sharpe = -999
    best_params = None
    best_result = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        lookback = params["lookback"]
        top_n = params["top_n"]
        weights = _weights_from_momentum_slope(params["w_momentum"], params["w_slope"])

        try:
            bt_result = run_backtest(
                daily_df=bt_df,
                signal_func=generate_persistence_signal,
                top_n=top_n,
                min_history=max(60, lookback * 2),
                lookback=lookback,
                weights=weights,
            )

            sharpe = bt_result.sharpe_ratio
            mdd = abs(bt_result.max_drawdown)

            results.append({
                "params": params,
                "weights": weights,
                "sharpe": round(sharpe, 3),
                "calmar": round(bt_result.calmar_ratio, 3),
                "annual_return": round(bt_result.annual_return * 100, 2),
                "max_drawdown": round(mdd * 100, 2),
                "win_rate": round(bt_result.win_rate_daily * 100, 1),
            })

            if sharpe > best_sharpe and mdd < 0.35:
                best_sharpe = sharpe
                best_params = {"lookback": lookback, "top_n": top_n, "weights": weights}
                best_result = bt_result

        except Exception as e:
            logger.debug("组合 %s 失败: %s", params, e)
            continue

        if (i + 1) % 100 == 0:
            logger.info("  进度: %d/%d, 当前最优夏普=%.3f", i + 1, total, best_sharpe)

    # 排序
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    logger.info("=" * 50)
    logger.info("网格搜索完成: %d 个有效组合", len(results))
    logger.info("最优参数: %s", best_params)
    logger.info("最优夏普: %.3f", best_sharpe)

    # 输出 top 10
    for i, r in enumerate(results[:10]):
        logger.info(
            "  #%d 夏普=%.3f 年化=%.1f%% MDD=%.1f%% lookback=%d top_n=%d w_mom=%.2f w_slope=%.2f",
            i + 1, r["sharpe"], r["annual_return"], r["max_drawdown"],
            r["params"]["lookback"], r["params"]["top_n"],
            r["weights"]["momentum_score"], r["weights"]["return_slope"],
        )

    # 保存最优参数
    if output_path and best_params:
        output = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "optimization_target": "sharpe_ratio",
            "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
            "total_combinations": total,
            "valid_results": len(results),
            "best_params": best_params,
            "best_metrics": {
                "sharpe": best_sharpe,
                "annual_return": round(best_result.annual_return * 100, 2) if best_result else None,
                "max_drawdown": round(best_result.max_drawdown * 100, 2) if best_result else None,
            },
            "top_10": results[:10],
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("最优参数已保存至 %s", output_path)

    return {
        "best_params": best_params,
        "best_sharpe": best_sharpe,
        "best_result": best_result,
        "all_results": results,
    }


def load_params(params_path: str) -> Optional[dict]:
    """从 params.json 加载最优参数。"""
    if not os.path.exists(params_path):
        return None
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params")
