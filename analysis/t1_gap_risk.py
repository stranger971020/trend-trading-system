#!/usr/bin/env python3
"""V6 Master Plan — T+1 Gap-Risk Predictor（A股实战特异性修正）。

背景（HERMES-20260801-001）：
A股是 T+1 制度。历史回测表明硬止损常被次日集合竞价跳空低开 (Gap Down)
瞬间穿透——名义「5%」止损实际亏损超 7%。本模块基于本地 stock_daily
计算目标股票过去 N 天里「大涨次日直接跳空低开」的历史频率概率
(T+1 Lock-Risk Score)，并在最终加权总分上强制降权。

实现：
  P(T+1 Lock) = count(大涨日 且 次日跳空低开) / count(大涨日)
  大涨日      ：当日 pct_chg >= big_up_pct（默认 +5%）
  次日跳空低开：下一交易日 open <= pre_close * (1 + gap_down_pct)（默认 -1%）

惩罚逻辑（任务硬要求）：
  P(T+1 Lock) > 40% → 最终加权总分强制扣除 -15.0 分，或直接淘汰（可配置）。
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from typing import Iterable

import numpy as np
import pandas as pd

# 项目根路径引导
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COL_TS_CODE, COL_TRADE_DATE, DB_PATH

logger = logging.getLogger(__name__)

# 默认参数（任务硬要求）
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_BIG_UP_PCT = 5.0        # 单日大涨阈值（%）
DEFAULT_GAP_DOWN_PCT = -1.0     # 次日跳空低开阈值（open/pre_close - 1, %）
DEFAULT_RISK_THRESHOLD = 0.40   # 40% 红线
DEFAULT_PENALTY = -15.0         # 强制降权分数


# ═══════════════════════════════════════════════════════════════
# 1. 单只股票 T+1 Lock-Risk Score
# ═══════════════════════════════════════════════════════════════

def compute_t1_lock_risk_score(
    stock_df: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    big_up_pct: float = DEFAULT_BIG_UP_PCT,
    gap_down_pct: float = DEFAULT_GAP_DOWN_PCT,
) -> dict:
    """计算单只股票过去 N 天的 T+1 Lock-Risk Score。

    Args:
        stock_df: 单只股票日线，至少含 trade_date/open/pre_close/pct_chg，
                  按 trade_date 升序（不排序则内部排序）
        lookback_days: 回看窗口（默认 60 个交易日）
        big_up_pct: 大涨阈值 %
        gap_down_pct: 跳空低开阈值 %（负值）

    Returns:
        {"score": float, "big_up_days": int, "gap_down_after_big_up": int,
         "window_start": str, "window_end": str, "insufficient": bool}
    """
    if stock_df is None or stock_df.empty:
        return {"score": 0.0, "big_up_days": 0, "gap_down_after_big_up": 0,
                "insufficient": True}
    df = stock_df.sort_values(COL_TRADE_DATE).reset_index(drop=True)

    # 补齐 open/pre_close：若缺 pre_close 用 close.shift(1)
    if "pre_close" not in df.columns:
        df["pre_close"] = df["close"].shift(1)
    if "pct_chg" not in df.columns and "close" in df.columns:
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100

    # 回看窗口：取最后 lookback_days 个交易日
    window = df.tail(lookback_days).copy()
    if len(window) < 20:
        return {"score": 0.0, "big_up_days": 0, "gap_down_after_big_up": 0,
                "window_start": str(window[COL_TRADE_DATE].iloc[0]) if len(window) else "",
                "window_end": str(window[COL_TRADE_DATE].iloc[-1]) if len(window) else "",
                "insufficient": True}

    # 大涨日
    big_up = window["pct_chg"].fillna(0) >= big_up_pct
    n_big_up = int(big_up.sum())
    if n_big_up == 0:
        return {"score": 0.0, "big_up_days": 0, "gap_down_after_big_up": 0,
                "window_start": str(window[COL_TRADE_DATE].iloc[0]),
                "window_end": str(window[COL_TRADE_DATE].iloc[-1]),
                "insufficient": False}

    # 次日跳空低开：open <= pre_close * (1 + gap_down_pct/100)
    gap = window["open"] / window["pre_close"] - 1
    gap_down_next = gap.shift(-1).fillna(0) * 100 <= gap_down_pct
    hit = int((big_up & gap_down_next).sum())

    score = round(hit / n_big_up, 4)
    return {
        "score": score,
        "big_up_days": n_big_up,
        "gap_down_after_big_up": hit,
        "window_start": str(window[COL_TRADE_DATE].iloc[0]),
        "window_end": str(window[COL_TRADE_DATE].iloc[-1]),
        "insufficient": False,
    }


def compute_t1_lock_risk_batch(
    stock_daily_df: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    big_up_pct: float = DEFAULT_BIG_UP_PCT,
    gap_down_pct: float = DEFAULT_GAP_DOWN_PCT,
) -> pd.DataFrame:
    """批量计算每只股票的 T+1 Lock-Risk Score。

    Args:
        stock_daily_df: 多只股票日线（含 ts_code）

    Returns:
        DataFrame: [ts_code, score, big_up_days, gap_down_after_big_up,
                    window_start, window_end]
    """
    if stock_daily_df is None or stock_daily_df.empty:
        return pd.DataFrame(columns=["ts_code", "score", "big_up_days", "gap_down_after_big_up"])
    records = []
    for code, grp in stock_daily_df.groupby(COL_TS_CODE):
        r = compute_t1_lock_risk_score(grp, lookback_days, big_up_pct, gap_down_pct)
        records.append({"ts_code": code, **r})
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# 2. 数据加载（本地 stock_daily）
# ═══════════════════════════════════════════════════════════════

def load_stock_daily(
    codes: Iterable[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    db_path: str | None = None,
) -> pd.DataFrame:
    """从本地 stock_daily 加载近 lookback_days 个交易日的日线（含缓冲）。

    Args:
        codes: 股票代码列表；None 则加载全部
        lookback_days: 需要的最小回看天数（内部多取 5 天缓冲）
        db_path: SQLite 路径，默认 config.DB_PATH

    Returns:
        DataFrame: ts_code/trade_date/open/high/low/close/pre_close/pct_chg
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        # 先取全局最近的 lookback_days+5 个交易日
        recent = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM stock_daily ORDER BY trade_date DESC LIMIT ?",
            conn, params=(lookback_days + 5,),
        )
        if recent.empty:
            return pd.DataFrame()
        cutoff = recent["trade_date"].min()
        if codes is not None:
            codes = list(codes)
            ph = ",".join("?" for _ in codes)
            df = pd.read_sql_query(
                f"SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg "
                f"FROM stock_daily WHERE trade_date >= ? AND ts_code IN ({ph}) "
                f"ORDER BY ts_code, trade_date",
                conn, params=[cutoff] + codes,
            )
        else:
            df = pd.read_sql_query(
                "SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg "
                "FROM stock_daily WHERE trade_date >= ? ORDER BY ts_code, trade_date",
                conn, params=[cutoff,],
            )
    finally:
        conn.close()
    logger.info("T+1 gap-risk: 加载 %d 行 (>= %s)", len(df), cutoff)
    return df


# ═══════════════════════════════════════════════════════════════
# 3. 惩罚逻辑（硬要求：>40% 扣 -15 分 或 直接淘汰）
# ═══════════════════════════════════════════════════════════════

def apply_t1_gap_penalty(
    scored_df: pd.DataFrame,
    risk_scores: pd.DataFrame,
    threshold: float = DEFAULT_RISK_THRESHOLD,
    penalty: float = DEFAULT_PENALTY,
    eliminate: bool = False,
    score_col: str = "score",
    risk_col: str = "t1_lock_risk",
) -> pd.DataFrame:
    """对最终加权总分施加 T+1 Gap 惩罚。

    Args:
        scored_df: 评分结果（含 score_col）
        risk_scores: compute_t1_lock_risk_batch 输出（ts_code + score）
        threshold: 风险红线，默认 0.40
        penalty: 超过红线扣分，默认 -15.0
        eliminate: True 则直接淘汰高风险股（默认 False = 降权）
        score_col: 总分列名
        risk_col: 风险分输出列名

    Returns:
        带 risk 列的 DataFrame；risk > threshold 的股票总分被扣 penalty
    """
    if scored_df is None or scored_df.empty:
        return scored_df
    risk = risk_scores[["ts_code", "score"]].rename(columns={"score": risk_col})
    df = scored_df.merge(risk, on="ts_code", how="left")
    df[risk_col] = df[risk_col].fillna(0.0)
    df["t1_high_risk"] = df[risk_col] > threshold

    df[score_col] = df[score_col].astype(float)
    df.loc[df["t1_high_risk"], score_col] += penalty

    n_penalized = int(df["t1_high_risk"].sum())
    logger.info("T+1 闸门: 阈值=%.0f%%, 罚分=%.1f, 命中 %d 只, eliminate=%s",
                threshold * 100, penalty, n_penalized, eliminate)
    if eliminate:
        df = df[~df["t1_high_risk"]].copy()
    return df


# ═══════════════════════════════════════════════════════════════
# 4. 自检
# ═══════════════════════════════════════════════════════════════

def _self_test() -> dict:
    """合成日线验证：大涨次日跳空低开的概率计算与惩罚逻辑。"""
    rng = np.random.default_rng(11)

    def _make_stock(pattern: str, n: int = 70) -> pd.DataFrame:
        """构造日线：每 8 天一个大涨日 (+6%)，次日按 pattern 跳空。

        pattern="gap_hit"  → 大涨次日跳空低开 (-2%)，score 应高
        pattern="gap_clean"→ 大涨次日跳空高开 (+2%)，score 应低
        """
        dates = pd.bdate_range("2026-01-05", periods=n).strftime("%Y%m%d")
        big_days = set(range(5, n - 1, 8))
        close = [10.0]
        for i in range(1, n):
            prev = close[-1]
            if i in big_days:
                close.append(prev * 1.06)          # 当日大涨 +6%
            else:
                close.append(prev * (1 + rng.normal(0, 0.008)))
        close = np.array(close)
        opens = [10.0]
        for i in range(1, n):
            prev_c = close[i - 1]
            if i in big_days:
                opens.append(prev_c * 1.005)        # 大涨日小幅高开
            elif (i - 1) in big_days:
                # 大涨次日
                opens.append(prev_c * (0.98 if pattern == "gap_hit" else 1.02))
            else:
                opens.append(prev_c * (1 + rng.normal(0, 0.003)))
        df = pd.DataFrame({
            COL_TRADE_DATE: dates, "open": np.array(opens), "close": close,
            "high": np.maximum(opens, close) * 1.01, "low": np.minimum(opens, close) * 0.99,
        })
        df["pre_close"] = df["close"].shift(1)
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100
        return df.dropna()

    hit = _make_stock("gap_hit")
    clean = _make_stock("gap_clean")

    r_hit = compute_t1_lock_risk_score(hit)
    r_clean = compute_t1_lock_risk_score(clean)

    # 惩罚逻辑
    scored = pd.DataFrame({
        "ts_code": ["A", "B", "C"],
        "score": [80.0, 75.0, 70.0],
    })
    risks = pd.DataFrame({
        "ts_code": ["A", "B", "C"],
        "score": [0.5, 0.1, 0.45],
    })
    out = apply_t1_gap_penalty(scored, risks)
    penalized = out[out["t1_high_risk"]].sort_values("ts_code")
    assert set(penalized["ts_code"]) == {"A", "C"}
    assert (penalized["score"] == scored.set_index("ts_code").loc[penalized["ts_code"], "score"].values + DEFAULT_PENALTY).all()

    return {
        "gap_hit_score": r_hit["score"],
        "gap_hit_big_up_days": r_hit["big_up_days"],
        "gap_clean_score": r_clean["score"],
        "gap_clean_big_up_days": r_clean["big_up_days"],
        "penalty_applied_codes": sorted(penalized["ts_code"].tolist()),
        "penalty_value": DEFAULT_PENALTY,
        "risk_threshold": DEFAULT_RISK_THRESHOLD,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import json
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))
