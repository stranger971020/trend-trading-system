#!/usr/bin/env python3
"""V6 Master Plan — Alpha Decay Check + Volume Shock（严格 T-1 时效约束注入）。

PATCH (HERMES-20260801-005) —— 为 in-flight 任务 HERMES-20260801-004 的
「Alpha Decay Check & Volume Shock」注入严格数据时效边界，防止系统引用
非公开盘口/盘中数据导致 Crash。

═══════════════════════════════════════════════════════════════════
[T+1 STRICT TIMING BOUNDARY]（硬性约束，违反即抛错）
1. Alpha Decay Check (P_decay) 与 Volume Shock (Vol_20d_Spike) 的
   输入必须且只能是「T-1 整日收盘快照」(full-day closing snapshot)：
   每只股票 T-1 交易日的 Close / Vol / Amount，加上截至 T-1 的
   整日收盘序列。绝不允许引用 T 日(当日)的盘中分笔/分钟线数据。
2. Re-calculation Window：北京 9:30 开盘前重算窗口内，系统读取的
   数据截止日 = T-1（昨天一整天的 Close, Vol, Turnover）。
   `load_t1_daily(asof_date=T-1)` 强制 `trade_date <= asof_date`。
3. ❌ DO NOT USE Tushare `stk_mins` API（当日 9:30~15:00 分钟线快照）。
   本模块不 import、不调用任何分钟线接口；唯一数据源是本地
   `stock_daily` 整日 K 线表（full-day closing bars）。
4. ✅ P_decay 只允许在 T-1 整日收盘快照上计算。任何疑似盘中数据
   （minute/time 列、同 (ts_code, trade_date) 多行）都会触发
   `_assert_closing_snapshot_only` 拒绝。

[Volume Shock Constraint Injection]
   Vol_20d_Spike = vol_{T-1} / mean(vol_{T-20 .. T-1})
   分子 = T-1 收盘当天的实际整日成交量（不是实时流盘推演）
   分母 = 过去 20 个交易日的整日平均成交量
   排序/打分只依赖 T-1 及更早的整日收盘快照，不依赖实时流盘数据。
   动量突破要求 vol_20d_spike > 1.5 才给绝对加分；均值回归要求
   vol_20d_spike < 0.7（缩量回踩）才提升置信度。
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

from config import COL_TS_CODE, COL_TRADE_DATE, COL_VOL, DB_PATH

logger = logging.getLogger(__name__)

# ── 严格 T-1 约束常量 ──
T1_DATA_SOURCE = "stock_daily (full-day closing bars)"
BANNED_INTRADAY_API = "tushare.stk_mins"        # 明文禁止：盘中分钟线接口
INTRAday_RISK_COLUMNS = {"time", "minute", "datetime", "freq", "interval"}
T1_SNAPSHOT_ROW_TAG = "is_t1_snapshot"           # 标记「T-1 整日收盘快照」行

# ── Alpha Decay 参数（任务硬要求）──
DEFAULT_DECAY_RATIO = 0.75       # P(Today) < P(Init) * 0.75 → 立即砍仓
DEFAULT_DECAY_COLS = {           # 候选/持仓中的 P(Win) 列（V6 引擎输出）
    "p_init_col": "win_prob_entry",
    "p_today_col": "win_prob_t1",
}
DECAY_CUT_COL = "alpha_decay_cut"

# ── Volume Shock 参数（任务硬要求）──
DEFAULT_VOL_MA_WINDOW = 20          # 过去 20 个交易日
VOL_SPIKE_BREAKOUT_THRESHOLD = 1.5  # 动量突破需量能放大 > 1.5
VOL_SPIKE_BREAKOUT_BONUS_RANGE = (5.0, 8.0)   # 绝对加分 +5 ~ 8 分
VOL_SPIKE_REVERSION_THRESHOLD = 0.7 # 缩量回踩 < 0.7 → Reversion 置信度提升
VOL_SPIKE_REVERSION_BOOST = 1.1     # 缩量回踩置信度加成倍数
VOL_SPIKE_FEATURE_COL = "vol_20d_spike_t1"


# ═══════════════════════════════════════════════════════════════
# 0. 严格 T-1 数据守卫
# ═══════════════════════════════════════════════════════════════

def _assert_closing_snapshot_only(df: pd.DataFrame, context: str = "") -> None:
    """硬校验：输入必须是整日收盘快照，拒绝任何盘中/分钟线形态。

    违规场景（任一即抛 ValueError）：
      1. 存在 intraday 特征列（time / minute / datetime / freq / interval）
      2. 同一 (ts_code, trade_date) 出现多行（分钟线每交易日多行）
      3. 缺少整日收盘必需列（close / vol）
      4. vol 出现负值（整日成交量不可能为负）
    """
    if df is None or df.empty:
        return
    tag = f"[{context}] " if context else ""
    hit = [c for c in INTRAday_RISK_COLUMNS if c in df.columns]
    if hit:
        raise ValueError(
            f"{tag}检测到盘中/分钟线数据列 {hit}，违反 T-1 严格时效约束。"
            f"Alpha Decay / Volume Shock 只允许消费 {T1_DATA_SOURCE}，"
            f"禁止 {BANNED_INTRADAY_API} 等盘中接口。")
    for col in ("close", COL_VOL):
        if col not in df.columns:
            raise ValueError(f"{tag}缺少整日收盘必需列 {col!r}，无法确认是整日快照")
    if COL_TS_CODE in df.columns and COL_TRADE_DATE in df.columns:
        dup = df.duplicated(subset=[COL_TS_CODE, COL_TRADE_DATE]).sum()
        if dup:
            raise ValueError(
                f"{tag}发现 {dup} 行重复 (ts_code, trade_date)——"
                f"同一交易日多行 = 分钟线数据，违反 T-1 严格时效约束。")
    if (df[COL_VOL] < 0).any():
        raise ValueError(f"{tag}vol 出现负值，非合法整日成交量快照")


def latest_t1_snapshot(
    stock_df: pd.DataFrame,
    asof_date: str | None = None,
) -> pd.DataFrame:
    """返回截至 asof_date（默认数据末行）的 T-1 整日收盘快照行。

    语义：北京 9:30 开盘前重算窗口，`asof_date = T-1`，取该股在
    T-1 当天的整日收盘行（Close / Vol / Amount）。若数据里已存在
    更晚（T 日）的行，会被严格剔除——本函数绝不下探当日盘中。
    """
    if stock_df is None or stock_df.empty:
        return pd.DataFrame()
    df = stock_df.sort_values(COL_TRADE_DATE).copy()
    if asof_date is not None:
        df = df[df[COL_TRADE_DATE] <= asof_date]
    if df.empty:
        return df
    last = df.tail(1).copy()
    last[T1_SNAPSHOT_ROW_TAG] = True
    return last


# ═══════════════════════════════════════════════════════════════
# 1. 数据装载（只读本地整日 K 线，截止 T-1，绝无分钟线）
# ═══════════════════════════════════════════════════════════════

def load_t1_daily(
    codes: Iterable[str] | None = None,
    lookback_days: int = DEFAULT_VOL_MA_WINDOW + 5,
    asof_date: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """从本地 stock_daily 加载「截至 asof_date (T-1)」的整日收盘 K 线。

    Re-calculation Window 强制：SQL 只允许 `trade_date <= asof_date`，
    asof_date 缺省取表内最新交易日（即对「今天」而言的 T-1）。本函数
    不接触任何盘中数据源；`BANNED_INTRADAY_API` 在该模块中无任何调用。

    Args:
        codes: 股票代码列表；None 则加载全部
        lookback_days: 需要的最小回看交易日（含缓冲）
        asof_date: 数据截止日（T-1），格式 YYYYMMDD；None → 表内最新
        db_path: SQLite 路径，默认 config.DB_PATH

    Returns:
        DataFrame: ts_code/trade_date/open/high/low/close/pre_close/pct_chg/vol/amount
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        if asof_date is None:
            recent = pd.read_sql_query(
                "SELECT MAX(trade_date) AS md FROM stock_daily", conn)
            asof_date = str(recent["md"].iloc[0]) if not recent.empty else None
        if asof_date is None:
            return pd.DataFrame()
        # 取截至 T-1 的最近 lookback_days 个交易日
        recent = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM stock_daily "
            "WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
            conn, params=(asof_date, lookback_days + 5))
        if recent.empty:
            return pd.DataFrame()
        cutoff = recent["trade_date"].min()
        if codes is not None:
            codes = list(codes)
            ph = ",".join("?" for _ in codes)
            df = pd.read_sql_query(
                f"SELECT ts_code, trade_date, open, high, low, close, "
                f"pre_close, pct_chg, vol, amount FROM stock_daily "
                f"WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({ph}) "
                f"ORDER BY ts_code, trade_date",
                conn, params=[cutoff, asof_date] + codes)
        else:
            df = pd.read_sql_query(
                "SELECT ts_code, trade_date, open, high, low, close, "
                "pre_close, pct_chg, vol, amount FROM stock_daily "
                "WHERE trade_date >= ? AND trade_date <= ? "
                "ORDER BY ts_code, trade_date",
                conn, params=[cutoff, asof_date])
    finally:
        conn.close()
    # 数据源守卫：stock_daily 是整日 K 线表，但再验一遍形态
    _assert_closing_snapshot_only(df, "load_t1_daily")
    logger.info("T-1 daily: %d 行 (T-1<=%s, >=%s)",
                len(df), asof_date, cutoff)
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Volume Shock Factor（T-1 整日量能异动）
# ═══════════════════════════════════════════════════════════════

def compute_volume_shock(
    stock_df: pd.DataFrame,
    asof_date: str | None = None,
    ma_window: int = DEFAULT_VOL_MA_WINDOW,
) -> dict:
    """计算单只股票 T-1 的 Vol_20d_Spike（量能异动因子）。

    Vol_20d_Spike = vol_{T-1} / mean(vol_{T-ma_window .. T-1})
      分子 = T-1 收盘当天的实际整日成交量
      分母 = 过去 ma_window 个交易日的整日平均成交量
    排序打分只依赖 T-1 及更早的整日收盘快照，不依赖实时流盘。

    Args:
        stock_df: 单只股票日线（含 trade_date/vol），按 trade_date 升序
        asof_date: T-1 截止日（YYYYMMDD），None → 取数据内最新
        ma_window: 量能均线窗口（默认 20 个交易日）

    Returns:
        {"vol_20d_spike": float, "vol_t1": float, "vol_ma20": float,
         "snapshot_date": str, "insufficient": bool}
    """
    if stock_df is None or stock_df.empty:
        return {"vol_20d_spike": 1.0, "vol_t1": 0.0, "vol_ma20": 0.0,
                "snapshot_date": None, "insufficient": True}
    df = stock_df.sort_values(COL_TRADE_DATE).reset_index(drop=True)
    if asof_date is not None:
        df = df[df[COL_TRADE_DATE] <= asof_date]
    if len(df) < 2:
        return {"vol_20d_spike": 1.0, "vol_t1": float(df[COL_VOL].iloc[-1]),
                "vol_ma20": float(df[COL_VOL].iloc[-1]) if len(df) else 0.0,
                "snapshot_date": str(df[COL_TRADE_DATE].iloc[-1]) if len(df) else None,
                "insufficient": True}
    vol = df[COL_VOL].astype(float)
    ma_vol = vol.rolling(ma_window, min_periods=1).mean()
    vol_t1 = float(vol.iloc[-1])          # 分子：T-1 整日实际量能
    ma20 = float(ma_vol.iloc[-1])         # 分母：过去 20 日整日平均量能
    spike = round(vol_t1 / ma20, 4) if ma20 > 0 else 1.0
    return {
        "vol_20d_spike": spike,
        "vol_t1": vol_t1,
        "vol_ma20": ma20,
        "snapshot_date": str(df[COL_TRADE_DATE].iloc[-1]),
        "insufficient": len(df) < ma_window,
    }


def compute_volume_shock_batch(
    stock_daily_df: pd.DataFrame,
    asof_date: str | None = None,
    ma_window: int = DEFAULT_VOL_MA_WINDOW,
) -> pd.DataFrame:
    """批量计算每只股票的 T-1 Vol_20d_Spike。"""
    if stock_daily_df is None or stock_daily_df.empty:
        return pd.DataFrame(columns=[COL_TS_CODE, VOL_SPIKE_FEATURE_COL, "snapshot_date"])
    _assert_closing_snapshot_only(stock_daily_df, "compute_volume_shock_batch")
    records = []
    for code, grp in stock_daily_df.groupby(COL_TS_CODE):
        r = compute_volume_shock(grp, asof_date=asof_date, ma_window=ma_window)
        records.append({COL_TS_CODE: code, **r})
    out = pd.DataFrame(records)
    out[VOL_SPIKE_FEATURE_COL] = out["vol_20d_spike"].fillna(1.0)
    return out


def add_volume_shock_features(
    feature_df: pd.DataFrame,
    vol_col: str = COL_VOL,
    asof_date: str | None = None,
    ma_window: int = DEFAULT_VOL_MA_WINDOW,
) -> pd.DataFrame:
    """给特征矩阵追加 T-1 Vol_20d_Spike 列（严格 T-1）。

    输出列：
      vol_20d_spike_t1 : Vol_20d_Spike 量能异动比
      is_t1_snapshot   : 是否为 T-1 整日收盘快照行（供 9:30 重算窗口使用）

    说明：本函数只消费整日收盘 K 线（trade_date + vol），逐股票
    rolling(ma_window).mean() 计算分母，取该股最新一天（T-1）作分子。
    若 feature_df 已含 vol 列，可直接在原矩阵上追加。
    """
    if feature_df is None or feature_df.empty or vol_col not in feature_df.columns:
        return feature_df
    _assert_closing_snapshot_only(feature_df, "add_volume_shock_features")
    out = feature_df.copy()
    if COL_TS_CODE not in out.columns:
        return out
    spike = out.groupby(COL_TS_CODE)[vol_col].transform(
        lambda x: x.rolling(ma_window, min_periods=1).mean())
    out[VOL_SPIKE_FEATURE_COL] = (out[vol_col].astype(float) / spike.replace(0, np.nan)).round(4)
    out[VOL_SPIKE_FEATURE_COL] = out[VOL_SPIKE_FEATURE_COL].fillna(1.0)
    out[T1_SNAPSHOT_ROW_TAG] = False
    out.loc[out.groupby(COL_TS_CODE)[COL_TRADE_DATE].idxmax(), T1_SNAPSHOT_ROW_TAG] = True
    return out


# ═══════════════════════════════════════════════════════════════
# 3. Alpha Decay Check（T-1 动态衰退止损）
# ═══════════════════════════════════════════════════════════════

def compute_alpha_decay_decision(
    p_init: float,
    p_today: float,
    decay_ratio: float = DEFAULT_DECAY_RATIO,
) -> dict:
    """单票 Alpha Decay 决策。

    Rule（任务硬要求）：持有股第二天早上重算的 V6 胜率
    P(Today) < P(Init) * 0.75 → 立即直接砍仓（不依赖盘中固定止损）。

    P(Today) 必须是在 T-1 整日收盘快照上重算的胜率（见模块头约束）。

    Returns:
        {"cut": bool, "p_init": float, "p_today": float,
         "threshold": float, "decay": float}
    """
    p_init = float(p_init or 0.0)
    p_today = float(p_today or 0.0)
    threshold = round(p_init * decay_ratio, 4)
    decay = round(p_today / p_init, 4) if p_init > 0 else 1.0
    return {
        "cut": bool(p_today < threshold),
        "p_init": p_init,
        "p_today": p_today,
        "threshold": threshold,
        "decay": decay,
    }


def compute_alpha_decay_batch(
    holdings_df: pd.DataFrame,
    p_today_df: pd.DataFrame,
    decay_ratio: float = DEFAULT_DECAY_RATIO,
    p_init_col: str = DEFAULT_DECAY_COLS["p_init_col"],
    p_today_col: str = DEFAULT_DECAY_COLS["p_today_col"],
) -> pd.DataFrame:
    """批量 Alpha Decay 决策（持仓评估）。

    Args:
        holdings_df: 持仓表，须含 ts_code + p_init_col（入场时 P(Win)）
        p_today_df:  今日重算表，须含 ts_code + p_today_col（T-1 快照上
                     重算的 P(Win)；数据窗口由 load_t1_daily 保证只到 T-1）
        decay_ratio: 砍仓阈值比例（默认 0.75）

    Returns:
        DataFrame: [ts_code, p_init, p_today, threshold, decay,
                    alpha_decay_cut]
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(columns=[COL_TS_CODE, DECAY_CUT_COL])
    h = holdings_df[[COL_TS_CODE, p_init_col]].copy()
    p = p_today_df[[COL_TS_CODE, p_today_col]].copy()
    df = h.merge(p, on=COL_TS_CODE, how="left")
    df = df.dropna(subset=[p_init_col])
    df = df.fillna({p_today_col: 0.0})
    rows = df.apply(
        lambda r: compute_alpha_decay_decision(r[p_init_col], r[p_today_col],
                                               decay_ratio=decay_ratio),
        axis=1, result_type="expand")
    df = pd.concat([df[[COL_TS_CODE, p_init_col, p_today_col]].reset_index(drop=True),
                    rows.reset_index(drop=True)], axis=1)
    df[DECAY_CUT_COL] = df["cut"].astype(bool)
    n_cut = int(df[DECAY_CUT_COL].sum())
    logger.info("Alpha Decay: 评估 %d 只, 触发砍仓 %d 只 (P(Today)<P(Init)*%.2f)",
                len(df), n_cut, decay_ratio)
    return df


def apply_alpha_decay_cut(
    scored_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """对候选/持仓施加 Alpha Decay 砍仓（持仓评估模块钩子）。

    触发 DECAY_CUT_COL=True 的行被直接移出组合（立即直接砍仓，
    不依赖盘中固定止损）。

    Args:
        scored_df: 候选或持仓评分表（含 ts_code + score_col）
        decay_df:  compute_alpha_decay_batch 输出（ts_code + alpha_decay_cut）

    Returns:
        剔除触发砍仓行后的 DataFrame，附 alpha_decay_cut 列。
    """
    if scored_df is None or scored_df.empty:
        return scored_df
    if decay_df is None or decay_df.empty:
        out = scored_df.copy()
        out[DECAY_CUT_COL] = False
        return out
    keep = decay_df[[COL_TS_CODE, DECAY_CUT_COL]]
    out = scored_df.merge(keep, on=COL_TS_CODE, how="left")
    out[DECAY_CUT_COL] = out[DECAY_CUT_COL].fillna(False).astype(bool)
    n_cut = int(out[DECAY_CUT_COL].sum())
    if n_cut:
        logger.info("Alpha Decay 砍仓: 移出 %d 只", n_cut)
    return out[~out[DECAY_CUT_COL]].copy()


# ═══════════════════════════════════════════════════════════════
# 4. Volume Shock 引擎加分（strategy_feature_masker 集成钩子）
# ═══════════════════════════════════════════════════════════════

def apply_volume_shock_bonus(
    scored_df: pd.DataFrame,
    spike_col: str = VOL_SPIKE_FEATURE_COL,
    breakout_threshold: float = VOL_SPIKE_BREAKOUT_THRESHOLD,
    breakout_bonus_range: tuple[float, float] = VOL_SPIKE_BREAKOUT_BONUS_RANGE,
    reversion_threshold: float = VOL_SPIKE_REVERSION_THRESHOLD,
    reversion_boost: float = VOL_SPIKE_REVERSION_BOOST,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """给引擎评分施加绝对量能验证加分（严格 T-1 量能）。

    规则（HERMES-20260801-004 + 005 分母/分子语义）：
      - Breakout / Momentum 突破：vol_20d_spike > 1.5 → 绝对加分
        bonus ∈ [breakout_bonus_range]（+5 ~ 8 分），量能越大加分越高，
        确认资金真实入场。
      - Reversion 缩量回踩：vol_20d_spike < 0.7 → 置信度 ×reversion_boost
        （防止「放量下跌没到底」的假反弹）。

    Args:
        scored_df: 评分表（含 spike_col 与 score_col）
        spike_col: Vol_20d_Spike 列名
        score_col: 被加分的绝对评分列（注意：0-1 概率列不适合直接加 5~8 分，
                   调用方应传入绝对评分列或先缩放）

    Returns:
        带 vol_shock_bonus / vol_shock_conf_boost 列的 DataFrame。
    """
    if scored_df is None or scored_df.empty or spike_col not in scored_df.columns:
        return scored_df
    out = scored_df.copy()
    spike = out[spike_col].fillna(1.0).astype(float)
    lo, hi = breakout_bonus_range
    # 突破加分：> 阈值后在 [lo, hi] 内按量能线性插值
    out["vol_shock_bonus"] = np.where(
        spike > breakout_threshold,
        np.round(lo + (hi - lo) * np.clip((spike - breakout_threshold) /
                                          (3.0 - breakout_threshold), 0, 1), 2),
        0.0)
    # 缩量回踩：< 阈值 → 置信度加成
    out["vol_shock_conf_boost"] = np.where(
        spike < reversion_threshold, reversion_boost, 1.0)
    if score_col in out.columns:
        out[score_col] = out[score_col].astype(float) + out["vol_shock_bonus"]
        out[score_col] = out[score_col] * out["vol_shock_conf_boost"]
    n_break = int((spike > breakout_threshold).sum())
    n_damp = int((spike < reversion_threshold).sum())
    logger.info("Volume Shock: 突破加分 %d 只, 缩量回踩 %d 只", n_break, n_damp)
    return out


# ═══════════════════════════════════════════════════════════════
# 5. 自检
# ═══════════════════════════════════════════════════════════════

def _self_test() -> dict:
    """合成整日收盘数据验证：T-1 时效约束 + Alpha Decay + Volume Shock。"""
    rng = np.random.default_rng(7)
    n = 40
    dates = pd.bdate_range("2026-06-01", periods=n).strftime("%Y%m%d").tolist()

    def _make_stock(base_vol: float, last_vol: float) -> pd.DataFrame:
        vols = rng.normal(base_vol, base_vol * 0.15, n)
        vols = np.clip(vols, base_vol * 0.5, base_vol * 1.5)
        vols[-1] = last_vol                       # T-1 量能 = 分子
        closes = np.cumprod(1 + rng.normal(0, 0.01, n)) * 10
        df = pd.DataFrame({
            COL_TRADE_DATE: dates, "close": closes,
            COL_VOL: vols.astype(float), "amount": vols * closes,
            "open": closes * 0.99, "high": closes * 1.02, "low": closes * 0.98,
            "pre_close": np.roll(closes, 1), "pct_chg": 0.0,
        })
        df["pre_close"] = df["close"].shift(1)
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100
        df[COL_TS_CODE] = "X"
        return df

    # ① 放量突破票：T-1 量能 2.2×MA20 → spike 高
    boom = _make_stock(1_000_000, 2_200_000)
    # ② 缩量回踩票：T-1 量能 0.5×MA20 → spike 低
    quiet = _make_stock(1_000_000, 500_000)
    both = pd.concat([boom.assign(**{COL_TS_CODE: "BOOM"}), quiet.assign(**{COL_TS_CODE: "QUIET"})])

    # T-1 守卫：拒绝盘中数据形态
    guard_rejected = False
    try:
        bad = both.copy()
        bad["time"] = "09:31:00"
        _assert_closing_snapshot_only(bad, "self_test")
    except ValueError:
        guard_rejected = True

    # ② Volume Shock 批量
    spike_df = compute_volume_shock_batch(both, asof_date=dates[-1])
    s_boom = spike_df.set_index(COL_TS_CODE).loc["BOOM"]["vol_20d_spike"]
    s_quiet = spike_df.set_index(COL_TS_CODE).loc["QUIET"]["vol_20d_spike"]
    assert s_boom > 1.5, f"放量票 spike 应 >1.5, got {s_boom}"
    assert s_quiet < 0.7, f"缩量票 spike 应 <0.7, got {s_quiet}"

    # ③ Alpha Decay 决策
    holdings = pd.DataFrame({
        COL_TS_CODE: ["BOOM", "QUIET"],
        "win_prob_entry": [0.80, 0.62],       # P(Init)
    })
    p_today = pd.DataFrame({
        COL_TS_CODE: ["BOOM", "QUIET"],
        "win_prob_t1": [0.78, 0.40],          # T-1 快照重算 P(Today)
    })
    decay = compute_alpha_decay_batch(holdings, p_today)
    d_quiet = decay.set_index(COL_TS_CODE).loc["QUIET"]
    d_boom = decay.set_index(COL_TS_CODE).loc["BOOM"]
    assert bool(d_quiet[DECAY_CUT_COL]) is True, "QUIET P(0.40)<0.62*0.75 → 应砍仓"
    assert bool(d_boom[DECAY_CUT_COL]) is False, "BOOM P(0.78)>0.80*0.75 → 不应砍仓"

    # ④ 砍仓应用
    port = pd.DataFrame({
        COL_TS_CODE: ["BOOM", "QUIET"], "composite_score": [0.85, 0.62],
    })
    out = apply_alpha_decay_cut(port, decay)
    assert "QUIET" not in out[COL_TS_CODE].tolist(), "QUIET 应被移出组合"
    assert "BOOM" in out[COL_TS_CODE].tolist(), "BOOM 应保留"

    # ⑤ Volume Shock 加分（绝对评分列）
    scored = spike_df.rename(columns={COL_TS_CODE: "ts_code"})
    scored["composite_score"] = [80.0, 75.0]
    scored = apply_volume_shock_bonus(scored)
    b_row = scored.set_index("ts_code").loc["BOOM"]
    assert b_row["vol_shock_bonus"] >= 5.0, "放量突破应 +5~8 绝对加分"

    return {
        "t1_guard_rejects_intraday": guard_rejected,
        "t1_snapshot_date": dates[-1],
        "volume_shock": {
            "boom_spike": s_boom, "quiet_spike": s_quiet,
            "threshold_breakout": VOL_SPIKE_BREAKOUT_THRESHOLD,
            "threshold_reversion": VOL_SPIKE_REVERSION_THRESHOLD,
        },
        "alpha_decay": {
            "decay_ratio": DEFAULT_DECAY_RATIO,
            "quiet_cut": bool(d_quiet[DECAY_CUT_COL]),
            "boom_kept": bool(not d_boom[DECAY_CUT_COL]),
            "cut_applied": "QUIET" not in out[COL_TS_CODE].tolist(),
        },
        "volume_shock_bonus_boom": float(b_row["vol_shock_bonus"]),
        "banned_intraday_api": BANNED_INTRADAY_API,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import json
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))
