# -*- coding: utf-8 -*-
"""
A股市场情绪周期指标 (EOD, 零新增数据源)

数据来源: data_storage/sw_index_data.db -> stock_daily (trade_date/ts_code/open/high/low/close/pre_close/pct_chg/vol/amount)
股票池/板块: data_storage/stock_industry_mapping.csv (market: 主板/创业板/科创板)

情绪六指标 (全部可从 EOD 推导):
  1. 涨停家数        limit_up
  2. 跌停家数        limit_down
  3. 封板率          seal_rate = 封板数 / 曾涨停数
  4. 最高连板高度    max_streak
  5. 连板家数        multi_board (streak>=2)
  6. 涨跌家数比      updown_ratio
  派生: 连板晋级率   promotion_rate = 昨日涨停今日仍涨停 / 昨日涨停

三阶段: 上升期 / 分歧期 / 退潮期 (由 trailing ts_rank 情绪分映射)

Changelog
─────────
2026-09-25 Hermes: 新建。目标服务 V6 日报"现在该不该出手"层(择时)。
                   全部 EOD 可得, 不依赖个股盘中/封单量/龙虎榜等不可得数据。
"""

import os
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
INDUSTRY_CSV = os.path.join(PROJECT_ROOT, "data_storage", "stock_industry_mapping.csv")

_LIMIT_PCT = {"主板": 0.10, "创业板": 0.20, "科创板": 0.20}
_BOARD_CACHE = None
_EPS = 0.011          # 涨停/跌停价容差(元), 吸收四舍五入
_TS_WINDOW = 250      # 情绪分 trailing 排名窗口(交易日)
_TS_MIN = 60

PHASE_UP = "上升期"
PHASE_MID = "分歧期"
PHASE_DOWN = "退潮期"

# 阈值来自 backtest/emotion_cycle_validation_report.md (2636 交易日, 2015-11~2026-09)
#   最冷20%(≤34分) 未来5日 -0.70% / T+1胜率 51.3%
#   最热20%(≥66分) 未来5日 -1.24% / T+1胜率 44.2%
#   跌停≥25家(最多10%) 未来5日 -0.09% / T+1胜率 60.2%  ← 超跌反弹
#   择时叠加(热→半仓/冷→满仓/其余→七五) vs 满仓基线: +0.27pp
SCORE_COLD = 35.0
SCORE_HOT = 65.0
LIMIT_DOWN_CAPITULATION = 25


def emotion_advice(emo_score, limit_down, phase):
    """数据驱动解读 (方向源自回测: 情绪过热→未来偏弱)。"""
    if limit_down is not None and limit_down >= LIMIT_DOWN_CAPITULATION:
        return (f"情绪冰点/恐慌（跌停 {limit_down} 家 ≥{LIMIT_DOWN_CAPITULATION}）——历史 T+1 胜率 60.2%，"
                f"超跌反弹概率抬升，可留意低吸机会，但仍需个股信号确认。")
    if emo_score is None:
        return "情绪数据不足。"
    if emo_score >= SCORE_HOT:
        return (f"情绪过热（{emo_score:.0f} 分）——历史未来 5 日偏弱（T+1 胜率 44.2%），"
                f"宜降低仓位、快进快出，回避高位接力。")
    if emo_score <= SCORE_COLD:
        return (f"情绪偏冷（{emo_score:.0f} 分）——历史未来 5 日相对抗跌（T+1 胜率 51.3%），"
                f"可适度积极，但需等待赚钱效应（涨停家数/晋级率）回升确认。")
    return (f"情绪中性（{emo_score:.0f} 分，{phase}）——无极端信号，按温度计与个体信号操作。")


def _board_map():
    global _BOARD_CACHE
    if _BOARD_CACHE is None:
        try:
            m = pd.read_csv(INDUSTRY_CSV, dtype=str)
            _BOARD_CACHE = dict(zip(m["ts_code"], m["market"]))
        except Exception:
            _BOARD_CACHE = {}
    return _BOARD_CACHE


def _limit_pct_for(code, board):
    if board in _LIMIT_PCT:
        return _LIMIT_PCT[board]
    c = str(code)
    if c.startswith("300") or c.startswith("688"):
        return 0.20
    return 0.10


def load_daily(db_path=STOCK_DB, start="20150101", end="20991231"):
    conn = sqlite3.connect(db_path)
    try:
        q = ("SELECT trade_date, ts_code, open, high, low, close, pre_close, pct_chg, vol, amount "
             "FROM stock_daily WHERE trade_date BETWEEN ? AND ?")
        df = pd.read_sql_query(q, conn, params=[start, end])
    finally:
        conn.close()
    return df


def _flag_limits(df):
    """标注涨停/跌停/曾涨停 + 连板 streak。"""
    if df.empty:
        return df
    board = _board_map()
    df = df.copy()
    df["_board"] = df["ts_code"].map(board)
    df["_lpct"] = [_limit_pct_for(c, b) for c, b in zip(df["ts_code"], df["_board"])]
    lu_price = np.round(df["pre_close"].to_numpy() * (1.0 + df["_lpct"].to_numpy()), 2)
    ld_price = np.round(df["pre_close"].to_numpy() * (1.0 - df["_lpct"].to_numpy()), 2)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    df["is_lu"] = close >= (lu_price - _EPS)
    df["touch_lu"] = high >= (lu_price - _EPS)
    df["is_ld"] = close <= (ld_price + _EPS)

    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort")
    # 连板 streak: 连续涨停计数
    blk = (~df["is_lu"]).groupby(df["ts_code"], sort=False).cumsum()
    df["streak"] = df["is_lu"].groupby([df["ts_code"], blk], sort=False).cumsum().astype(int)
    # 昨日是否涨停
    df["lu_prev"] = df.groupby("ts_code", sort=False)["is_lu"].shift(1).fillna(False).infer_objects(copy=False).astype(bool)
    return df


def _aggregate(df):
    """按交易日聚合六指标 + 晋级率。"""
    if df.empty:
        return pd.DataFrame()
    daily = pd.DataFrame(index=sorted(df["trade_date"].unique()))
    lu = df[df["is_lu"]]
    daily["limit_up"] = lu.groupby("trade_date").size()
    daily["limit_down"] = df[df["is_ld"]].groupby("trade_date").size()
    daily["touched"] = df[df["touch_lu"]].groupby("trade_date").size()
    daily["max_streak"] = lu.groupby("trade_date")["streak"].max()
    daily["multi_board"] = lu[lu["streak"] >= 2].groupby("trade_date").size()
    daily["up_count"] = df[df["pct_chg"] > 0].groupby("trade_date").size()
    daily["down_count"] = df[df["pct_chg"] < 0].groupby("trade_date").size()
    daily["n_stocks"] = df.groupby("trade_date").size()
    # 晋级率
    prev = df[df["lu_prev"]]
    daily["lu_prev_n"] = prev.groupby("trade_date").size()
    daily["lu_prev_up"] = df[df["lu_prev"] & df["is_lu"]].groupby("trade_date").size()
    daily = daily.fillna(0.0)

    daily["seal_rate"] = daily["limit_up"] / daily["touched"].replace(0, np.nan)
    daily["updown_ratio"] = daily["up_count"] / daily["down_count"].replace(0, np.nan)
    daily["promotion_rate"] = daily["lu_prev_up"] / daily["lu_prev_n"].replace(0, np.nan)
    # 市场赚钱效应: 全市场中位数收益(bp)
    daily["med_ret"] = df.groupby("trade_date")["pct_chg"].median()
    return daily


def _ts_rank(s, window=_TS_WINDOW, min_periods=_TS_MIN):
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: (x <= x[-1]).mean(), raw=True)


def _add_score(daily):
    """情绪分(0-100): 四指标 trailing 排名的均值。

    方向约定: 分数越高 = 情绪越热(涨停多/晋级率高/连板高/涨多跌少)。
    过热/过冷语义由回测确定(见 backtest/emotion_cycle_backtest.py)。
    """
    d = daily.copy()
    r_promo = _ts_rank(d["promotion_rate"].fillna(0.0))
    r_streak = _ts_rank(d["max_streak"].fillna(0.0))
    r_ratio = _ts_rank(d["updown_ratio"].fillna(0.0))
    r_seal = _ts_rank(d["seal_rate"].fillna(0.0))
    r_ld = _ts_rank(d["limit_down"].fillna(0.0))            # 高=跌停多=冷
    comp = (r_promo + r_streak + r_ratio + r_seal + (1 - r_ld)) / 5.0
    d["emo_score"] = (comp * 100).round(1)
    return d


def compute_emotion_series(db_path=STOCK_DB, start="20150101", end="20991231"):
    """全区间情绪序列 (含 score/phase), 供回测与快照复用。"""
    raw = load_daily(db_path, start, end)
    flagged = _flag_limits(raw)
    daily = _aggregate(flagged)
    if daily.empty:
        return daily
    daily = _add_score(daily)
    daily["phase"] = daily["emo_score"].apply(_phase_of)
    return daily


def _phase_of(score):
    if pd.isna(score):
        return "未知"
    if score >= 65:
        return PHASE_UP
    if score <= 35:
        return PHASE_DOWN
    return PHASE_MID


def compute_emotion(asof_date, db_path=STOCK_DB, window=400):
    """asof 快照: 六指标 + 情绪分 + 三阶段 + 近5日趋势。

    window 需 >= TS_WINDOW(250) + 缓冲 才能算出有效 trailing 排名。
    """
    # 取足够历史: 用交易日估算的日历缓冲 (window*2 天足够覆盖节假日)
    start = (pd.Timestamp(str(asof_date)) - pd.Timedelta(days=int(window * 2.2))).strftime("%Y%m%d")
    series = compute_emotion_series(db_path, start=start, end=str(asof_date))
    if series.empty:
        return None
    s = series.loc[:str(asof_date)]
    if s.empty:
        return None
    row = s.iloc[-1]
    hist = s.tail(5)

    def _f(v):
        return None if pd.isna(v) else round(float(v), 4)

    return {
        "asof": str(s.index[-1]),
        "n_stocks": int(row["n_stocks"]),
        "limit_up": int(row["limit_up"]),
        "limit_down": int(row["limit_down"]),
        "seal_rate": _f(row["seal_rate"]),
        "max_streak": int(row["max_streak"]),
        "multi_board": int(row["multi_board"]),
        "promotion_rate": _f(row["promotion_rate"]),
        "up_count": int(row["up_count"]),
        "down_count": int(row["down_count"]),
        "updown_ratio": _f(row["updown_ratio"]),
        "emo_score": _f(row["emo_score"]),
        "phase": str(row["phase"]),
        "advice": emotion_advice(_f(row["emo_score"]), int(row["limit_down"]), str(row["phase"])),
        "med_ret_pct": _f(row["med_ret"]),
        "hist5": [
            {"date": str(i), "emo_score": _f(r["emo_score"]), "limit_up": int(r["limit_up"]),
             "max_streak": int(r["max_streak"]), "promotion_rate": _f(r["promotion_rate"]),
             "updown_ratio": _f(r["updown_ratio"])}
            for i, r in hist.iterrows()
        ],
    }


if __name__ == "__main__":
    import sys
    asof = sys.argv[1] if len(sys.argv) > 1 else "20260924"
    info = compute_emotion(asof)
    if not info:
        print("无数据")
        sys.exit(1)
    print(f"=== A股情绪周期 @ {info['asof']} (n={info['n_stocks']}) ===")
    print(f"情绪分: {info['emo_score']} / 100  -> {info['phase']}")
    print(f"涨停 {info['limit_up']} · 跌停 {info['limit_down']} · 封板率 {info['seal_rate']}")
    print(f"最高连板 {info['max_streak']} 板 · 连板家数 {info['multi_board']} · 晋级率 {info['promotion_rate']}")
    print(f"涨 {info['up_count']} / 跌 {info['down_count']} · 比值 {info['updown_ratio']} · 中位收益 {info['med_ret_pct']}%")
    print("近5日:")
    for h in info["hist5"]:
        print(f"  {h['date']} score={h['emo_score']} 涨停={h['limit_up']} 高度={h['max_streak']} 晋级={h['promotion_rate']} 涨跌比={h['updown_ratio']}")
