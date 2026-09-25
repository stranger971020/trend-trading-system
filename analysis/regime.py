# -*- coding: utf-8 -*-
"""
市场阶段识别器 (Regime) — 服务"按阶段切换策略"的思路

用户思路: A股单一策略不可能一直有效; 核心是**机会成本**——
  市场不好时, 空仓本身就是"反向收益"(避免亏损);
  不同阶段(过冷/过热/震荡轮动/上涨/下跌)需要不同策略。

阶段定义(日频, 市场级):
  过冷/恐慌 : 广度≤20% 且 (指数距250日高点≤-12% 或 跌停≥25家)
  过热      : 距高点≥-5% 且 广度≥60% 且 近20日指数涨幅≥8%
  上涨趋势  : 指数 > MA20 > MA60
  下跌趋势  : 指数 < MA20 < MA60
  震荡/轮动 : 其余

数据: sw_index_daily(000300) + stock_daily + analysis.emotion_cycle

Changelog
─────────
2026-09-25 Hermes: 新建。回应用户"按市场阶段切换策略"思路。
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.emotion_cycle import compute_emotion_series, STOCK_DB  # noqa: E402

INDEX = "000300.SH"
REGIMES = ["过冷/恐慌", "上涨趋势", "过热", "震荡/轮动", "下跌趋势"]


def _index_features():
    conn = sqlite3.connect(STOCK_DB)
    try:
        d = pd.read_sql_query(
            "SELECT trade_date, close FROM sw_index_daily WHERE ts_code=? ORDER BY trade_date",
            conn, params=[INDEX])
    finally:
        conn.close()
    d = d.set_index("trade_date")
    c = d["close"]
    d["ma20"] = c.rolling(20).mean()
    d["ma60"] = c.rolling(60).mean()
    d["dd250"] = c / c.rolling(250).max() - 1
    d["ret20"] = c / c.shift(20) - 1
    return d


def _breadth():
    conn = sqlite3.connect(STOCK_DB)
    try:
        d = pd.read_sql_query(
            "SELECT trade_date, ts_code, close FROM stock_daily WHERE trade_date >= '20150101'", conn)
    finally:
        conn.close()
    d = d.sort_values(["ts_code", "trade_date"], kind="mergesort")
    d["ma20"] = d.groupby("ts_code", sort=False)["close"].transform(lambda s: s.rolling(20).mean())
    d["above"] = (d["close"] > d["ma20"]).astype(float)
    g = d.groupby("trade_date").agg(breadth=("above", "mean"), n_stocks=("above", "size"))
    return g


CACHE = os.path.join(PROJECT_ROOT, "data_storage", "regime_market.parquet")


def build_market(start="20150101", end="20991231", force=False):
    if os.path.exists(CACHE) and not force:
        m = pd.read_parquet(CACHE)
        if len(m) > 0:
            m["regime"] = [_classify(r) for r in m.itertuples()]
            return m
    print("  情绪序列 ...", file=sys.stderr)
    emo = compute_emotion_series(STOCK_DB, start=start, end=end)
    print("  指数特征 ...", file=sys.stderr)
    idx = _index_features()
    print("  广度 ...", file=sys.stderr)
    br = _breadth()
    m = idx.join(br, how="inner").join(
        emo[["emo_score", "limit_up", "limit_down", "updown_ratio", "med_ret"]], how="left")
    m = m.dropna(subset=["ma60", "dd250", "breadth"])
    try:
        m.drop(columns=["regime"]).to_parquet(CACHE)
    except Exception:
        pass
    m["regime"] = [_classify(r) for r in m.itertuples()]
    return m


def _classify(r):
    try:
        if r.breadth <= 0.20 and (r.dd250 <= -0.12 or r.limit_down >= 25):
            return "过冷/恐慌"
        if r.emo_score >= 60 and r.breadth >= 0.55:
            return "过热"
        if r.close > r.ma20 > r.ma60:
            return "上涨趋势"
        if r.close < r.ma20 < r.ma60:
            return "下跌趋势"
        return "震荡/轮动"
    except Exception:
        return "震荡/轮动"


def current_regime(asof=None):
    m = build_market()
    if asof:
        m = m.loc[:str(asof)]
    row = m.iloc[-1]
    return {"asof": str(m.index[-1]), "regime": row["regime"],
            "breadth": round(float(row["breadth"]), 3), "dd250": round(float(row["dd250"]), 4),
            "ret20": round(float(row["ret20"]), 4), "emo_score": float(row["emo_score"]) if pd.notna(row["emo_score"]) else None,
            "limit_up": int(row["limit_up"]) if pd.notna(row["limit_up"]) else None,
            "limit_down": int(row["limit_down"]) if pd.notna(row["limit_down"]) else None,
            "counts": m["regime"].value_counts().to_dict()}


if __name__ == "__main__":
    print("构建市场特征 ...", file=sys.stderr)
    m = build_market()
    print(f"交易日 {len(m)} · {m.index[0]}~{m.index[-1]}\n")
    print("=== 各阶段分布 ===")
    vc = m["regime"].value_counts()
    for k, v in vc.items():
        print(f"  {k:<10} {v:5d} 天 ({v/len(m)*100:4.1f}%)")
    print("\n=== 当前状态 ===")
    r = current_regime()
    for k, v in r.items():
        if k != "counts":
            print(f"  {k}: {v}")
