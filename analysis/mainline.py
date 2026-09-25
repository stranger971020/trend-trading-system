# -*- coding: utf-8 -*-
"""
主线识别器 v2（题材/行业动量 + 广度）— 信号扫描定标

用户定义: 某题材/行业内个股上涨有"普遍性"和"强度", 且至少持续一个月;
          参与其中一两周通常可获 ~30%。

v1 教训: 用"20日强度+20日持续性"确认 → 抓的是**已走完**的行情
         (确认后未来10日中位 -0.6%, ≥30%达成率仅 1%, 持续中位 4 天)。
v2 改为**信号扫描**: 一次性算全特征, 对多个候选信号测前瞻分布, 用数据选定义。

数据: sw_index_data.db -> stock_daily; stock_industry_mapping.csv (l1/l2/l3_name)

Changelog
─────────
2026-09-25 Hermes: v2 重写为信号扫描定标, 修正 v1 抓晚问题。
"""

import os
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
INDUSTRY_CSV = os.path.join(PROJECT_ROOT, "data_storage", "stock_industry_mapping.csv")

LEVELS = {"L1": "l1_name", "L2": "l2_name", "L3": "l3_name"}
_STOCK_CACHE = {}


def _load_stocks(level="L2", start="20150101", end="20991231"):
    key = (level, start, end)
    if key in _STOCK_CACHE:
        return _STOCK_CACHE[key]
    col = LEVELS[level]
    m = pd.read_csv(INDUSTRY_CSV, dtype=str)
    code2ind = dict(zip(m["ts_code"], m[col]))
    conn = sqlite3.connect(STOCK_DB)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, ts_code, close FROM stock_daily "
            "WHERE trade_date BETWEEN ? AND ?", conn, params=[start, end])
    finally:
        conn.close()
    df["ind"] = df["ts_code"].map(code2ind)
    df = df.dropna(subset=["ind", "close"]).sort_values(["ts_code", "trade_date"], kind="mergesort")
    g = df.groupby("ts_code", sort=False)["close"]
    for k in (5, 10, 20):
        df[f"r{k}"] = g.transform(lambda s, k=k: s / s.shift(k) - 1)
        df[f"pos{k}"] = (df[f"r{k}"] > 0).astype(float)
    for k in (5, 10, 20):
        df[f"fwd{k}"] = g.transform(lambda s, k=k: s.shift(-k) / s - 1)
    _STOCK_CACHE[key] = df
    return df


def build_industry_frame(level="L2", start="20150101", end="20991231"):
    """逐日逐行业聚合: 强度(strk)/广度(brk) + 前瞻个股收益分布。

    前瞻列为**行业内个股**的分布(用户参与的是主线中的强势股, 故看 P75/P90/达标率)。
    """
    df = _load_stocks(level, start, end)
    agg = df.groupby(["trade_date", "ind"], sort=False).agg(
        n=("r20", "size"),
        str5=("r5", "median"), str10=("r10", "median"), str20=("r20", "median"),
        br5=("pos5", "mean"), br10=("pos10", "mean"), br20=("pos20", "mean"),
        med_fwd5=("fwd5", "median"), med_fwd10=("fwd10", "median"),
    ).reset_index()
    # 前瞻 quantiles / 达标率 (需分离聚合)
    q = df.groupby(["trade_date", "ind"], sort=False).agg(
        p75_fwd10=("fwd10", lambda x: x.quantile(0.75)),
        p90_fwd10=("fwd10", lambda x: x.quantile(0.90)),
        hit15=("fwd10", lambda x: (x >= 0.15).mean()),
        hit30=("fwd10", lambda x: (x >= 0.30).mean()),
    ).reset_index()
    agg = agg.merge(q, on=["trade_date", "ind"], how="left")
    # 行业内截面百分位
    for k in (5, 10, 20):
        agg[f"p_str{k}"] = agg.groupby("trade_date")[f"str{k}"].rank(pct=True) * 100
        agg[f"p_br{k}"] = agg.groupby("trade_date")[f"br{k}"].rank(pct=True) * 100
    agg["ml20"] = 0.5 * agg["p_str20"] + 0.5 * agg["p_br20"]
    return agg.sort_values(["ind", "trade_date"], kind="mergesort")


def signal_scan(frame, level="L2"):
    """候选信号 → 前瞻分布, 用数据选定义。"""
    f = frame.dropna(subset=["med_fwd10"]).copy()
    base = {"n": len(f), "med_fwd10": f["med_fwd10"].median(),
            "p75": f["p75_fwd10"].median(), "p90": f["p90_fwd10"].median(),
            "hit15": f["hit15"].mean(), "hit30": f["hit30"].mean()}
    # 信号定义
    f["d_br5_20"] = f["br5"] - f["br20"]          # 广度扩张
    f["d_r5_20"] = f["str5"] - f["str20"]         # 强度加速
    sigs = {
        "v1: 20d强度&广度双高(top20%)": (f["p_str20"] >= 80) & (f["p_br20"] >= 80),
        "A: 5d强度&广度双高(top20%)": (f["p_str5"] >= 80) & (f["p_br5"] >= 80),
        "B: 5d双高 + 20d未过热(rank<70)": (f["p_str5"] >= 80) & (f["p_br5"] >= 80) & (f["p_str20"] < 70),
        "C: 广度扩张(br5>br20) + 5d强(top30%)": (f["d_br5_20"] > 0) & (f["p_str5"] >= 70),
        "D: 强度加速(r5>r20) + 5d广度top30%": (f["d_r5_20"] > 0) & (f["p_br5"] >= 70),
        "E: 5d强top10% + 广度≥70%": (f["p_str5"] >= 90) & (f["br5"] >= 0.70),
        "F: 冷启动(20d rank<50 且 5d rank≥90)": (f["p_str20"] < 50) & (f["p_str5"] >= 90),
    }
    out = []
    for name, mask in sigs.items():
        s = f[mask]
        if len(s) < 30:
            out.append({"name": name, "n": len(s)})
            continue
        out.append({"name": name, "n": int(len(s)),
                    "med_fwd10": float(s["med_fwd10"].median()),
                    "p75": float(s["p75_fwd10"].median()), "p90": float(s["p90_fwd10"].median()),
                    "hit15": float(s["hit15"].mean()), "hit30": float(s["hit30"].mean())})
    return {"baseline": base, "signals": out}


def current_mainlines(asof_date, level="L2", topk=12):
    """当前主线榜(按 20 日主线分)。**仅取近 200 天窗口**以控制日报耗时。"""
    start = (pd.Timestamp(str(asof_date)) - pd.Timedelta(days=200)).strftime("%Y%m%d")
    f = build_industry_frame(level, start=start, end=str(asof_date))
    a = f[f["trade_date"] == str(asof_date)]
    if a.empty:
        a = f[f["trade_date"] == f["trade_date"].max()]
    a = a.sort_values("ml20", ascending=False)
    rows = [{"ind": r["ind"], "ml20": round(float(r["ml20"]), 1),
             "str5": None if pd.isna(r["str5"]) else round(float(r["str5"]), 4),
             "str20": None if pd.isna(r["str20"]) else round(float(r["str20"]), 4),
             "br5": None if pd.isna(r["br5"]) else round(float(r["br5"]), 3),
             "br20": None if pd.isna(r["br20"]) else round(float(r["br20"]), 3),
             "n": int(r["n"])} for _, r in a.head(topk).iterrows()]
    return {"asof": str(a["trade_date"].iloc[0]), "level": level, "ranking": rows}


if __name__ == "__main__":
    import sys
    lvl = sys.argv[1] if len(sys.argv) > 1 else "L2"
    print(f"构建 {lvl} 特征 ...", file=sys.stderr)
    frame = build_industry_frame(lvl)
    print(f"行数 {len(frame)} · 行业 {frame['ind'].nunique()} · 区间 {frame['trade_date'].min()}~{frame['trade_date'].max()}\n")
    sc = signal_scan(frame, lvl)
    b = sc["baseline"]
    print(f"=== 基线 (全行业全样本) n={b['n']} ===")
    print(f"  行业内个股未来10日: 中位 {b['med_fwd10']*100:+.2f}% | P75 {b['p75']*100:+.2f}% | "
          f"P90 {b['p90']*100:+.2f}% | ≥15% {b['hit15']*100:.1f}% | ≥30% {b['hit30']*100:.2f}%\n")
    print("=== 候选信号对比 (按 P90 排序) ===")
    for s in sorted(sc["signals"], key=lambda x: -x.get("p90", -9)):
        if "p90" not in s:
            print(f"  {s['name']:<40} n={s['n']} (样本不足)"); continue
        print(f"  {s['name']:<40} n={s['n']:5d} | 中位 {s['med_fwd10']*100:+6.2f}% | "
              f"P75 {s['p75']*100:+6.2f}% | P90 {s['p90']*100:+6.2f}% | ≥15% {s['hit15']*100:4.1f}% | ≥30% {s['hit30']*100:4.2f}%")
