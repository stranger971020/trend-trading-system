# -*- coding: utf-8 -*-
"""
信号组合消歧 —— 行业臂 × β倾斜 是否重复作用?

背景: T6 β倾斜在全宇宙验证(Calmar 0.88→1.33); 行业臂单独验证。
      两者叠加未测 → 震荡日决策单出现"预期全负"的怪异结果, 疑重复惩罚。
本脚本在 震荡/过热 期上测 4 组合, 用超额口径判定。

输出: backtest/signal_combo_test.md / .json
"""
import json
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.mainline import INDUSTRY_CSV, build_industry_frame  # noqa: E402
from analysis.regime import build_market  # noqa: E402
from analysis.swing_engine import build_labels, feature_columns  # noqa: E402

MATRIX = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data_storage", "swing_models")
CAP_CACHE = os.path.join(PROJECT_ROOT, "data_storage", "mktcap_cache.csv")
OUT_MD = os.path.join(PROJECT_ROOT, "backtest", "signal_combo_test.md")
OUT_JSON = os.path.join(PROJECT_ROOT, "backtest", "signal_combo_test.json")
REPR = np.array([-0.15, -0.02, 0.12, 0.30])
N, HOLD, COST = 20, 20, 0.0015
LOW_YI, HIGH_YI = 1e6, 5e6
BENCH, BETA_COL = "000300.SH", "beta_60d"
LAM = 0.2
FOLDS = [("swing_20201231", "20210101", "20211231"),
         ("swing_20221231", "20230101", "20241231"),
         ("swing_20241231", "20250101", "20261231")]
STATES = ["震荡/轮动", "过热"]


def met(v, idx):
    r = np.asarray(v, float); x = np.asarray(idx, float)
    m = ~np.isnan(r) & ~np.isnan(x); r, x = r[m], x[m]
    if len(r) == 0:
        return {}
    ex = r - x
    return {"n": len(r), "abs": round(float(r.mean()) * 100, 2),
            "excess": round(float(ex.mean()) * 100, 2),
            "rel_win": round(float((ex > 0).mean()) * 100, 1)}


def main():
    fc = feature_columns()
    schema = set(pq.ParquetFile(MATRIX).schema_arrow.names)
    need = [c for c in fc if c in schema]
    for e in ["mom1", "is_20cm_eligible", "mom20", BETA_COL]:
        if e in schema and e not in need:
            need.append(e)
    print("加载特征 ...", file=sys.stderr)
    df = pq.read_table(MATRIX, columns=["ts_code", "trade_date", "close"] + need).to_pandas()
    df = build_labels(df)
    caps = pd.read_csv(CAP_CACHE, dtype={"trade_date": str}) if os.path.exists(CAP_CACHE) else None
    imap = pd.read_csv(INDUSTRY_CSV, dtype=str)
    df["ind"] = df["ts_code"].map(dict(zip(imap["ts_code"], imap["l2_name"])))
    mk = build_market(force=False)[["regime"]].reset_index()
    mk.columns = ["trade_date", "regime"]
    fr = build_industry_frame("L2", start="20201201", end="20261231")
    mlu = fr[["trade_date", "ind", "ml20"]].dropna()
    ml_top = {d: set(g.nlargest(12, "ml20")["ind"]) for d, g in mlu.groupby("trade_date")}

    c = sqlite3.connect(STOCK_DB)
    ix = pd.read_sql_query("SELECT trade_date,close FROM sw_index_daily WHERE ts_code=? ORDER BY trade_date",
                           c, params=[BENCH])
    c.close()
    pos_of = {d: i for i, d in enumerate(ix["trade_date"])}
    iclose = ix["close"].to_numpy()

    frames = []
    for name, t0, t1 in FOLDS:
        p = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            art = pickle.load(f)
        m, feats = art["model"], [x for x in art["features"] if x in schema]
        te = df[(df["trade_date"] >= t0) & (df["trade_date"] <= t1) & (df["y"] >= 0)].copy()
        pr = m.predict_proba(te[feats].astype("float32"))
        te["expect_ret"] = pr @ REPR
        frames.append(te)
    allp = pd.concat(frames, ignore_index=True)

    rows = []
    for d, g in allp.groupby("trade_date"):
        day = g.dropna(subset=["fwd20", "expect_ret", "mom1", "mom20", BETA_COL]).copy()
        thr = np.where(day["is_20cm_eligible"].to_numpy() > 0, 19.5, 9.5)
        day = day[day["mom1"].abs().to_numpy() < thr]
        if caps is not None:
            cc = caps[caps["trade_date"] == d][["ts_code", "total_mv"]]
            day = day.merge(cc, on="ts_code", how="inner")
            day = day[day["total_mv"].between(LOW_YI, HIGH_YI)]
        if len(day) < N:
            continue
        reg = mk[mk["trade_date"] == d]["regime"]
        reg = reg.iloc[0] if len(reg) else None
        if reg not in STATES:
            continue
        i = pos_of.get(d)
        if i is None or i + HOLD >= len(iclose):
            continue
        indm = day.dropna(subset=["ind"]).groupby("ind")["mom20"].median()
        irank = indm.rank(pct=True) * 100
        day = day.assign(ind_rank=day["ind"].map(irank))
        base = day.nlargest(N, "expect_ret")
        # 行业臂: 震荡=主线∩热度带70-90; 过热=主线
        tops = ml_top.get(d) or set()
        if reg == "震荡/轮动":
            sub = day[(day["ind_rank"] >= 70) & (day["ind_rank"] < 90)]
            if tops:
                it = sub[sub["ind"].isin(tops)]
                if len(it) >= N:
                    sub = it
        else:
            sub = day[day["ind"].isin(tops)] if tops else day
        indarm = sub if len(sub) >= N else base
        # β倾斜 (T6: 震荡/过热 均压β)
        day2 = day.assign(_s=day["expect_ret"].rank(pct=True) - LAM * day[BETA_COL].rank(pct=True))
        tilt_full = day2.nlargest(N, "_s")
        sub2 = indarm.assign(_s=indarm["expect_ret"].rank(pct=True) - LAM * indarm[BETA_COL].rank(pct=True))
        tilt_ind = sub2.nlargest(N, "_s")

        idx = float(iclose[i + HOLD] / iclose[i] - 1)
        r = {"date": d, "regime": reg, "idx": idx}
        for k, p_ in {"全行业·无倾斜": base, "全行业·β倾斜": tilt_full,
                      "行业臂·无倾斜": indarm, "行业臂·β倾斜": tilt_ind}.items():
            r[k] = float(p_["fwd20"].mean()) - 0.9 * 2 * COST
        rows.append(r)
    rr = pd.DataFrame(rows)
    print(f"期数 {len(rr)} (震荡+过热)", file=sys.stderr)

    ARMS = ["全行业·无倾斜", "全行业·β倾斜", "行业臂·无倾斜", "行业臂·β倾斜"]
    out = {"overall": {}, "by_regime": {}}
    print(f"\n{'组合':<18}{'n':>5}{'绝对':>9}{'超额':>9}{'相对胜率':>10}", file=sys.stderr)
    for a in ARMS:
        m = met(rr[a], rr["idx"])
        out["overall"][a] = m
        print(f"{a:<18}{m['n']:>5}{m['abs']:>8.2f}%{m['excess']:>8.2f}%{m['rel_win']:>9.0f}%", file=sys.stderr)
    for reg in STATES:
        s = rr[rr["regime"] == reg]
        if len(s) == 0:
            continue
        out["by_regime"][reg] = {a: met(s[a], s["idx"]) for a in ARMS}
        print(f"\n--- {reg} (n={len(s)}) ---", file=sys.stderr)
        for a in ARMS:
            m = out["by_regime"][reg][a]
            print(f"  {a:<18} 超额{m['excess']:+.2f}% 相对胜率{m['rel_win']:.0f}%", file=sys.stderr)
        best = max(ARMS, key=lambda a: out["by_regime"][reg][a].get("excess", -99))
        out["by_regime"][reg]["_best"] = best
        print(f"  → 最优: {best}", file=sys.stderr)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    L = ["# 信号组合消歧（行业臂 × β倾斜）\n", f"> {len(rr)}期(震荡+过热) · 超额口径\n",
         "\n## 全期\n", "| 组合 | 期数 | 绝对 | 超额 | 相对胜率 |", "|---|---|---|---|---|"]
    for a, m in out["overall"].items():
        L.append(f"| {a} | {m['n']} | {m['abs']:+.2f}% | **{m['excess']:+.2f}%** | {m['rel_win']:.0f}% |")
    L.append("\n## 分状态\n")
    L.append("| 状态 | 期数 | " + " | ".join(ARMS) + " | 最优 |")
    L.append("|---" * (len(ARMS) + 3) + "|")
    for reg, dd in out["by_regime"].items():
        L.append(f"| {reg} | {dd[ARMS[0]]['n']} | " + " | ".join(f"{dd[a]['excess']:+.2f}" for a in ARMS)
                 + f" | **{dd['_best']}** |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n✅ {OUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
