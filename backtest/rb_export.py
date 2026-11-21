"""风险预算研究 · 导出看板/报告用的关键指标 JSON

2026-11-21 Claude。只导出可复核的数字，不在这里下结论。
输出：backtest/risk_budget_metrics.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import rb_study
import rb_trailing
import rb_portfolio_trail

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_budget_metrics.json")


def met(fwd: np.ndarray, final: np.ndarray, peak: np.ndarray) -> dict:
    return {
        "n": int(len(fwd)),
        "mean": round(float(np.mean(fwd)), 4),
        "median": round(float(np.median(fwd)), 4),
        "cvar5": round(rb_study.cvar(fwd), 4),
        "p_lt_5": round(float((fwd < -0.05).mean()), 4),
        "p_lt_10": round(float((fwd < -0.10).mean()), 4),
        "final_median": round(float(np.median(final)), 4),
        "final_mean": round(float(np.mean(final)), 4),
        "giveback": round(float(np.mean(peak - final)), 4),
    }


def main():
    ev = rb_study.build_events()
    path, mapath, _ = rb_trailing.build_paths(ev)
    ok = np.isfinite(path[:, 20])
    ev, path, mapath = ev[ok].reset_index(drop=True), path[ok], mapath[ok]

    out = {
        "schema": 1,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sample": {
            "start": str(ev.trade_date.min()), "end": str(ev.trade_date.max()),
            "universe": "PIT 100-500亿 (mktcap 快照推隐含股本)",
            "threshold": "近20日涨幅>=15%",
            "outcome": "未来20日，t日收盘可得",
            "cost": "单边0.15%",
        },
    }

    # 1) fragile+crash
    m = ((ev["r20"] >= 0.15) & ev["frag_state"].isin(["fragile", "crash"])).values
    p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
    w7 = rb_study.weights(ev[m], "R7")
    h = rb_trailing.simulate(p, mp, st, None, False, False)
    c = rb_trailing.simulate(p, mp, st, 0.08, False, False, w0=w7)
    out["fragile_crash_15"] = {
        "dates": int(ev[m].trade_date.nunique()),
        "hold": met(h["fwd"], h["final_gain"], h["peak_gain"]),
        "rule": met(c["fwd"], c["final_gain"], c["peak_gain"]),
        "avg_initial_trim": round(float(1 - w7.mean()), 4),
        "trigger_rate": round(float(c["trig"].mean()), 4),
    }

    # 2) second shock
    m2 = ((ev["r20"] >= 0.15) & (ev["g60"] < -0.05) & (ev["shock10"] >= 2)).values
    p2, mp2, st2 = path[m2], mapath[m2], ev["frag_state"].values[m2]
    w72 = rb_study.weights(ev[m2], "R7")
    h2 = rb_trailing.simulate(p2, mp2, st2, None, False, False)
    c2 = rb_trailing.simulate(p2, mp2, st2, 0.08, False, False, w0=w72)
    out["second_shock_15"] = {
        "dates": int(ev[m2].trade_date.nunique()),
        "hold": met(h2["fwd"], h2["final_gain"], h2["peak_gain"]),
        "rule": met(c2["fwd"], c2["final_gain"], c2["peak_gain"]),
    }

    # 3) 峰值浮盈 +12~18% 切片（用户 case）
    peak = h["peak_gain"]
    mm = (peak >= 0.12) & (peak <= 0.18)
    sl = {
        "n": int(mm.sum()),
        "hold": met(h["fwd"][mm], h["final_gain"][mm], peak[mm]),
        "rule": met(c["fwd"][mm], c["final_gain"][mm], peak[mm]),
        "hold_p_final_neg": round(float((h["final_gain"][mm] < 0).mean()), 4),
        "rule_p_final_neg": round(float((c["final_gain"][mm] < 0).mean()), 4),
        "hold_p_giveback_gt10": round(float((peak[mm] - h["final_gain"][mm] > 0.10).mean()), 4),
        "rule_p_giveback_gt10": round(float((peak[mm] - c["final_gain"][mm] > 0.10).mean()), 4),
    }
    out["peak_12_18"] = sl

    # 4) 分年度（R7+trail8 vs hold）
    yr = ev[m].trade_date.str[:4].values
    by = []
    for y in sorted(set(yr)):
        mm = yr == y
        hh = rb_trailing.simulate(p[mm], mp[mm], st[mm], None, False, False)
        cc = rb_trailing.simulate(p[mm], mp[mm], st[mm], 0.08, False, False, w0=w7[mm])
        by.append({"year": y, "n": int(mm.sum()),
                   "hold_mean": round(float(hh["fwd"].mean()), 4),
                   "hold_cvar5": round(rb_study.cvar(hh["fwd"]), 4),
                   "rule_mean": round(float(cc["fwd"].mean()), 4),
                   "rule_cvar5": round(rb_study.cvar(cc["fwd"]), 4),
                   "rule_p_lt_10": round(float((cc["fwd"] < -0.10).mean()), 4)})
    out["by_year"] = by

    # 5) 成本敏感性
    costs = []
    for cst in (0.0005, 0.0015, 0.0030):
        cc = rb_trailing.simulate(p, mp, st, 0.08, False, False, cost=cst, w0=w7)
        costs.append({"cost": cst, "mean": round(float(cc["fwd"].mean()), 4),
                      "cvar5": round(rb_study.cvar(cc["fwd"]), 4),
                      "p_lt_10": round(float((cc["fwd"] < -0.10).mean()), 4)})
    out["cost_sensitivity"] = costs

    # 6) 集中组合 1/3/5 只
    df = rb_portfolio_trail.run(ev, path, mapath, state=("fragile", "crash"))
    conc = rb_portfolio_trail.summarize(df)
    out["concentration_fragile_crash"] = conc.round(4).to_dict(orient="records")

    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ {OUT}")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("fragile_crash_15", "second_shock_15", "peak_12_18")},
                     ensure_ascii=False, indent=2)[:1600])


if __name__ == "__main__":
    main()
