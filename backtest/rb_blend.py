"""风险预算研究 · 移动止损 + 保留 runner 的折中方案

问题：8% 移动止损在集中组合里把右尾砍得太多（上行捕获 ~19%）。
方案：初减后，剩余仓位一半走 8% 止损，一半持有到 20 日（或 12% 止损）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb_study
import rb_trailing


def metrics(fwd, final, peak, label):
    return dict(rule=label, n=len(fwd), mean=fwd.mean(), median=np.median(fwd),
                cvar5=rb_study.cvar(fwd),
                p_lt_5=(fwd < -0.05).mean(), p_lt_10=(fwd < -0.10).mean(),
                final_median=np.median(final), final_mean=final.mean(),
                giveback=(peak - final).mean())


def main():
    ev = rb_study.build_events()
    path, mapath, _ = rb_trailing.build_paths(ev)
    ok = np.isfinite(path[:, 20])
    ev, path, mapath = ev[ok].reset_index(drop=True), path[ok], mapath[ok]

    subsets = {
        "fragile+crash": ((ev["r20"] >= 0.15) & ev["frag_state"].isin(["fragile", "crash"])).values,
        "second_shock": ((ev["r20"] >= 0.15) & (ev["g60"] < -0.05) & (ev["shock10"] >= 2)).values,
    }
    for name, m in subsets.items():
        p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
        w = rb_study.weights(ev[m], "R7")
        hold = rb_trailing.simulate(p, mp, st, None, False, False)
        c8 = rb_trailing.simulate(p, mp, st, 0.08, False, False, w0=w)
        c12 = rb_trailing.simulate(p, mp, st, 0.12, False, False, w0=w)
        hold_rule = rb_trailing.simulate(p, mp, st, None, False, False, w0=w)
        blend8 = dict(fwd=0.5 * c8["fwd"] + 0.5 * hold_rule["fwd"],
                      final_gain=0.5 * c8["final_gain"] + 0.5 * hold_rule["final_gain"],
                      peak_gain=hold["peak_gain"])
        blend12 = dict(fwd=0.5 * c12["fwd"] + 0.5 * hold_rule["fwd"],
                       final_gain=0.5 * c12["final_gain"] + 0.5 * hold_rule["final_gain"],
                       peak_gain=hold["peak_gain"])
        rows = [metrics(hold["fwd"], hold["final_gain"], hold["peak_gain"], "hold"),
                metrics(c8["fwd"], c8["final_gain"], hold["peak_gain"], "R7+trail8"),
                metrics(c12["fwd"], c12["final_gain"], hold["peak_gain"], "R7+trail12"),
                metrics(blend8["fwd"], blend8["final_gain"], hold["peak_gain"], "R7+50%trail8+50%hold"),
                metrics(blend12["fwd"], blend12["final_gain"], hold["peak_gain"], "R7+50%trail12+50%hold")]
        df = pd.DataFrame(rows)
        q80 = np.quantile(hold["final_gain"], 0.8)
        up = hold["final_gain"] >= q80
        df["upside_capture"] = [np.mean(r["final_gain"][up]) / np.mean(hold["final_gain"][up])
                                for r in [hold, c8, c12, blend8, blend12]]
        print(f"\n=== r20>=15% · {name} · n={m.sum():,} ===")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
