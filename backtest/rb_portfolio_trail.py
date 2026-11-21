"""风险预算研究 · 集中组合（1/3/5只）的移动止损效果

2026-11-21 Claude。事件级 CVaR 会被分散掉；用户是 1-3 只，必须看组合层。
规则：R7+trail8%（弱市第二次冲击初减到 50%，其余仓位 8% 移动止损）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb_study
import rb_trailing


def run(ev, path, mapath, thr=0.15, state=("fragile", "crash"),
        n_draw=500, seed=17) -> pd.DataFrame:
    m = (ev["r20"].values >= thr) & ev["frag_state"].isin(state).values
    x = ev[m].reset_index(drop=True)
    p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
    w = rb_study.weights(x, "R7")
    h = rb_trailing.simulate(p, mp, st, None, False, False)
    c = rb_trailing.simulate(p, mp, st, 0.08, False, False, w0=w)
    fwd = p[:, 20:41] / p[:, 20][:, None]          # t→t+20，相对 t
    exit_idx = np.clip(c["exit_k"] - 20, 0, 20)
    rule_path = fwd.copy()
    for i in range(len(rule_path)):
        k = exit_idx[i]
        rule_path[i, k + 1:] = fwd[i, k]           # 退出后现金
    rule_val = w[:, None] * rule_path + (1 - w[:, None])

    rng = np.random.default_rng(seed)
    rows = []
    for d, g in x.groupby("trade_date").indices.items():
        if len(g) < 3:
            continue
        for n in (1, 3, 5):
            if len(g) < n:
                continue
            pick = rng.integers(0, len(g), size=(n_draw, n))
            hv = fwd[g[pick]].mean(axis=1)          # 对选中的股票取等权 → n_draw x 21
            rv = rule_val[g[pick]].mean(axis=1)
            h_mdd = (hv / np.maximum.accumulate(hv, axis=1) - 1).min(axis=1)
            r_mdd = (rv / np.maximum.accumulate(rv, axis=1) - 1).min(axis=1)
            rows.append(pd.DataFrame({
                "date": d, "n": n,
                "hold": h["fwd"][g[pick]].mean(axis=1),
                "rule": c["fwd"][g[pick]].mean(axis=1),
                "hold_mdd": h_mdd, "rule_mdd": r_mdd,
                "hold_final": h["final_gain"][g[pick]].mean(axis=1),
                "rule_final": c["final_gain"][g[pick]].mean(axis=1),
            }))
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for n in (1, 3, 5):
        x = df[df.n == n]
        if x.empty:
            continue
        q80 = x["hold"].quantile(0.8)
        up = x[x["hold"] >= q80]
        for col, lab in [("hold", "hold"), ("rule", "R7+trail8")]:
            out.append(dict(
                n=n, rule=lab, draws=len(x),
                mean=x[col].mean(), median=x[col].median(),
                cvar5=rb_study.cvar(x[col].values),
                p_lt_10=(x[col] < -0.10).mean(),
                mdd_mean=x[col + "_mdd"].mean(),
                mdd_p95=x[col + "_mdd"].quantile(0.05),
                upside_capture=(up[col].mean() / up["hold"].mean()
                                if up["hold"].mean() else np.nan),
            ))
    return pd.DataFrame(out)


def main():
    ev = rb_study.build_events()
    path, mapath, _ = rb_trailing.build_paths(ev)
    ok = np.isfinite(path[:, 20])
    ev, path, mapath = ev[ok].reset_index(drop=True), path[ok], mapath[ok]

    for states, label in [(("fragile", "crash"), "fragile+crash"),
                          (("normal", "warning", "fragile", "crash"), "all")]:
        df = run(ev, path, mapath, state=states)
        print(f"\n=== r20>=15% · {label} · 集中组合 ===")
        print(summarize(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 第二次冲击
    m = ((ev["r20"] >= 0.15) & (ev["g60"] < -0.05) & (ev["shock10"] >= 2)).values
    sub = ev[m].reset_index(drop=True)
    p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
    w = rb_study.weights(sub, "R7")
    h = rb_trailing.simulate(p, mp, st, None, False, False)
    c = rb_trailing.simulate(p, mp, st, 0.08, False, False, w0=w)
    print(f"\n=== 第二次冲击 · 事件级 n={m.sum():,} ===")
    print(f"hold: mean{h['fwd'].mean()*100:+.2f}% CVaR{rb_study.cvar(h['fwd'])*100:+.2f}% "
          f"P(<-10%){(h['fwd']<-0.10).mean()*100:.0f}%")
    print(f"rule: mean{c['fwd'].mean()*100:+.2f}% CVaR{rb_study.cvar(c['fwd'])*100:+.2f}% "
          f"P(<-10%){(c['fwd']<-0.10).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
