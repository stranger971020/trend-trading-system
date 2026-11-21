"""风险预算研究 · 样本外 / 分年度 / 成本敏感性 / 用户场景切片

2026-11-21 Claude。验证 R6+trail8%（市场状态初减 + 8% 移动止损）是否稳健：
  - 分年度（2021~2026）
  - 分样本（2021-2023 校准 / 2024-2026 检验）
  - 成本 0.05/0.15/0.30%
  - 用户场景：峰值浮盈 +12~18% 时，规则最终保住多少
  - 「第二次冲击」：shock10>=2 vs 第一次
  - 按日 block bootstrap 置信区间
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb_data
import rb_study
import rb_trailing

COST = 0.0015


def metrics(res: dict, label: str) -> dict:
    fwd, fg, peak = res["fwd"], res["final_gain"], res["peak_gain"]
    return dict(rule=label, n=len(fwd),
                mean=fwd.mean(), median=np.median(fwd),
                cvar5=rb_study.cvar(fwd),
                p_lt_5=(fwd < -0.05).mean(), p_lt_10=(fwd < -0.10).mean(),
                giveback=(peak - fg).mean(),
                final_mean=fg.mean(), final_median=np.median(fg),
                trig=res["trig"].mean(), exit_days=res["exit_k"].mean() - 20)


def boot_diff(hold: dict, rule: dict, by: np.ndarray, n_boot: int = 1000, seed: int = 5):
    """按日 block bootstrap：mean 差与 CVaR 差的 95% CI。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"d": by, "h": hold["fwd"], "r": rule["fwd"]}).dropna()
    groups = [g for _, g in df.groupby("d")]
    md, cd = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        part = pd.concat([groups[i] for i in pick])
        md.append(part["r"].mean() - part["h"].mean())
        cd.append(rb_study.cvar(part["r"].values) - rb_study.cvar(part["h"].values))
    return (np.percentile(md, [2.5, 97.5]), np.percentile(cd, [2.5, 97.5]))


def main():
    ev = rb_study.build_events()
    path, mapath, _ = rb_trailing.build_paths(ev)
    ok = np.isfinite(path[:, 20])
    ev = ev[ok].reset_index(drop=True)
    path, mapath = path[ok], mapath[ok]

    for thr in (0.10, 0.15, 0.20):
        for states, sname in [(("fragile", "crash"), "fragile+crash"),
                              (("normal", "warning", "fragile", "crash"), "all")]:
            m = (ev["r20"].values >= thr) & ev["frag_state"].isin(states).values
            p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
            w6 = rb_study.weights(ev[m], "R6")
            w7 = rb_study.weights(ev[m], "R7")
            hold = rb_trailing.simulate(p, mp, st, None, False, False)
            trail8 = rb_trailing.simulate(p, mp, st, 0.08, False, False)
            combo = rb_trailing.simulate(p, mp, st, 0.08, False, False, w0=w6)
            combo7 = rb_trailing.simulate(p, mp, st, 0.08, False, False, w0=w7)

            print(f"\n{'='*92}\n### r20>={thr:.0%} · {sname} · n={m.sum():,} · "
                  f"dates={ev[m].trade_date.nunique()}")
            rows = [metrics(hold, "hold"), metrics(trail8, "trail8%"),
                    metrics(combo, "R6+trail8%"), metrics(combo7, "R7+trail8%")]
            print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

            # 按日聚类 CI：R7+trail8 - hold
            md, cd = boot_diff(hold, combo7, ev[m].trade_date.values)
            print(f"R7+trail8 - hold: mean 差 {np.mean(combo7['fwd']-hold['fwd'])*100:+.2f}pp "
                  f"CI[{md[0]*100:+.2f},{md[1]*100:+.2f}] | "
                  f"CVaR5 差 {(rb_study.cvar(combo7['fwd'])-rb_study.cvar(hold['fwd']))*100:+.2f}pp "
                  f"CI[{cd[0]*100:+.2f},{cd[1]*100:+.2f}]")

            # 分年度
            yr = ev[m].trade_date.str[:4].values
            print("\n分年度：")
            out = []
            for y in sorted(set(yr)):
                mm = yr == y
                h = rb_trailing.simulate(p[mm], mp[mm], st[mm], None, False, False)
                c = rb_trailing.simulate(p[mm], mp[mm], st[mm], 0.08, False, False, w0=w6[mm])
                out.append(dict(year=y, n=int(mm.sum()),
                                hold_mean=h["fwd"].mean(), hold_cvar=rb_study.cvar(h["fwd"]),
                                rule_mean=c["fwd"].mean(), rule_cvar=rb_study.cvar(c["fwd"]),
                                rule_p10=(c["fwd"] < -0.10).mean(),
                                rule_final_med=np.median(c["final_gain"])))
            print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

            # 用户场景：峰值浮盈 +12~18%
            print("\n峰值浮盈 +12~18% 的切片（你的 case）：")
            peak = hold["peak_gain"]
            mm = (peak >= 0.12) & (peak <= 0.18)
            if mm.sum() >= 30:
                for label, r in [("hold", hold), ("trail8%", trail8),
                                 ("R6+trail8%", combo), ("R7+trail8%", combo7)]:
                    fg = r["final_gain"][mm]
                    fwd = r["fwd"][mm]
                    print(f"  {label:10s} n={mm.sum():5d} 最终浮盈中位{np.median(fg)*100:+6.2f}% "
                          f"均值{fg.mean()*100:+6.2f}% P(最终<0){(fg<0).mean()*100:3.0f}% "
                          f"P(回吐>10pp){(r['peak_gain'][mm]-fg>0.10).mean()*100:3.0f}% "
                          f"20日均{fwd.mean()*100:+6.2f}% CVaR5{rb_study.cvar(fwd)*100:+6.2f}%")
            else:
                print(f"  样本不足 n={mm.sum()}")

    # 成本敏感性
    print(f"\n{'='*92}\n### 成本敏感性（r20>=15%, fragile+crash）")
    m = (ev["r20"].values >= 0.15) & ev["frag_state"].isin(["fragile", "crash"]).values
    p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
    w6 = rb_study.weights(ev[m], "R6")
    for c in (0.0005, 0.0015, 0.0030):
        r = rb_trailing.simulate(p, mp, st, 0.08, False, False, cost=c, w0=w6)
        print(f"成本{c*100:.2f}%: mean{r['fwd'].mean()*100:+.2f}% CVaR5{rb_study.cvar(r['fwd'])*100:+.2f}% "
              f"P(<-10%){(r['fwd']<-0.10).mean()*100:.0f}% 触发{r['trig'].mean()*100:.0f}%")

    # 第二次冲击
    print(f"\n{'='*92}\n### 「第二次冲击」是否更危险（r20>=15%）")
    x = ev[ev["r20"] >= 0.15].copy()
    x = x[x["g60"] < -0.05]
    for lo, hi, lab in [(0, 1, "无冲击"), (1, 1, "首次冲击(shock10=1)"),
                        (2, 99, "第二次及以上(shock10>=2)")]:
        mm = (x["shock10"] >= lo) & (x["shock10"] <= hi)
        if mm.sum() < 20:
            print(f"  {lab}: n={int(mm.sum())} 样本少")
            continue
        sub = x[mm]
        print(f"  {lab}: n={len(sub):5d} 未来20日均{sub.fwd20.mean()*100:+.2f}% "
              f"中位{sub.fwd20.median()*100:+.2f}% P(<-5%){(sub.fwd20<-0.05).mean()*100:3.0f}% "
              f"CVaR5{rb_study.cvar(sub.fwd20.values)*100:+.2f}%")

    # 第二次冲击场景的规则效果（用户明确要的）
    print(f"\n{'='*92}\n### 第二次冲击场景（r20>=15%, g60<-5%, shock10>=2）")
    m2 = ((ev["r20"] >= 0.15) & (ev["g60"] < -0.05) & (ev["shock10"] >= 2)).values
    if m2.sum() >= 50:
        p2, mp2, st2 = path[m2], mapath[m2], ev["frag_state"].values[m2]
        w72 = rb_study.weights(ev[m2], "R7")
        h2 = rb_trailing.simulate(p2, mp2, st2, None, False, False)
        c2 = rb_trailing.simulate(p2, mp2, st2, 0.08, False, False, w0=w72)
        print(f"n={m2.sum():,} dates={ev[m2].trade_date.nunique()}")
        print(pd.DataFrame([metrics(h2, "hold"), metrics(c2, "R7+trail8%")])
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        md, cd = boot_diff(h2, c2, ev[m2].trade_date.values)
        print(f"mean 差 {np.mean(c2['fwd']-h2['fwd'])*100:+.2f}pp CI[{md[0]*100:+.2f},{md[1]*100:+.2f}] | "
              f"CVaR5 差 {(rb_study.cvar(c2['fwd'])-rb_study.cvar(h2['fwd']))*100:+.2f}pp "
              f"CI[{cd[0]*100:+.2f},{cd[1]*100:+.2f}]")
    else:
        print(f"样本不足 n={int(m2.sum())}")


if __name__ == "__main__":
    main()
