"""风险预算研究 · 移动止盈（trailing stop）路径回测

2026-11-21 Claude。用户的核心痛点是「+15% 浮盈 → -5%」，固定止盈会砍掉右尾，
所以必须测「让利润奔跑、跌破移动止损才走」。

路径：t-20（入场代理）→ t（决策日）→ t+20（观察窗）。
  - 峰值 = t-20..当前 的最高价
  - 回撤 = 当前价 / 峰值 - 1
  - 触发：回撤 <= -s（收盘价触发，按触发日收盘成交，保守）
  - MA20 规则：收盘跌破 20 日线
  - 状态条件：只在 fragile/crash 启用移动止损

输出：控制台 + cache/rb_trailing_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb_data
import rb_study

CACHE = "backtest/cache"


def build_paths(ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = rb_data.load()
    adj, close = d["adj"], d["close"]
    ma20 = adj.rolling(20).mean()
    dates = list(adj.index)
    dpos = {d: i for i, d in enumerate(dates)}
    cpos = {c: j for j, c in enumerate(adj.columns)}

    n = len(ev)
    path = np.full((n, 41), np.nan, dtype=np.float32)
    mapath = np.full((n, 41), np.nan, dtype=np.float32)
    for day, idxs in ev.groupby("trade_date").indices.items():
        i = dpos[day]
        if i < 20 or i + 21 > len(dates):
            continue
        cols = [cpos[c] for c in ev.iloc[idxs]["ts_code"]]
        base = adj.values[i - 20, cols].astype(np.float32)
        p = adj.values[i - 20:i + 21, cols].T.astype(np.float32) / base[:, None]
        m = ma20.values[i - 20:i + 21, cols].T.astype(np.float32) / base[:, None]
        path[idxs] = p
        mapath[idxs] = m
    return path, mapath, ev["trade_date"].values


def simulate(path: np.ndarray, mapath: np.ndarray, state: np.ndarray,
             trail: float | np.ndarray | None, use_ma: bool, state_only: bool,
             cost: float = 0.0015, w0: np.ndarray | None = None) -> dict:
    """返回每条事件的退出收益（相对 t 日）、是否触发、退出天数。

    w0: 可选的初始仓位（R3 的减仓权重）；剩余仓位再走移动止损。
    """
    n = len(path)
    entry = path[:, 20]                     # t 日净值（相对 t-20）
    peak_run = np.maximum.accumulate(path, axis=1)
    dd = path / peak_run - 1
    exit_k = np.full(n, 40, dtype=int)
    trig = np.zeros(n, dtype=bool)

    active = np.ones(n, dtype=bool) if not state_only else np.isin(state, ["fragile", "crash"])

    def _hit(k: int) -> np.ndarray:
        if trail is None:
            return np.zeros(n, dtype=bool)
        t = trail if np.isscalar(trail) else trail
        return dd[:, k] <= -t

    # t 日已经跌破止损 → 立即退出（k=20）
    if trail is not None:
        hit = _hit(20) & active
        exit_k[hit] = 20
        trig[hit] = True
        for k in range(21, 41):
            hit = _hit(k) & active & ~trig
            exit_k[hit] = k
            trig[hit] = True
    if use_ma:
        hit = (path[:, 20] < mapath[:, 20]) & active
        exit_k[hit] = 20
        trig[hit] = True
        for k in range(21, 41):
            hit = (path[:, k] < mapath[:, k]) & active & ~trig
            exit_k[hit] = k
            trig[hit] = True

    exit_val = path[np.arange(n), exit_k]
    fwd = exit_val / entry - 1
    if w0 is None:
        fwd_net = fwd - cost
        final_gain = path[np.arange(n), exit_k] - 1 - cost
    else:
        fwd_net = w0 * fwd - cost
        final_gain = (1 + (entry - 1)) * (1 + fwd_net) - 1
    # 峰值浮盈（t 日之前，含 t）
    peak_gain_t = np.max(path[:, :21], axis=1) - 1
    return dict(fwd=fwd_net, trig=trig, exit_k=exit_k, peak_gain=peak_gain_t,
                final_gain=final_gain, active=active)


def summarize(res: dict, label: str) -> dict:
    fwd = res["fwd"]
    fg = res["final_gain"]
    peak = res["peak_gain"]
    return dict(rule=label, n=len(fwd),
                mean=fwd.mean(), median=np.median(fwd),
                cvar5=rb_study.cvar(fwd), p_lt_5=(fwd < -0.05).mean(),
                p_lt_10=(fwd < -0.10).mean(),
                giveback=(peak - fg).mean(),
                trig=res["trig"].mean(),
                avg_exit_days=res["exit_k"].mean() - 20)


def main():
    ev = rb_study.build_events()
    path, mapath, dates = build_paths(ev)
    print(f"paths {path.shape}, 有效 {np.isfinite(path[:, 20]).sum():,}")

    for thr in (0.10, 0.15, 0.20):
        for states, sname in [(("fragile", "crash"), "fragile+crash"),
                              (("normal", "warning", "fragile", "crash"), "all")]:
            m = (ev["r20"].values >= thr) & ev["frag_state"].isin(states).values
            m &= np.isfinite(path[:, 20])
            if m.sum() < 100:
                continue
            p, mp, st = path[m], mapath[m], ev["frag_state"].values[m]
            vol = ev["vol20"].values[m]
            w3 = rb_study.weights(ev[m], "R3")
            w2 = rb_study.weights(ev[m], "R2")
            w6 = rb_study.weights(ev[m], "R6")
            w7 = rb_study.weights(ev[m], "R7")
            specs = [
                ("hold", None, False, False, None),
                ("trail5%", 0.05, False, False, None),
                ("trail8%", 0.08, False, False, None),
                ("trail10%", 0.10, False, False, None),
                ("trail12%", 0.12, False, False, None),
                ("trail_vol1.0", np.clip(1.0 * vol * np.sqrt(10), 0.04, 0.15), False, False, None),
                ("trail_vol1.5", np.clip(1.5 * vol * np.sqrt(10), 0.04, 0.15), False, False, None),
                ("trail_vol2.0", np.clip(2.0 * vol * np.sqrt(10), 0.04, 0.15), False, False, None),
                ("MA20", None, True, False, None),
                ("trail8%+MA20", 0.08, True, False, None),
                ("R3+trail8%", 0.08, False, False, w3),
                ("R6+trail8%", 0.08, False, False, w6),
                ("R2+trail8%", 0.08, False, False, w2),
                ("R6+trail10%", 0.10, False, False, w6),
                ("R7+trail8%", 0.08, False, False, w7),
                ("R7+trail5%", 0.05, False, False, w7),
                ("R3+trail_vol1.5", np.clip(1.5 * vol * np.sqrt(10), 0.04, 0.15), False, False, w3),
            ]
            results = [(label, simulate(p, mp, st, s, ma, so, w0=w))
                       for label, s, ma, so, w in specs]
            base = results[0][1]
            q80 = np.quantile(base["final_gain"], 0.8)
            up = base["final_gain"] >= q80
            denom = np.mean(base["final_gain"][up])
            rows = []
            for label, res in results:
                row = summarize(res, label)
                row["upside_capture"] = (np.mean(res["final_gain"][up]) / denom
                                         if denom != 0 else np.nan)
                rows.append(row)
            df = pd.DataFrame(rows)
            print(f"\n=== r20>={thr:.0%} · {sname} · n={m.sum():,} ===")
            print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            df.insert(0, "sample", f"{thr:.0%}_{sname}")
            df.to_csv(f"{CACHE}/rb_trailing_{int(thr*100)}_{sname.replace('+','_')}.csv", index=False)


if __name__ == "__main__":
    main()
