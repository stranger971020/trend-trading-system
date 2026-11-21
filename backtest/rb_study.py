"""风险预算研究 · 个股事件面板 + 规则回测

2026-11-21 Claude。分析单元 = 个股，不是组合代理。

事件定义：
  t 日收盘，股票在 PIT 100-500 亿带内，近 20 日涨幅 >= 阈值（默认 5%，后续切片）。
  决策在 t 日收盘做出，前向收益从 t 日收盘起算。

关键量：
  accrued20   = adj[t]/adj[t-20]-1        （近 20 日浮盈，代理用户体感）
  peak_gain   = max(adj[t-20..t])/adj[t-20]-1（这段时间的最高浮盈）
  fwd5/10/20  = adj[t+h]/adj[t]-1
  z20         = r20 / (vol20*sqrt(20))    （浮盈折合几个标准差）
  frag_state  = normal/warning/fragile/crash（市场脆弱度，t 日收盘可得）

规则（单次减仓近似；组合层另做路径模拟）：
  R0 持有
  R1 固定止盈：accrued20 >= TP -> 清仓
  R2 z 阶梯：z<1.5 持有；1.5~2.5 减 1/3；2.5~3 减 1/2；>=3 减 2/3
  R3 z × 市场状态（本方案）
  R4 极端崩跌：shock5>=3 -> 30%，否则持有（近似，路径版另测）
  R5 R3 + 极端崩跌

输出：backtest/cache/rb_events.parquet + 控制台摘要
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import rb_data

CACHE = "backtest/cache"
COST = 0.0015  # 单边
R20_THR = 0.05


def _fwd_extreme(adj: pd.DataFrame, h: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """未来 h 日（t+1..t+h）的最大/最小价格，避免 20 次 shift 的内存峰值。"""
    arr = adj.values.astype(np.float32)
    mx = np.full_like(arr, np.nan)
    mn = np.full_like(arr, np.nan)
    for k in range(1, h + 1):
        sh = np.full_like(arr, np.nan)
        sh[:-k] = arr[k:]
        mx = np.fmax(mx, sh)
        mn = np.fmin(mn, sh)
    return (pd.DataFrame(mx, index=adj.index, columns=adj.columns),
            pd.DataFrame(mn, index=adj.index, columns=adj.columns))


def build_events(force: bool = False) -> pd.DataFrame:
    path = os.path.join(CACHE, "rb_events.parquet")
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)

    d = rb_data.load()
    close, adj, band, mkt = d["close"], d["adj"], d["band"], d["market"]
    pct = d["pct"]

    ret = adj.pct_change(fill_method=None)
    r20 = adj / adj.shift(20) - 1
    vol20 = ret.rolling(20).std()
    z20 = r20 / (vol20 * np.sqrt(20))
    r60 = adj / adj.shift(60) - 1
    ma20 = adj.rolling(20).mean()
    above = (adj > ma20)
    amount20 = d["amount"].rolling(20).mean()

    # 近 20 日最高浮盈（含 t 日）
    peak = adj.rolling(21).max() / adj.shift(20) - 1

    fwd5 = adj.shift(-5) / adj - 1
    fwd10 = adj.shift(-10) / adj - 1
    fwd20 = adj.shift(-20) / adj - 1
    fmax, fmin = _fwd_extreme(adj, 20)
    fmax = fmax / adj - 1
    fmin = fmin / adj - 1

    mask = band & (r20 >= R20_THR) & fwd20.notna() & vol20.notna() & z20.notna()
    ii, jj = np.where(mask.values)
    dates = r20.index[ii]
    stocks = r20.columns[jj]

    def col(df):
        return df.values[ii, jj]

    ev = pd.DataFrame({
        "trade_date": dates,
        "ts_code": stocks,
        "r20": col(r20),
        "z20": col(z20),
        "vol20": col(vol20),
        "r60": col(r60),
        "above_ma20_stock": col(above),
        "amount20": col(amount20),
        "peak_gain": col(peak),
        "fwd5": col(fwd5),
        "fwd10": col(fwd10),
        "fwd20": col(fwd20),
        "fwd_max20": col(fmax),
        "fwd_min20": col(fmin),
    })
    # 市场状态按日并入（t 日收盘可得）
    ev = ev.merge(mkt.reset_index().rename(columns={"index": "trade_date"}),
                  on="trade_date", how="left")
    ev.to_parquet(path)
    return ev


# ---------- 规则 ----------

def weights(ev: pd.DataFrame, rule: str, tp: float = 0.15) -> np.ndarray:
    z = ev["z20"].values
    st = ev["frag_state"].values
    shock5 = ev["shock5"].fillna(0).values

    if rule == "R0":
        return np.ones(len(ev))
    if rule == "R1":
        return np.where(ev["r20"].values >= tp, 0.0, 1.0)
    if rule == "R2":
        w = np.ones(len(ev))
        w[(z >= 1.5) & (z < 2.5)] = 2 / 3
        w[(z >= 2.5) & (z < 3.0)] = 0.5
        w[z >= 3.0] = 1 / 3
        return w
    if rule == "R3":
        # 3x3 矩阵的初版：z 三档 × 市场三档
        w = np.ones(len(ev))
        # warning
        m = st == "warning"
        w[m & (z >= 1.5) & (z < 2.5)] = 2 / 3
        w[m & (z >= 2.5)] = 0.5
        # fragile
        m = st == "fragile"
        w[m & (z >= 1.0) & (z < 1.5)] = 3 / 4
        w[m & (z >= 1.5) & (z < 2.5)] = 0.5
        w[m & (z >= 2.5)] = 1 / 3
        # crash
        m = st == "crash"
        w[m & (z >= 0.5) & (z < 1.5)] = 2 / 3
        w[m & (z >= 1.5)] = 1 / 3
        return w
    if rule == "R6":
        # 只看市场状态、不看 z：更少参数，作为 R3 的简约对照
        return np.select([st == "fragile", st == "crash"], [0.75, 0.6], default=1.0)
    if rule == "R7":
        # 用户场景的「第二次冲击」：弱市 + 10日内第二次 80% 下跌 → 初减到 50%
        weak = ev["g60"].values < -0.05
        second = ev["shock10"].fillna(0).values >= 2
        base = np.select([st == "fragile", st == "crash"], [0.75, 0.6], default=1.0)
        return np.where(weak & second, np.minimum(base, 0.5), base)
    if rule == "R4":
        return np.where(shock5 >= 3, 0.3, 1.0)
    if rule == "R5":
        return np.minimum(weights(ev, "R3"), weights(ev, "R4"))
    raise ValueError(rule)


def simulate(ev: pd.DataFrame, rule: str, tp: float = 0.15) -> pd.DataFrame:
    w = weights(ev, rule, tp)
    fwd = ev["fwd20"].values
    acc = ev["r20"].values
    # 减仓部分现金 0 收益；成本只对减掉的仓位收一次
    cost = (1 - w) * COST
    fwd_rule = w * fwd - cost
    hold_gain = (1 + acc) * (1 + fwd) - 1
    rule_gain = (1 + acc) * (1 + fwd_rule) - 1
    peak = ev["peak_gain"].values
    return pd.DataFrame({
        "trade_date": ev["trade_date"].values,
        "ts_code": ev["ts_code"].values,
        "state": ev["frag_state"].values,
        "z20": ev["z20"].values,
        "r20": acc,
        "w": w,
        "fwd20": fwd,
        "fwd_rule": fwd_rule,
        "hold_gain": hold_gain,
        "rule_gain": rule_gain,
        "peak_gain": peak,
        "giveback_hold": peak - hold_gain,
        "giveback_rule": peak - rule_gain,
    })


def boot_ci(x: np.ndarray, by: np.ndarray, n_boot: int = 1000, seed: int = 7) -> tuple[float, float]:
    """按日 block bootstrap：以「日」为重采样单位，保留同日横截面相关性。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": x, "d": by}).dropna()
    groups = [g["x"].values for _, g in df.groupby("d")]
    if not groups:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cvar(x: np.ndarray, q: float = 0.05) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    v = np.percentile(x, q * 100)
    return float(x[x <= v].mean())


def summarize(ev: pd.DataFrame, label: str = "") -> pd.DataFrame:
    rows = []
    for rule in ("R0", "R1", "R2", "R3", "R4", "R5"):
        s = simulate(ev, rule)
        for state in ("all", "normal", "warning", "fragile", "crash"):
            x = s if state == "all" else s[s["state"] == state]
            if len(x) < 30:
                continue
            lo, hi = boot_ci(x["rule_gain"].values, x["trade_date"].values)
            # 上行捕获：hold 前 20% 的样本里，规则保留多少
            q80 = x["hold_gain"].quantile(0.8)
            up = x[x["hold_gain"] >= q80]
            cap = up["rule_gain"].mean() / up["hold_gain"].mean() if up["hold_gain"].mean() != 0 else np.nan
            rows.append(dict(
                rule=rule, state=state, n=len(x), dates=x["trade_date"].nunique(),
                mean_gain=x["rule_gain"].mean(), median_gain=x["rule_gain"].median(),
                cvar5=cvar(x["rule_gain"].values),
                mean_gain_ci_lo=lo, mean_gain_ci_hi=hi,
                giveback=x["giveback_rule"].mean(),
                giveback_hold=x["giveback_hold"].mean(),
                upside_capture=cap,
                avg_w=x["w"].mean(),
            ))
    out = pd.DataFrame(rows)
    out.insert(0, "sample", label)
    return out


def main():
    ev = build_events()
    print(f"事件面板 {len(ev):,} 行，{ev.trade_date.nunique()} 个交易日，"
          f"{ev.trade_date.min()}~{ev.trade_date.max()}")
    print(ev.groupby("frag_state").size().to_string())
    print("\n按 r20 阈值的样本量（弱市=g60<-5%）：")
    weak = ev["g60"] < -0.05
    for thr in (0.10, 0.15, 0.20, 0.25):
        m = ev["r20"] >= thr
        print(f"  r20>={thr:.0%}: 全部 {int(m.sum()):,} / 弱市 {int((m & weak).sum()):,}")

    print("\n=== 信号：弱市 × 高浮盈 的未来 20 日 ===")
    for thr in (0.10, 0.15, 0.20):
        x = ev[(ev["r20"] >= thr) & weak]
        y = ev[ev["r20"] >= thr]
        print(f"r20>={thr:.0%}: 弱市 n={len(x):,} 20日均{x.fwd20.mean()*100:+.2f}% "
              f"中位{x.fwd20.median()*100:+.2f}% P(<-5%){(x.fwd20<-0.05).mean()*100:.0f}% "
              f"CVaR5%{cvar(x.fwd20.values)*100:+.2f}% | 非弱市 n={len(y)-len(x):,} "
              f"均{y[~(y.g60<-0.05)].fwd20.mean()*100:+.2f}%")

    print("\n=== 规则对比（全样本） ===")
    out = summarize(ev, "all")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== R3 vs R0 按市场状态 ===")
    s0 = simulate(ev, "R0"); s3 = simulate(ev, "R3")
    for st in ("normal", "warning", "fragile", "crash"):
        a = s0[s0.state == st]; b = s3[s3.state == st]
        if len(a) < 30:
            continue
        print(f"{st:8s} n={len(a):6d} 持有均{a.fwd20.mean()*100:+.2f}% "
              f"CVaR5%{cvar(a.fwd20.values)*100:+.2f}% | R3均{b.fwd_rule.mean()*100:+.2f}% "
              f"CVaR5%{cvar(b.fwd_rule.values)*100:+.2f}% 平均仓位{b.w.mean():.2f}")

    out.to_csv(os.path.join(CACHE, "rb_rule_summary.csv"), index=False)
    print("\n✅ cache/rb_rule_summary.csv")


if __name__ == "__main__":
    main()
