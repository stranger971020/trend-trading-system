"""浮盈阈值扫描：找平台区，而不是拟合用户的个案

问题：用户 +15% 是个例。真正该问的是「哪个阈值有效、是否稳健」。

两层检验：
  A) 个股横截面（样本最大）：近 20 日涨幅达到 T 的票 → 未来 5 日跌 >5% 的概率
     —— 找「尾部概率随 T 跃升」的位置，以及它是否稳定
  B) 组合层面（可直接交易）：浮盈 ≥ T → 降到 30%，持有 5 日 → Calmar
     —— 分期间看平台区，过拟合的峰值在分段后会塌

关键：不做单点最优，只认「多个相邻阈值都好」的平台区。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"
START = "20201201"


def load():
    c = sqlite3.connect(DB)
    cap = pd.read_csv("data_storage/mktcap_cache.csv", dtype={"trade_date": str})
    band = list(cap[cap.trade_date == cap.trade_date.max()]
                .query("total_mv>=1e6 and total_mv<=5e6").ts_code)
    px = pd.read_sql_query(
        "SELECT trade_date, ts_code, pct_chg FROM stock_daily WHERE trade_date>=?",
        c, params=(START,))
    pv = px.pivot_table(index="trade_date", columns="ts_code", values="pct_chg") / 100
    pv = pv[[x for x in pv.columns if x in band]]
    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>=? ORDER BY trade_date", c, params=("20201001",)).set_index("trade_date").close
    return c, pv, idx


def cross_section(pv: pd.DataFrame):
    """个股横截面：近20日涨幅分档 → 未来5日跌>5%的概率"""
    nav = (1 + pv.fillna(0)).cumprod()
    r20 = nav / nav.shift(20) - 1
    cs = np.log1p(pv.fillna(0)).cumsum()
    f5 = (np.exp(cs.shift(-5) - cs) - 1)

    # 面板化
    rows = []
    idx = r20.index[20:-5]
    for t in idx:
        g = r20.loc[t]
        f = f5.loc[t]
        ok = g.notna() & f.notna()
        if ok.sum() < 200:
            continue
        rows.append(pd.DataFrame({"g": g[ok].values, "f": f[ok].values, "date": t}))
    panel = pd.concat(rows, ignore_index=True)
    print(f"横截面面板 {len(panel):,} 个观测（{panel.date.min()}~{panel.date.max()}）\n")

    print("=== A) 个股：近20日涨幅 → 未来5日跌>5% 的概率 ===")
    print(f"{'涨幅区间':>12s} {'n':>9s} {'P(跌>5%)':>9s} {'P(跌>10%)':>10s} {'未来5日均':>10s} {'中位':>8s}")
    bins = [(-9, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
            (0.20, 0.25), (0.25, 0.30), (0.30, 0.40), (0.40, 0.60), (0.60, 99)]
    out = []
    for lo, hi in bins:
        x = panel[(panel.g >= lo) & (panel.g < hi)]
        if len(x) < 500:
            continue
        out.append({"区间": f"{lo:.0%}~{hi:.0%}", "n": len(x),
                    "P5": (x.f < -0.05).mean(), "P10": (x.f < -0.10).mean(),
                    "mean": x.f.mean(), "med": x.f.median()})
        print(f"{lo:>5.0%}~{hi:<6.0%} {len(x):9,d} {(x.f<-0.05).mean():8.1%} "
              f"{(x.f<-0.10).mean():9.1%} {x.f.mean():+9.2%} {x.f.median():+7.2%}")
    return pd.DataFrame(out)


def portfolio_scan(pv: pd.DataFrame, idx: pd.Series):
    """组合层面：浮盈 ≥ T 时降到 30%，持有 H 日 → Calmar"""
    nav = (1 + pv.fillna(0)).cumprod()
    r20 = nav / nav.shift(20) - 1
    rank = r20.rank(axis=1, pct=True)
    port = pv.where(rank.shift(1) >= 0.8).mean(axis=1).fillna(0)   # PIT
    pnav = (1 + port).cumprod()
    gain = pnav / pnav.shift(20) - 1
    g60 = (idx / idx.shift(60) - 1).reindex(port.index)

    def backtest(thr, hold_n, target, weak_only=False):
        sig = (gain >= thr).fillna(False)
        if weak_only:
            sig &= (g60 < -0.10).fillna(False)
        pos = np.ones(len(port))
        h = 0
        cur = 1.0
        ret = np.zeros(len(port))
        nev = 0
        for i in range(len(port)):
            if sig.iloc[i]:
                if h == 0:
                    nev += 1
                h = hold_n
            want = target if h > 0 else 1.0
            tc = abs(want - cur) * 0.0015
            cur = want
            if h > 0:
                h -= 1
            ret[i] = cur * port.iloc[i] - tc
        nav2 = pd.Series((1 + ret).cumprod(), index=port.index)
        r = nav2.pct_change().dropna()
        yrs = len(r) / 244
        ann = nav2.iloc[-1] ** (1 / yrs) - 1
        mdd = (nav2 / nav2.cummax() - 1).min()
        return dict(年化=ann, MDD=mdd, Calmar=ann / abs(mdd), 事件=nev, nav=nav2)

    base = backtest(9, 5, 1.0)
    print("\n=== B) 组合层面：浮盈阈值扫描（降到 30%，持有 5 日，含成本）===")
    print(f"{'阈值':>7s} {'Calmar':>8s} {'年化':>8s} {'MDD':>9s} {'事件':>5s} {'vs无规则':>9s}")
    print(f"{'无规则':>7s} {base['Calmar']:8.3f} {base['年化']*100:+7.2f}% {base['MDD']*100:+8.2f}% {0:5d} {'1.00×':>9s}")
    res = {}
    for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40):
        m = backtest(thr, 5, 0.3)
        res[thr] = m
        print(f"{thr:>6.0%} {m['Calmar']:8.3f} {m['年化']*100:+7.2f}% {m['MDD']*100:+8.2f}% "
              f"{m['事件']:5d} {m['Calmar']/base['Calmar']:8.2f}×")
    return res, base, backtest, port


def main():
    c, pv, idx = load()
    cross_section(pv)
    res, base, bt, port = portfolio_scan(pv, idx)

    # 分期间稳定性（只看平台是否保持）
    print("\n=== C) 分期间稳定性（同一阈值在两段是否都好）===")
    print(f"{'阈值':>7s} {'2021-2023':>11s} {'2024-2026':>11s} {'结论':>8s}")
    for thr in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        vals = []
        for a, b in (("20210101", "20231231"), ("20240101", "20261231")):
            sub_p = port.loc[a:b]
            sub_g = (1 + sub_p).cumprod()
            gain = sub_g / sub_g.shift(20) - 1
            sig = (gain >= thr).fillna(False)
            pos = np.ones(len(sub_p))
            h = 0
            cur = 1.0
            ret = np.zeros(len(sub_p))
            for i in range(len(sub_p)):
                if sig.iloc[i]:
                    h = 5
                want = 0.3 if h > 0 else 1.0
                tc = abs(want - cur) * 0.0015
                cur = want
                if h > 0:
                    h -= 1
                ret[i] = cur * sub_p.iloc[i] - tc
            n2 = pd.Series((1 + ret).cumprod(), index=sub_p.index)
            r = n2.pct_change().dropna()
            ann = n2.iloc[-1] ** (244 / len(r)) - 1
            mdd = (n2 / n2.cummax() - 1).min()
            vals.append(ann / abs(mdd) if mdd else np.nan)
        ok = "✅平台" if all(v > 0.25 for v in vals) else "⚠️不稳"
        print(f"{thr:>6.0%} {vals[0]:11.3f} {vals[1]:11.3f} {ok:>8s}")


if __name__ == "__main__":
    sys.exit(main())
