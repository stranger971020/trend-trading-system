"""浮盈阈值扫描 v2：**必须**限定大盘疲弱背景（上一版漏了这个条件）

上一版 bug：backtest(thr, 5, 0.3) 未传 weak_only，导致测的是「任意时候浮盈」，
而用户场景是「大盘疲弱 + 浮盈」—— 条件不同，结论可能完全相反。

本版三组对照：
  ① 不限背景（复现上一版）
  ② 大盘前60日 < -5%（疲弱）
  ③ 大盘前60日 < -10%（深疲弱）
每组内部：不做事 vs 各浮盈阈值减仓，**同背景比较**（公平）。
同时报触发次数：样本少的阈值不可信，必须标出。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"
START = "20201201"
COST = 0.0015
HOLD = 5
TARGET = 0.3


def main():
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
        "AND trade_date>=? ORDER BY trade_date", c, params=("20200701",)).set_index("trade_date").close

    nav = (1 + pv.fillna(0)).cumprod()
    r20 = nav / nav.shift(20) - 1
    rank = r20.rank(axis=1, pct=True)
    port = pv.where(rank.shift(1) >= 0.8).mean(axis=1).fillna(0)      # PIT
    pnav = (1 + port).cumprod()
    gain = pnav / pnav.shift(20) - 1
    g60 = (idx / idx.shift(60) - 1).reindex(port.index)

    def run(sig: pd.Series, target: float):
        pos_prev = 1.0
        h = 0
        ret = np.zeros(len(port))
        nev = 0
        for i in range(len(port)):
            if bool(sig.iloc[i]):
                if h == 0:
                    nev += 1
                h = HOLD
            want = target if h > 0 else 1.0
            tc = abs(want - pos_prev) * COST
            pos_prev = want
            if h > 0:
                h -= 1
            ret[i] = want * port.iloc[i] - tc
        n2 = pd.Series((1 + ret).cumprod(), index=port.index)
        r = n2.pct_change().dropna()
        ann = n2.iloc[-1] ** (244 / len(r)) - 1
        mdd = (n2 / n2.cummax() - 1).min()
        return dict(ann=ann, mdd=mdd, calmar=ann / abs(mdd), nev=nev, sig_days=int(sig.sum()))

    none_sig = pd.Series(False, index=port.index)
    groups = [("① 不限背景", pd.Series(True, index=port.index)),
              ("② 大盘60日<-5%", g60 < -0.05),
              ("③ 大盘60日<-10%", g60 < -0.10)]

    for label, bg in groups:
        bg = bg.fillna(False)
        base = run(none_sig, 1.0)          # 不做事（全程满仓）作为绝对参照
        # 同背景基准：只在背景成立的时段做"是否减仓"的对照 —— 用不触发信号（等于满仓）
        print(f"\n{'='*78}\n{label}（背景覆盖 {int(bg.sum())} 天 / {len(bg)} = {bg.mean()*100:.0f}%）")
        print(f"{'阈值':>6s} {'Calmar':>8s} {'年化':>8s} {'MDD':>9s} {'触发次数':>7s} {'信号天数':>7s} {'vs满仓':>8s} {'样本判断':>8s}")
        print(f"{'满仓':>6s} {base['calmar']:8.3f} {base['ann']*100:+7.2f}% {base['mdd']*100:+8.2f}% "
              f"{0:7d} {0:7d} {1.0:7.2f}×")
        for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30):
            sig = (gain >= thr) & bg
            m = run(sig, TARGET)
            ratio = m["calmar"] / base["calmar"] if base["calmar"] else float("nan")
            flag = "样本少!" if m["nev"] < 8 else ("有效" if ratio > 1.05 else "")
            print(f"{thr:>5.0%} {m['calmar']:8.3f} {m['ann']*100:+7.2f}% {m['mdd']*100:+8.2f}% "
                  f"{m['nev']:7d} {m['sig_days']:7d} {ratio:7.2f}× {flag:>8s}")

    # 附：背景下的「持有 vs 减仓」纯事件对照（不含成本，看原始差异）
    print(f"\n{'='*78}\n④ 事件级对照：背景成立且浮盈达标的那些天，未来 5 日强势组收益")
    cs = np.log1p(port).cumsum()
    f5 = np.exp(cs.shift(-5) - cs) - 1
    for label, bg in groups[1:]:
        bg = bg.fillna(False)
        print(f"\n{label}")
        for thr in (0.05, 0.10, 0.15, 0.20, 0.25):
            sub = f5[(gain >= thr) & bg].dropna()
            if len(sub) < 5:
                print(f"  浮盈≥{thr:.0%}: n={len(sub)} 样本少")
                continue
            print(f"  浮盈≥{thr:>4.0%}: n={len(sub):3d} 未来5日均{sub.mean()*100:+6.2f}% "
                  f"中位{sub.median()*100:+6.2f}% 最差{sub.min()*100:+7.2f}% P(<-5%){(sub<-0.05).mean()*100:3.0f}%")


if __name__ == "__main__":
    sys.exit(main())
