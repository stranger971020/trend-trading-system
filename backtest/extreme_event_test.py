"""极端崩跌事件检验（口径修正：尾部风险，不是均值）

用户纠正：「我不是要一个大部分时候有效的策略，而是要一个在极端市场有效的策略」。
上一版用「未来5日均值」评估 = 被 91% 的反弹样本稀释，方向错了。
本版改为：
  1) 极端事件的精确定义与历史频率（他问：「一周五天 4000+ 下跌是常见情况吗」）
  2) 事件后的**尾部**（最差 10% / 25%、P(未来N日大跌)），而不是均值
  3) 叠加上「事件前有浮盈」这个他的确切前提
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"


def main():
    c = sqlite3.connect(DB)
    s = pd.read_sql_query(
        """SELECT trade_date, COUNT(*) n,
                  SUM(CASE WHEN pct_chg>0 THEN 1 ELSE 0 END) up,
                  SUM(CASE WHEN pct_chg<0 THEN 1 ELSE 0 END) down,
                  SUM(CASE WHEN pct_chg>=9.8 THEN 1 ELSE 0 END) lu,
                  SUM(CASE WHEN pct_chg<=-9.8 THEN 1 ELSE 0 END) ld
           FROM stock_daily WHERE trade_date>='20150817' GROUP BY trade_date
           ORDER BY trade_date""", c).set_index("trade_date")
    s["up_ratio"] = s.up / s.n
    s["down_ratio"] = s.down / s.n

    print("=== 0) 用户问的事实：4000+ 下跌常见吗 ===")
    d4000 = s[s.down >= 4000]
    print(f"2015-08~2026-11 共 {len(s)} 个交易日：下跌家数≥4000 的有 {len(d4000)} 天 "
          f"（{len(d4000)/len(s)*100:.1f}%）")
    # 5 日窗口内 ≥3 天
    cnt5 = (s.down >= 4000).astype(int).rolling(5).sum()
    ev = cnt5 >= 3
    # 只保留事件起点（去连续段）
    starts = ev & ~ev.shift(1).fillna(False)
    print(f"「5 日内 ≥3 天 4000+ 下跌」事件：{int(starts.sum())} 次（去重起点），"
          f"覆盖 {int(ev.sum())} 天（{ev.mean()*100:.1f}% 的时间）")
    print(f"  事件年份分布：{pd.Series(starts[starts].index).astype(str).str[:4].value_counts().sort_index().to_dict()}")
    print("\n最近 5 次事件起点：", list(starts[starts].index[-5:]))
    print("\n2026-11 每日下跌家数：")
    for t in s.loc["20261028":].index:
        r = s.loc[t]
        mark = " ←4000+" if r.down >= 4000 else ""
        print(f"  {t} 下跌{int(r.down):5d} 上涨{int(r.up):5d} up_ratio {r.up_ratio:5.1%}{mark}")

    # ── 1) 事件后尾部（沪深300）──
    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>='20150817' ORDER BY trade_date", c).set_index("trade_date").close
    rets = {}
    for h in (5, 10, 20):
        rets[h] = idx.shift(-h) / idx - 1
    df = pd.DataFrame(rets).reindex(s.index)
    df["ev"] = ev.reindex(df.index).fillna(False)
    df["up_ratio"] = s.up_ratio
    df = df.dropna(subset=[5, 10, 20])

    def tail_report(mask, label):
        sub = df[mask]
        out = {"n": len(sub)}
        for h in (5, 10, 20):
            x = sub[h]
            out[f"h{h}_mean"] = x.mean()
            out[f"h{h}_med"] = x.median()
            out[f"h{h}_p10"] = x.quantile(0.10)
            out[f"h{h}_p25"] = x.quantile(0.25)
            out[f"h{h}_worst"] = x.min()
        return label, out

    print("\n=== 1) 极端事件后的沪深300 收益分布（尾部口径）===")
    hdr = f"{'条件':22s} {'n':>4s} | {'5日 均':>8s}{'中位':>8s}{'最差10%':>9s}{'最差25%':>9s} | {'20日 均':>8s}{'最差10%':>9s}"
    print(hdr)
    rows = [tail_report(pd.Series(True, index=df.index), "全样本基准"),
            tail_report(df.ev, "极端事件(5日≥3次4000+)")]
    for label, o in rows:
        print(f"{label:22s} {o['n']:4d} | {o['h5_mean']*100:+7.2f}%{o['h5_med']*100:+7.2f}%"
              f"{o['h5_p10']*100:+8.2f}%{o['h5_p25']*100:+8.2f}% | "
              f"{o['h20_mean']*100:+7.2f}%{o['h20_p10']*100:+8.2f}%")

    # ── 2) 危险概率：事件后大跌 vs 无条件 ──
    print("\n=== 2) 「继续大跌」的条件概率 ===")
    for h, th in ((5, -0.05), (10, -0.08), (20, -0.10)):
        base = (df[h] < th).mean()
        cond = (df.loc[df.ev, h] < th).mean()
        lift = cond / base if base > 0 else float("nan")
        print(f"  未来{h:2d}日跌超{-th*100:.0f}%：全样本 {base*100:4.1f}%  →  事件后 {cond*100:4.1f}%  "
              f"（提升 {lift:.1f}×，n_event={int(df.ev.sum())}）")

    # ── 3) 事件 × 事前浮盈 ──
    print("\n=== 3) 「事件前 20 日有浮盈」时（用户的确切前提）===")
    g20 = idx / idx.shift(20) - 1
    df["g20"] = g20.reindex(df.index)
    for gth, lab in ((0.0, "事前20日涨幅>0"), (0.05, "事前20日涨幅>5%")):
        m = df.ev & (df.g20 > gth)
        sub = df[m]
        if len(sub) < 5:
            print(f"  {lab}: n={len(sub)} 样本过少")
            continue
        print(f"  {lab}: n={len(sub)} 未来5日 均{sub[5].mean()*100:+.2f}% 中位{sub[5].median()*100:+.2f}% "
              f"最差{sub[5].min()*100:+.1f}% 跌超5%概率{(sub[5]<-0.05).mean()*100:.0f}% · "
              f"未来20日 均{sub[20].mean()*100:+.2f}% 最差{sub[20].min()*100:+.1f}%")

    # ── 4) 连跌背景（用户说"连跌几个月"）──
    print("\n=== 4) 「事件前 60 日已经跌」叠加（连跌背景）===")
    g60 = idx / idx.shift(60) - 1
    df["g60"] = g60.reindex(df.index)
    m = df.ev & (df.g60 < -0.05)
    sub = df[m]
    print(f"  事件 + 前60日跌超5%: n={len(sub)} 未来5日 均{sub[5].mean()*100:+.2f}% 中位{sub[5].median()*100:+.2f}% "
          f"最差{sub[5].min()*100:+.1f}% P(<-5%)={(sub[5]<-0.05).mean()*100:.0f}% · "
          f"未来20日 均{sub[20].mean()*100:+.2f}% 最差{sub[20].min()*100:+.1f}%")
    if len(sub) >= 5:
        print(f"  明细（日期 / 未来5日 / 未来20日）:")
        for t, r in sub.iterrows():
            print(f"    {t}  {r[5]*100:+6.2f}%  {r[20]*100:+7.2f}%")


if __name__ == "__main__":
    sys.exit(main())
