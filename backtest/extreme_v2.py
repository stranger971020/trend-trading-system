"""极端崩跌事件 v2 —— 修正样本偏差：用「下跌占比」而非绝对家数

v1 的问题：用「4000+ 家下跌」作阈值，2015-2020 全市场只有 3000-4000 只股票，
客观上不可能触发 → 样本全部落在 2024-2026，是「股票数量变多」造成的假象，不是市场变了。
本版改用 down_ratio（下跌占比）>= 0.80，历史可比，能覆盖 2015 股灾/2016 熔断/2018 熊市。
"""
import sqlite3
import sys

import pandas as pd

DB = "data_storage/sw_index_data.db"


def main():
    c = sqlite3.connect(DB)
    s = pd.read_sql_query(
        """SELECT trade_date, COUNT(*) n,
                  SUM(CASE WHEN pct_chg>0 THEN 1 ELSE 0 END) up,
                  SUM(CASE WHEN pct_chg<0 THEN 1 ELSE 0 END) down
           FROM stock_daily WHERE trade_date>='20150817' GROUP BY trade_date
           ORDER BY trade_date""", c).set_index("trade_date")
    s["down_ratio"] = s.down / s.n
    s["up_ratio"] = s.up / s.n

    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>='20150817' ORDER BY trade_date", c).set_index("trade_date").close
    R = pd.DataFrame({h: idx.shift(-h) / idx - 1 for h in (5, 10, 20)})
    R["g20"] = idx / idx.shift(20) - 1
    R["g60"] = idx / idx.shift(60) - 1

    print("=== 0) 下跌占比 ≥80% 的历史频率（按年）===")
    hit = (s.down_ratio >= 0.80)
    by_year = hit.groupby(hit.index.astype(str).str[:4]).agg(["sum", "size"])
    for y, r in by_year.iterrows():
        if r["sum"] > 0:
            print(f"  {y}: {int(r['sum'])} 天 / {int(r['size'])} 交易日（{r['sum']/r['size']*100:.1f}%）")
    print(f"  合计 {int(hit.sum())} 天 / {len(s)}（{hit.mean()*100:.1f}%）")

    # 事件：5 日内 ≥3 天 down_ratio>=0.80
    cnt5 = hit.astype(int).rolling(5).sum()
    ev = cnt5 >= 3
    starts = ev & ~ev.shift(1).fillna(False)
    print(f"\n事件（5日内≥3天 下跌占比≥80%）：{int(starts.sum())} 次，覆盖 {int(ev.sum())} 天")
    print("事件起点（全部）：", list(starts[starts].index))

    df = R.copy()
    df["ev"] = ev.reindex(df.index).fillna(False)
    df["dr"] = s.down_ratio
    df = df.dropna(subset=[5, 10, 20])

    def rep(mask, label):
        sub = df[mask]
        if len(sub) == 0:
            print(f"{label:26s} n=0")
            return
        print(f"{label:26s} n={len(sub):3d} | 5日 均{sub[5].mean()*100:+6.2f}% 中位{sub[5].median()*100:+6.2f}% "
              f"p10 {sub[5].quantile(.1)*100:+6.2f}% 最差{sub[5].min()*100:+7.2f}% "
              f"P(<-5%){(sub[5]<-0.05).mean()*100:4.0f}% | 20日 均{sub[20].mean()*100:+6.2f}% "
              f"p10 {sub[20].quantile(.1)*100:+6.2f}% 最差{sub[20].min()*100:+7.2f}% "
              f"P(<-10%){(sub[20]<-0.10).mean()*100:4.0f}%")

    print("\n=== 1) 全样本 vs 事件后（尾部口径）===")
    rep(pd.Series(True, index=df.index), "全样本基准")
    rep(df.ev, "事件后（全期）")
    rep(df.ev & (df.index < "20200101"), "事件后 · 2015-2019")
    rep(df.ev & (df.index >= "20200101"), "事件后 · 2020-2026")

    print("\n=== 2) 事件 + 连跌背景（前60日跌超10%）===")
    rep(df.ev & (df.g60 < -0.10), "事件+前60日跌>10%")
    rep(df.ev & (df.g60 < -0.10) & (df.index < "20200101"), "  └ 2015-2019")
    rep(df.ev & (df.g60 < -0.10) & (df.index >= "20200101"), "  └ 2020-2026")

    print("\n=== 3) 事件 + 事前20日浮盈（指数涨>3%）===")
    rep(df.ev & (df.g20 > 0.03), "事件+前20日涨>3%")
    rep(df.ev & (df.g20 > 0.03) & (df.index < "20200101"), "  └ 2015-2019")
    rep(df.ev & (df.g20 > 0.03) & (df.index >= "20200101"), "  └ 2020-2026")

    print("\n=== 4) 事件起点逐条明细 ===")
    for t in starts[starts].index:
        if t not in df.index:
            continue
        r = df.loc[t]
        tag = "连跌" if (r.g60 is not None and pd.notna(r.g60) and r.g60 < -0.10) else ""
        print(f"  {t} 前20日{r.g20*100:+6.1f}% 前60日{r.g60*100:+7.1f}% {tag:4s} → "
              f"5日{r[5]*100:+6.2f}% 10日{r[10]*100:+6.2f}% 20日{r[20]*100:+7.2f}%")


if __name__ == "__main__":
    sys.exit(main())
