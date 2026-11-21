"""事前预警 v2：真实映射用户场景 = 「大盘疲弱 × 我的持仓强」的背离

v1 的问题：「单日 80% 下跌」本身没有预测力（首次触发 5 日 p10 -2.71%，比基准 -3.94% 还好），
因为 80% 下跌的日子有 311 天，多数是超跌反弹前夜。

用户真正描述的状态是**背离**：
  大盘：过去 60 日疲弱（他在上周五就知道）
  持仓：自己的票在涨（浮盈，他也知道）
  → 这时才该「降低浮盈预期」，并在市场再次出现广泛下跌时快速减仓

本脚本测的正是这个背离状态（全部 PIT），看它是否真能预警补跌。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"
START = "20201201"


def main():
    c = sqlite3.connect(DB)
    s = pd.read_sql_query(
        """SELECT trade_date, COUNT(*) n,
                  SUM(CASE WHEN pct_chg<0 THEN 1 ELSE 0 END) down,
                  AVG(pct_chg)/100 eq
           FROM stock_daily WHERE trade_date>=? GROUP BY trade_date ORDER BY trade_date""",
        c, params=("20150701",)).set_index("trade_date")
    s["dr"] = s.down / s.n

    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>='20150401' ORDER BY trade_date", c).set_index("trade_date").close
    g60 = (idx / idx.shift(60) - 1).reindex(s.index)
    g60_120 = (idx / idx.shift(120) - 1).reindex(s.index)

    # 强势组合（100-500亿域·近20日涨幅前20%等权）的滚动净值与浮盈
    cap = pd.read_csv("data_storage/mktcap_cache.csv", dtype={"trade_date": str})
    band = list(cap[cap.trade_date == cap.trade_date.max()]
                .query("total_mv>=1e6 and total_mv<=5e6").ts_code)
    px = pd.read_sql_query(
        "SELECT trade_date, ts_code, pct_chg FROM stock_daily WHERE trade_date>=? ",
        c, params=(START,))
    pv = px.pivot_table(index="trade_date", columns="ts_code", values="pct_chg") / 100
    pv = pv[[x for x in pv.columns if x in band]]
    nav = (1 + pv.fillna(0)).cumprod()
    r20 = nav / nav.shift(20) - 1
    rank = r20.rank(axis=1, pct=True)

    port = pv.where(rank.shift(1) >= 0.8).mean(axis=1).fillna(0)   # ← 用 t-1 日排名决定 t 日持仓（PIT）
    pnav = (1 + port).cumprod()
    gain20 = pnav / pnav.shift(20) - 1                       # 强势组 20 日浮盈（≈ 用户体感）
    cs = np.log1p(port).cumsum()
    fwd5 = np.exp(cs.shift(-5) - cs) - 1
    fwd10 = np.exp(cs.shift(-10) - cs) - 1
    fwd20 = np.exp(cs.shift(-20) - cs) - 1

    df = pd.DataFrame({"gain20": gain20, "g60": g60.reindex(pnav.index),
                       "dr": s.dr.reindex(pnav.index),
                       "fwd5": fwd5, "fwd10": fwd10, "fwd20": fwd20}).dropna(subset=["fwd5"])
    print(f"样本 {len(df)} 日（{df.index[0]}~{df.index[-1]}，100-500亿域强势组口径）\n")

    def rep(mask, label):
        x = df[mask]
        if len(x) < 5:
            print(f"{label:36s} n={len(x):4d}  样本过少")
            return
        print(f"{label:36s} n={len(x):4d} | 5日均{x.fwd5.mean()*100:+6.2f}% p10{x.fwd5.quantile(.1)*100:+6.2f}% "
              f"最差{x.fwd5.min()*100:+7.2f}% P<-5%{(x.fwd5<-0.05).mean()*100:3.0f}%"
              f" | 20日均{x.fwd20.mean()*100:+6.2f}% p10{x.fwd20.quantile(.1)*100:+6.2f}% "
              f"最差{x.fwd20.min()*100:+7.2f}% P<-10%{(x.fwd20<-0.10).mean()*100:3.0f}%")

    print("=== 1) 基准 ===")
    rep(pd.Series(True, index=df.index), "全样本")
    rep(df.gain20 > 0.10, "仅浮盈>10%")
    rep(df.g60 < -0.05, "仅大盘前60日跌>5%")
    rep(df.g60 < -0.10, "仅大盘前60日跌>10%")

    print("\n=== 2) 背离状态：浮盈 + 大盘疲弱（用户场景）===")
    rep((df.gain20 > 0.10) & (df.g60 < -0.05), "浮盈>10% × 大盘跌>5%")
    rep((df.gain20 > 0.10) & (df.g60 < -0.10), "浮盈>10% × 大盘跌>10%")
    rep((df.gain20 > 0.15) & (df.g60 < -0.05), "浮盈>15% × 大盘跌>5%")
    rep((df.gain20 > 0.15) & (df.g60 < -0.10), "浮盈>15% × 大盘跌>10%")
    rep((df.gain20 > 0.20) & (df.g60 < -0.10), "浮盈>20% × 大盘跌>10%")

    print("\n=== 3) 背离 + 当日再遭广泛下跌（他说的「再一次 4000+」）===")
    for dr_th in (0.80, 0.85, 0.90):
        rep((df.gain20 > 0.10) & (df.g60 < -0.05) & (df.dr >= dr_th),
            f"背离 + 当日下跌占比≥{dr_th:.0%}")

    print("\n=== 4) 2026-11 逐日状态（PIT）===")
    print(f"{'日期':>10s} {'强势组浮盈':>10s} {'大盘前60日':>10s} {'下跌占比':>8s} {'未来5日':>9s} {'未来20日':>9s}")
    for t in df.loc["20261026":].index:
        r = df.loc[t]
        f20 = f"{r.fwd20*100:+.2f}%" if pd.notna(r.fwd20) else "  n/a"
        print(f"{t:>10s} {r.gain20*100:+9.1f}% {r.g60*100:+9.1f}% {r.dr:8.1%} {r.fwd5*100:+8.2f}% {f20:>9s}")

    print("\n=== 5) 历史上「浮盈>10% × 大盘跌>5%」出现过的时段（按月聚合）===")
    m = (df.gain20 > 0.10) & (df.g60 < -0.05)
    grp = df[m].groupby(df[m].index.astype(str).str[:6]).agg(
        天数=("fwd5", "size"), 未来5日均=("fwd5", "mean"), 未来20日均=("fwd20", "mean"),
        最差5日=("fwd5", "min"))
    for k, r in grp.iterrows():
        print(f"  {k}  {int(r['天数']):2d}天  5日{r['未来5日均']*100:+6.2f}%  20日{r['未来20日均']*100:+7.2f}%  最差{r['最差5日']*100:+7.2f}%")


if __name__ == "__main__":
    sys.exit(main())
