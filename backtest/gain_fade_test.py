"""B 规则定标：组合浮盈后是否回吐？（2026-11-21）

用户假设：「大盘很惨 + 浮盈 + 情绪退潮 → 高概率补跌，应该兑现」
上一步已否证「退潮 → 大盘未来5日下跌」（反而上涨）。本步直接测用户的真实场景：

  构造一个会浮盈的组合（每日按近20日涨幅前20%等权持有，滚动的强势股组合），
  看它在「浮盈达到 X%」之后的未来 5 日收益分布，并叠加情绪状态。

若浮盈 15%+ 后未来 5 日仍为正 → 兑现规则无效（与止损同类结局）。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"
START = "20201201"


def main():
    c = sqlite3.connect(DB)
    sen = pd.read_sql_query(
        """SELECT trade_date, COUNT(*) n,
                  SUM(CASE WHEN pct_chg>0 THEN 1 ELSE 0 END) up,
                  SUM(CASE WHEN pct_chg>=9.8 THEN 1 ELSE 0 END) lu
           FROM stock_daily WHERE trade_date>=? GROUP BY trade_date ORDER BY trade_date""",
        c, params=(START,)).set_index("trade_date")
    sen["up_ratio"] = sen.up / sen.n

    cap = pd.read_csv("data_storage/mktcap_cache.csv", dtype={"trade_date": str})
    band = set(cap[cap.trade_date == cap.trade_date.max()]
               .query("total_mv>=1e6 and total_mv<=5e6").ts_code)

    px = pd.read_sql_query(
        "SELECT trade_date, ts_code, close FROM stock_daily WHERE trade_date>=? AND close>0",
        c, params=(START,))
    pv = px.pivot_table(index="trade_date", columns="ts_code", values="close")
    pv = pv[[x for x in pv.columns if x in band]]
    print(f"价格面板 {pv.shape[0]} 日 × {pv.shape[1]} 只（100-500亿域）")

    r1 = pv.pct_change()
    r20 = pv / pv.shift(20) - 1
    rank = r20.rank(axis=1, pct=True)

    # ── 滚动强势组合的日收益序列（每日取前20%等权，持有1日）──
    strong = (rank >= 0.8)
    port = (r1.where(strong)).mean(axis=1)          # 每日等权
    port = port.fillna(0.0)
    nav = (1 + port).cumprod()

    # 浮盈：过去 20 日该组合的累计涨幅（≈ 持有一个月后的账面浮盈）
    gain20 = nav / nav.shift(20) - 1
    # 浮盈 5 日 / 10 日（更快达标的持仓）
    fwd5 = nav.shift(-5) / nav - 1

    df = pd.DataFrame({"gain20": gain20, "fwd5": fwd5,
                       "up_ratio": sen.up_ratio.reindex(nav.index)}).dropna()
    print(f"样本 {len(df)} 个交易日\n")

    # ── 1) 浮盈分档 → 未来5日 ──
    bins = [-9, 0, 0.05, 0.10, 0.15, 0.25, 9]
    labels = ["浮亏", "0~5%", "5~10%", "10~15%", "15~25%", ">25%"]
    df["b"] = pd.cut(df.gain20, bins, labels=labels)
    print("=== 强势组合 20日浮盈 → 未来5日收益（全体本）===")
    print(f"{'浮盈档':>8s} {'n':>5s} {'未来5日均':>9s} {'中位':>8s} {'亏损概率':>8s} {'跌超3%':>8s}")
    for b, r in df.groupby("b", observed=True):
        print(f"{str(b):>8s} {len(r):5d} {r.fwd5.mean()*100:+8.2f}% {r.fwd5.median()*100:+7.2f}% "
              f"{(r.fwd5<0).mean()*100:7.0f}% {(r.fwd5<-0.03).mean()*100:7.0f}%")

    # ── 2) 浮盈×情绪 交叉 ──
    print("\n=== 浮盈 × 情绪（up_ratio）→ 未来5日均收益 / 亏损概率 / n ===")
    df["emo"] = pd.cut(df.up_ratio, [0, 0.20, 0.35, 0.60, 1.01],
                       labels=["退潮<0.2", "弱0.2~0.35", "中0.35~0.6", "强>0.6"])
    piv_mean = df.pivot_table(index="b", columns="emo", values="fwd5", aggfunc="mean",
                              observed=False) * 100
    piv_p = df.pivot_table(index="b", columns="emo", values="fwd5",
                           aggfunc=lambda x: (x < 0).mean(), observed=False) * 100
    piv_n = df.pivot_table(index="b", columns="emo", values="fwd5", aggfunc="size",
                           observed=False)
    for b in piv_mean.index:
        line = f"{str(b):>8s} |"
        for e in piv_mean.columns:
            m, p, n = piv_mean.loc[b, e], piv_p.loc[b, e], piv_n.loc[b, e]
            line += f" {str(e):>11s}: {m:+6.2f}%/{p:3.0f}%/{int(n):3d}" if n == n else \
                    f" {str(e):>11s}:     —        "
        print(line)

    # ── 3) 若在「浮盈≥15% 且 退潮」时清仓，能躲开多少？（条件收益）──
    print("\n=== 关键组合检验 ===")
    tests = {
        "浮盈≥15%（无条件）": df.gain20 >= 0.15,
        "浮盈≥15% 且 up_ratio<0.20": (df.gain20 >= 0.15) & (df.up_ratio < 0.20),
        "浮盈≥15% 且 up_ratio<0.35": (df.gain20 >= 0.15) & (df.up_ratio < 0.35),
        "浮盈≥10% 且 up_ratio<0.20": (df.gain20 >= 0.10) & (df.up_ratio < 0.20),
        "仅 up_ratio<0.20": df.up_ratio < 0.20,
    }
    allm = df.fwd5
    print(f"{'条件':32s} {'n':>4s} {'未来5日均':>9s} {'中位':>8s} {'亏损概率':>8s}")
    print(f"{'全样本基准':32s} {len(allm):4d} {allm.mean()*100:+8.2f}% "
          f"{allm.median()*100:+7.2f}% {(allm<0).mean()*100:7.0f}%")
    for name, m in tests.items():
        sub = df.loc[m, "fwd5"]
        if len(sub) < 5:
            print(f"{name:32s} {len(sub):4d} 样本过少")
            continue
        print(f"{name:32s} {len(sub):4d} {sub.mean()*100:+8.2f}% {sub.median()*100:+7.2f}% "
              f"{(sub<0).mean()*100:7.0f}%")

    # ── 4) 用 2026-11 的实际数据回看（是否真的补跌）──
    print("\n=== 2026-11 回看（强势组合净值 / 浮盈 / 当日情绪）===")
    m = nav.index >= "20261028"
    show = pd.DataFrame({"组合净值": nav[m].round(4),
                         "20日浮盈%": (gain20[m] * 100).round(1),
                         "up_ratio": sen.up_ratio.reindex(nav.index)[m].round(2)})
    print(show.to_string())


if __name__ == "__main__":
    sys.exit(main())
