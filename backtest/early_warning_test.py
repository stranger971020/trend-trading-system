"""事前（PIT）预警检验：大盘疲弱背景 + 首次 4000+ 下跌就减仓

用户批评（正确）：R2 用「5 日内 ≥3 天」= 等第 3 天才触发，那时灾难已发生 2-3 天，是事后确认。
用户要的是：上周五（11-13）用当时已知的信息（过去几个月大盘走势 + 每日下跌比例）
就判断「该降浮盈预期」，并在**第一次** 4000+ 出现时就知道大概率继续回撤。

本脚本严格 PIT：
  · 背景 = 前 60 日沪深300 收益（t 日收盘可得）
  · 触发 = 当日下跌占比 ≥80%（t 日收盘可得，次日开盘执行）
  · 「首次」= 前 20 个交易日无同类触发
统计的是 tail（最差 10%/25%、P(继续大跌)），不是均值。
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
                  AVG(pct_chg)/100 eq
           FROM stock_daily WHERE trade_date>='20150817' GROUP BY trade_date ORDER BY trade_date""",
        c).set_index("trade_date")
    s["dr"] = s.down / s.n
    s["ur"] = s.up / s.n

    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>='20150101' ORDER BY trade_date", c).set_index("trade_date").close
    g = pd.DataFrame({
        "g20": idx / idx.shift(20) - 1,
        "g60": idx / idx.shift(60) - 1,
        "g120": idx / idx.shift(120) - 1,
    }).reindex(s.index)

    # 全A等权未来收益（用户域近似，无幸存者偏差）：t+1 .. t+h 的累计
    eqr = s["eq"]
    cs = np.log1p(eqr).cumsum()
    fwd = pd.DataFrame(index=s.index)
    for h in (3, 5, 10, 20):
        fwd[h] = np.exp(cs.shift(-h) - cs) - 1

    df = pd.concat([s[["dr", "ur"]], g, fwd], axis=1).dropna(subset=[5, 20])
    E = df.dr >= 0.80
    # 首次：前 20 日无 E
    prev20 = E.shift(1).rolling(20, min_periods=1).sum()
    first = E & (prev20 == 0)
    later = E & (prev20 > 0)

    def tail(mask, label, h=(5, 20)):
        x = df[mask]
        if len(x) < 5:
            print(f"{label:34s} n={len(x):3d}  样本过少")
            return
        out = f"{label:34s} n={len(x):4d}"
        for hh in h:
            col = x[hh]
            out += (f" | {hh:2d}日 均{col.mean()*100:+6.2f}% p10{col.quantile(.1)*100:+6.2f}% "
                    f"最差{col.min()*100:+7.2f}% P<-5%{(col<-0.05).mean()*100:4.0f}%")
        print(out)

    print("=== 1) 全样本基准 ===")
    tail(pd.Series(True, index=df.index), "全样本")

    print("\n=== 2) 单日 80% 下跌：首次 vs 后续（不区分背景）===")
    tail(first, "首次（前20日无同类）")
    tail(later, "后续（20日内已触发过）")

    print("\n=== 3) 叠加「大盘疲弱」背景（前60日沪深300 涨跌）===")
    for lo, hi, nm in [(-9, -0.10, "深疲弱(< -10%)"), (-0.10, -0.05, "疲弱(-10~-5%)"),
                       (-0.05, 0.03, "横盘(-5~+3%)"), (0.03, 9, "偏强(>+3%)")]:
        m = first & (df.g60 >= lo) & (df.g60 < hi)
        tail(m, f"首次 + 背景{nm}")

    print("\n=== 4) 关键组合：疲弱背景 + 首次触发（这应是他的场景）===")
    weak = df.g60 < -0.05
    tail(first & weak, "首次 + 前60日跌>5%")
    tail(later & weak, "后续 + 前60日跌>5%")
    tail(E & weak, "任意触发 + 前60日跌>5%")

    # ── 5) 2026-11 逐日回看（严格 PIT）──
    print("\n=== 5) 2026-11 逐日（PIT：当日收盘已知的信息）===")
    print(f"{'日期':>10s} {'下跌家数':>7s} {'占比':>6s} {'前60日':>8s} {'前20日':>7s} {'首次?':>5s} {'未来5日':>9s}")
    for t in df.loc["20261020":].index:
        r = df.loc[t]
        if pd.isna(r.g60):
            continue
        f5 = f"{r[5]*100:+.2f}%" if pd.notna(r[5]) else "  n/a"
        print(f"{t:>10s} {int(s.loc[t,'down']):7d} {r.dr:6.1%} {r.g60*100:+8.1f}% {r.g20*100:+7.1f}% "
              f"{'★首发' if first.loc[t] else ('  后续' if later.loc[t] else '   — '):>5s} {f5:>9s}")

    print("\n=== 6) 事件清单：疲弱背景下的首次触发（全部历史）===")
    sel = df[first & weak]
    for t, r in sel.iterrows():
        print(f"  {t} 前60日{r.g60*100:+6.1f}% 当日{r.dr:5.1%} → 5日{r[5]*100:+6.2f}% "
              f"20日{r[20]*100:+7.2f}%")


if __name__ == "__main__":
    sys.exit(main())
