"""极端崩跌保护规则回测（口径：尾部/Calmar，不是均值）

用户口径纠正：「我要的是在极端市场有效的策略，不是大部分时候有效」。

事件定义（PIT，收盘可得）：
  EXT = 5 日内 ≥3 天「下跌家数占比 ≥80%」
  连跌背景 = 沪深300 前 60 日跌幅 > 10%

四臂：
  A 满仓（基准）
  R1 EXT 且 未连跌 → 降到 30%，持有 5 个交易日
  R2 EXT 且 未连跌 → 清仓（0%），持有 5 个交易日
  R3 EXT（不看连跌）→ 降到 30%
  R4 EXT 且 已连跌 → 降到 30%（对照：预期无效/有害）

口径限制（必须说明）：日频等权收益（AVG(pct_chg)）有再平衡虚高问题，
故本表用于**规则相对效果**比较，绝对年化不可当可实现收益。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"


def metrics(nav: pd.Series, label: str) -> dict:
    r = nav.pct_change().dropna()
    yrs = len(r) / 244
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    mdd = (nav / nav.cummax() - 1).min()
    vol = r.std() * np.sqrt(244)
    calmar = ann / abs(mdd) if mdd < 0 else float("nan")
    return {"组": label, "年化": ann, "波动": vol, "最大回撤": mdd,
            "Calmar": calmar, "Calmar/满仓": np.nan, "累计": nav.iloc[-1] - 1}


def main():
    c = sqlite3.connect(DB)
    eq = pd.read_sql_query(
        "SELECT trade_date, AVG(pct_chg)/100 r FROM stock_daily WHERE trade_date>='20150817' "
        "GROUP BY trade_date ORDER BY trade_date", c).set_index("trade_date").r
    s = pd.read_sql_query(
        """SELECT trade_date, COUNT(*) n, SUM(CASE WHEN pct_chg<0 THEN 1 ELSE 0 END) down
           FROM stock_daily WHERE trade_date>='20150817' GROUP BY trade_date ORDER BY trade_date""",
        c).set_index("trade_date")
    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>='20150401' ORDER BY trade_date", c).set_index("trade_date").close

    dr = (s.down / s.n).reindex(eq.index)
    g60 = (idx / idx.shift(60) - 1).reindex(eq.index)

    ext = (dr >= 0.80).astype(int).rolling(5).sum().fillna(0) >= 3
    down_bg = g60 < -0.10
    trig = {
        "R1 EXT&未连跌→30%": ext & ~down_bg,
        "R2 EXT&未连跌→0%": ext & ~down_bg,
        "R3 EXT→30%": ext,
        "R4 EXT&已连跌→30%": ext & down_bg,
    }

    rows = []
    navs = {}
    for label, mask in trig.items():
        target = 0.0 if "→0%" in label else 0.3
        pos = np.ones(len(eq))
        hold = 0
        m = mask.fillna(False).values
        for i in range(len(eq)):
            if m[i]:
                hold = 5
            if hold > 0:
                pos[i] = target
                hold -= 1
        nav = (1 + pos * eq.values).cumprod()
        navs[label] = pd.Series(nav, index=eq.index)
        rows.append(metrics(navs[label], label))

    base_nav = (1 + eq).cumprod()
    rows.insert(0, metrics(base_nav, "A 满仓（基准）"))
    for r in rows[1:]:
        r["Calmar/满仓"] = r["Calmar"] / rows[0]["Calmar"] if rows[0]["Calmar"] else np.nan

    print("=== 规则对比（2015-08~2026-11，日频等权口径）===")
    print(f"{'组':22s} {'年化':>8s} {'波动':>7s} {'最大回撤':>9s} {'Calmar':>8s} {'/满仓':>6s} {'触发天数':>7s}")
    for r in rows:
        nm = r["组"]
        nd = int(trig[nm].fillna(False).sum()) if nm in trig else 0
        print(f"{nm:22s} {r['年化']*100:+7.2f}% {r['波动']*100:6.1f}% {r['最大回撤']*100:+8.2f}% "
              f"{r['Calmar']:8.3f} {r['Calmar/满仓']:5.2f}× {nd:7d}")

    # ── 极端样本内表现：三次灾难窗口 ──
    print("\n=== 关键灾难窗口的表现（规则是否真的救了）===")
    wins = [("20150818", "20150831", "2015-08 股灾2.0"),
            ("20160104", "20160115", "2016-01 熔断"),
            ("20240122", "20240205", "2024-01 微盘崩"),
            ("20261030", "20261119", "2026-11 本次")]
    for a, b, nm in wins:
        seg = eq.loc[a:b]
        if len(seg) == 0:
            continue
        hold = (1 + seg).prod() - 1
        line = f"  {nm:16s} 满仓 {hold*100:+7.2f}%"
        for label in trig:
            nav = navs[label]
            v = nav.loc[a:b]
            if len(v) < 2:
                continue
            line += f" | {label.split('→')[0].replace('EXT',''):12s} {(v.iloc[-1]/v.iloc[0]-1)*100:+7.2f}%"
        print(line)

    # ── R2 清仓的事件级清单 ──
    print("\n=== R2（未连跌+极端 → 清仓5日）逐次事件效果 ===")
    pos = np.ones(len(eq))
    hold = 0
    m = trig["R2 EXT&未连跌→0%"].fillna(False).values
    starts = []
    for i in range(len(eq)):
        if m[i] and hold == 0:
            starts.append(eq.index[i])
        if m[i]:
            hold = 5
        if hold > 0:
            pos[i] = 0.0
            hold -= 1
    tot = 0.0
    for t in starts:
        i = eq.index.get_loc(t)
        seg = eq.iloc[i:i + 5]
        r_hold = (1 + seg).prod() - 1
        gain = -r_hold            # 清仓避免（正=避损）
        tot += gain
        print(f"  {t}  满仓5日 {r_hold*100:+7.2f}%  → 清仓避损 {gain*100:+7.2f}%")
    print(f"  合计：{len(starts)} 次触发，全天候清仓累计避损 {tot*100:+.2f}%（未计现金收益与交易成本）")


if __name__ == "__main__":
    sys.exit(main())
