"""情绪阈值标定 + 强势股补跌证据（为 A 警报与 B 规则定标，2026-11-21）

问题：
  1) 「大盘很惨」如何定量？（up_ratio / 涨停数 / 跌停数 的哪个阈值对应未来 5 日大概率下跌）
  2) 用户体感「浮盈 + 情绪退潮 → 高概率补跌」是否成立？（强势股在退潮日的未来 5 日收益）

口径：全部 PIT（t 日收盘可得的信息 → t+1..t+5 收益）。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data_storage/sw_index_data.db"


def daily_sentiment(c):
    return pd.read_sql_query(
        """SELECT trade_date, COUNT(*) n,
                  SUM(CASE WHEN pct_chg>0 THEN 1 ELSE 0 END) up,
                  SUM(CASE WHEN pct_chg>=9.8 THEN 1 ELSE 0 END) lu,
                  SUM(CASE WHEN pct_chg<=-9.8 THEN 1 ELSE 0 END) ld,
                  AVG(pct_chg) avg_pct
           FROM stock_daily WHERE trade_date>='20150817' GROUP BY trade_date
           ORDER BY trade_date""", c)


def main():
    c = sqlite3.connect(DB)
    s = daily_sentiment(c)
    s["up_ratio"] = s.up / s.n
    s["ld_ratio"] = s.ld / s.n
    s = s.set_index("trade_date")

    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date>='20150817' ORDER BY trade_date", c).set_index("trade_date").close
    fwd5 = idx.shift(-5) / idx - 1

    df = pd.DataFrame({"up_ratio": s.up_ratio, "lu": s.lu, "ld": s.ld,
                       "ld_ratio": s.ld_ratio, "avg_pct": s.avg_pct, "fwd5": fwd5}).dropna()
    print(f"样本 {len(df)} 个交易日 {df.index[0]}~{df.index[-1]}\n")

    # ── 1) up_ratio 十分位 → 未来5日 ──
    df["q"] = pd.qcut(df.up_ratio, 10, labels=[f"D{i}" for i in range(1, 11)])
    g = df.groupby("q", observed=True).agg(
        n=("fwd5", "size"), up_min=("up_ratio", "min"), up_max=("up_ratio", "max"),
        fwd5_mean=("fwd5", "mean"), fwd5_med=("fwd5", "median"),
        p_down=("fwd5", lambda x: (x < 0).mean()),
        p_bad=("fwd5", lambda x: (x < -0.03).mean()))
    print("=== 涨占比(up_ratio)十分位 → 沪深300 未来5日 ===")
    for q, r in g.iterrows():
        print(f"{q:>3s} 区间{r.up_min:5.2f}~{r.up_max:4.2f} n={int(r.n):4d} "
              f"均值{r.fwd5_mean*100:+5.2f}% 中位{r.fwd5_med*100:+5.2f}% "
              f"下跌概率{r.p_down*100:3.0f}% 跌超3%{r.p_bad*100:3.0f}%")

    # ── 2) 联合条件：up_ratio 低 + 涨停数相对 20 日均值骤降 ──
    df["lu_ma20"] = df.lu.rolling(20, min_periods=10).mean()
    df["lu_drop"] = df.lu / df.lu_ma20
    print("\n=== 联合条件（组合检验）===")
    conds = {
        "up_ratio<0.25": df.up_ratio < 0.25,
        "up_ratio<0.20": df.up_ratio < 0.20,
        "up_ratio<0.25 且 涨停<50%20日均": (df.up_ratio < 0.25) & (df.lu_drop < 0.5),
        "up_ratio<0.30 且 跌停>50家": (df.up_ratio < 0.30) & (df.ld > 50),
        "up_ratio<0.25 且 跌停>80家": (df.up_ratio < 0.25) & (df.ld > 80),
    }
    base = df.fwd5
    print(f"基准（全样本）     n={len(base):4d} 均值{base.mean()*100:+5.2f}% 下跌概率{(base<0).mean()*100:3.0f}% 跌超3%{(base<-0.03).mean()*100:3.0f}%")
    for name, m in conds.items():
        sub = df.loc[m, "fwd5"]
        if len(sub) < 10:
            print(f"{name:28s} n={len(sub)} 样本过少")
            continue
        print(f"{name:28s} n={len(sub):4d} 均值{sub.mean()*100:+5.2f}% 下跌概率{(sub<0).mean()*100:3.0f}% "
              f"跌超3%{(sub<-0.03).mean()*100:3.0f}%")

    # ── 3) 强势股在退潮日的补跌（核心：用户体感）──
    print("\n=== 强势股补跌检验（100-500亿域，近20日涨幅前20%）===")
    cap = pd.read_csv("data_storage/mktcap_cache.csv", dtype={"trade_date": str})
    last = cap.trade_date.max()
    band = set(cap[cap.trade_date == last].query("total_mv>=1e6 and total_mv<=5e6").ts_code)

    px = pd.read_sql_query(
        "SELECT trade_date, ts_code, close FROM stock_daily WHERE trade_date>='20260101' "
        "AND close>0", c)
    print(f"  价格样本 {len(px):,} 行（2026 年起，控制内存）；域内票 {len(band)} 只")
    pv = px.pivot_table(index="trade_date", columns="ts_code", values="close")
    pv = pv[[c for c in pv.columns if c in band]]
    r20 = pv / pv.shift(20) - 1
    f5 = pv.shift(-5) / pv - 1
    rank = r20.rank(axis=1, pct=True)          # 横截面分位（1=最强）

    def bucket_stats(mask_strong, label):
        rows = []
        for t in f5.index:
            if t not in df.index:
                continue
            r = df.loc[t]
            if not np.isfinite(r.fwd5):
                continue
            sel = pd.Series(mask_strong.loc[t]).values & f5.loc[t].notna().values
            if sel.sum() < 30:
                continue
            rows.append({"t": t, "up_ratio": r.up_ratio, "ret5": f5.loc[t][sel].mean()})
        d = pd.DataFrame(rows).set_index("t")
        print(f"\n  [{label}] 样本日 {len(d)}")
        for lo, hi, nm in [(0, 0.20, "退潮 up<0.20"), (0.20, 0.30, "弱 up0.2~0.3"),
                           (0.30, 0.60, "中性"), (0.60, 1.01, "强 up>0.6")]:
            sub = d[(d.up_ratio >= lo) & (d.up_ratio < hi)]
            if len(sub) < 5:
                continue
            print(f"    {nm:16s} n={len(sub):3d} 未来5日均收益 {sub.ret5.mean()*100:+5.2f}% "
                  f"中位 {sub.ret5.median()*100:+5.2f}% 亏损概率 {(sub.ret5<0).mean()*100:3.0f}%")
        return d

    strong = rank >= 0.8
    d_strong = bucket_stats(strong, "强势组(近20日涨幅前20%)")
    weak = rank <= 0.2
    bucket_stats(weak, "弱势组(后20%)")

    # 退潮日里，强势组 vs 弱势组 的未来5日差
    if len(d_strong):
        print("\n  提示：强势组在退潮日的未来5日均收益若仍明显低于弱势组 → 补跌成立")


if __name__ == "__main__":
    sys.exit(main())
