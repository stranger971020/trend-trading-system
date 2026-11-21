"""量化活跃度代理 · 事件验证（A 方案的第一道闸门）

2026-11-22 Claude。先验证，不先做看板：
  预注册代理 = 每日横截面 z 分数的等权平均：
      z(turnover_rate) + z(range) + z(|close/open-1|) + z(small_order_share)
  其中 small_order_share = (buy_sm+sell_sm) / 全部买卖金额。
  规模分组：每日 circ_mv 最低 20% = micro；最高 20% = large。
  事件：
      2024-01-22 ~ 2024-02-05 微盘崩
      2024-02-05 ~ 2024-02-08 量化平仓/反弹
  判据（先定后看）：
      micro_comp - large_comp 在事件前 10 日相对前 60 日均值 >= +1σ，
      且事件窗口内峰值 >= 前 60 日 80 分位 → 代理有抬升；否则判无效，停。

数据：Tushare daily_basic + moneyflow（逐日），本地 stock_daily OHLCV。
缓存：backtest/cache/quant_activity_raw.parquet
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "backtest", "cache")
DB = os.path.join(ROOT, "data_storage", "sw_index_data.db")
RAW = os.path.join(CACHE, "quant_activity_raw.parquet")
START, END = "20231101", "20240430"
EVENT1 = ("20240122", "20240205")
EVENT2 = ("20240205", "20240208")
PRE_DAYS = 10
BASE_DAYS = 60


def _trade_dates() -> list[str]:
    c = sqlite3.connect(DB)
    ds = [r[0] for r in c.execute(
        "select distinct trade_date from stock_daily where trade_date between ? and ? "
        "order by trade_date", (START, END))]
    c.close()
    return ds


def _fetch(force: bool = False) -> pd.DataFrame:
    if os.path.exists(RAW) and not force:
        df = pd.read_parquet(RAW)
        if str(df.trade_date.min()) <= START and str(df.trade_date.max()) >= END:
            return df
    import tushare as ts
    from config import TUSHARE_TOKEN
    pro = ts.pro_api(TUSHARE_TOKEN)
    dates = _trade_dates()
    rows = []
    for i, d in enumerate(dates, 1):
        try:
            bs = pro.daily_basic(trade_date=d, fields="ts_code,trade_date,turnover_rate,circ_mv,float_share")
            mf = pro.moneyflow(trade_date=d, fields=(
                "ts_code,trade_date,buy_elg_amount,buy_lg_amount,buy_md_amount,buy_sm_amount,"
                "sell_elg_amount,sell_lg_amount,sell_md_amount,sell_sm_amount"))
            if bs is not None and not bs.empty and mf is not None and not mf.empty:
                rows.append(bs.merge(mf, on=["ts_code", "trade_date"], how="inner"))
            print(f"[fetch] {i}/{len(dates)} {d} rows={0 if not rows else len(rows[-1])}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] {d} FAIL {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(0.12)
    if not rows:
        raise RuntimeError("未取到任何 daily_basic/moneyflow 数据")
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(RAW)
    return out


def _z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else s * 0


def build() -> pd.DataFrame:
    c = sqlite3.connect(DB)
    px = pd.read_sql_query(
        "select trade_date,ts_code,open,high,low,close,pre_close,amount,vol "
        "from stock_daily where trade_date between ? and ?", c, params=(START, END))
    c.close()
    raw = _fetch()
    df = px.merge(raw, on=["trade_date", "ts_code"], how="inner")
    df = df[(df.amount > 0) & (df.vol > 0) & df.turnover_rate.notna() & df.circ_mv.notna()]
    df["total_order"] = df[["buy_elg_amount", "buy_lg_amount", "buy_md_amount", "buy_sm_amount",
                            "sell_elg_amount", "sell_lg_amount", "sell_md_amount",
                            "sell_sm_amount"]].sum(axis=1)
    df = df[df.total_order > 0]
    df["small_share"] = (df.buy_sm_amount + df.sell_sm_amount) / df.total_order
    df["range"] = (df.high - df.low) / df.pre_close
    df["abs_intraday"] = (df.close / df.open - 1).abs()
    df["intraday"] = df.close / df.open - 1
    df["overnight"] = df.open / df.pre_close - 1
    # 每日横截面 z
    for col, zc in [("turnover_rate", "z_turn"), ("range", "z_range"),
                    ("abs_intraday", "z_intra"), ("small_share", "z_small")]:
        df[zc] = df.groupby("trade_date")[col].transform(_z)
    df["comp"] = df[["z_turn", "z_range", "z_intra", "z_small"]].mean(axis=1)
    # 每日规模分组
    df["size_rank"] = df.groupby("trade_date")["circ_mv"].rank(pct=True)
    df["group"] = np.where(df.size_rank <= 0.20, "micro",
                           np.where(df.size_rank >= 0.80, "large", "mid"))
    return df


def agg(df: pd.DataFrame, col: str = "comp") -> pd.DataFrame:
    # 成交额加权（amount 单位千元，权重比例即可）
    d = df.copy()
    d["_w"] = d[col] * d.amount
    g = d.groupby(["trade_date", "group"])[["_w", "amount"]].sum()
    out = (g["_w"] / g["amount"]).unstack("group")
    m = d.groupby("trade_date")[["_w", "amount"]].sum()
    out["market"] = m["_w"] / m["amount"]
    out["spread"] = out["micro"] - out["large"]
    out.index = out.index.astype(str)
    return out


def stat(series: pd.Series, event_start: str) -> dict:
    idx = list(series.index)
    if event_start not in idx:
        return {}
    i = idx.index(event_start)
    pre = series.iloc[max(0, i - PRE_DAYS):i]
    base = series.iloc[max(0, i - BASE_DAYS):i]
    return {
        "pre_mean": float(pre.mean()),
        "base_mean": float(base.mean()),
        "base_std": float(base.std()),
        "z": float((pre.mean() - base.mean()) / base.std()) if base.std() else np.nan,
        "pre_max": float(pre.max()),
        "base_p80": float(base.quantile(0.80)),
        "pass": bool(base.std() and (pre.mean() - base.mean()) / base.std() >= 1.0
                     and pre.max() >= base.quantile(0.80)),
    }


def main():
    print("构建代理数据...", flush=True)
    df = build()
    print(f"样本 {len(df):,} 行 · {df.trade_date.nunique()} 个交易日 "
          f"({df.trade_date.min()}~{df.trade_date.max()})")
    a = agg(df, "comp")
    print("\n=== 微盘-大盘 spread（comp）事件前后 ===")
    for ev in (EVENT1[0], EVENT2[0]):
        s = stat(a["spread"], ev)
        print(f"{ev}: {s}")
    print("\n=== 事件窗口逐日（2024-01-02 ~ 2024-02-29） ===")
    show = a.loc["20240102":"20240229", ["micro", "large", "spread", "market"]]
    print(show.round(3).to_string())
    print("\n=== 单特征 spread 的诊断（事件前 10 日 z） ===")
    for col in ("z_turn", "z_range", "z_intra", "z_small"):
        a2 = agg(df, col)
        s1 = stat(a2["spread"], EVENT1[0])
        s2 = stat(a2["spread"], EVENT2[0])
        z1 = s1.get("z", float("nan"))
        z2 = s2.get("z", float("nan"))
        print(f"{col:8s} 0122 z={z1:+.2f} pass={s1.get('pass')} | "
              f"0205 z={z2:+.2f} pass={s2.get('pass')}")
    # 保存验证明细
    out = os.path.join(ROOT, "backtest", "quant_activity_validation.csv")
    a.to_csv(out)
    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()
