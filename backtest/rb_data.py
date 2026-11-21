"""风险预算研究 · 数据底座（PIT 市值带 + 市场脆弱度）

2026-11-21 Claude。为「大盘疲软 + 持仓浮盈」的退出规则提供可复核数据。

设计约束（吸取 Hermes 本轮的教训）：
  1. 分析单元 = 个股（不是组合代理）；组合层结论只能由个股事件聚合而来。
  2. 市值带必须 PIT：用快照日 total_mv / 当日 close 推隐含股本，再逐日 cap = 股本 × close。
     2022 年 mktcap 快照缺失，用 2021-12-29 的隐含股本前推 —— 这是近似，报告里必须标注。
  3. 收益用 pct_chg 累乘（不是 raw close），避免除权跳空。
  4. 所有指标在 t 日收盘可得；前向收益从 t 日收盘起算。

缓存：backtest/cache/rb_*.parquet
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data_storage/sw_index_data.db")
CAP_CSV = os.path.join(ROOT, "data_storage/mktcap_cache.csv")
CACHE = os.path.join(ROOT, "backtest/cache")
CAP_LO, CAP_HI = 1_000_000.0, 5_000_000.0  # 万元 = 100~500 亿


def _cached(name: str, builder, force: bool = False) -> pd.DataFrame:
    path = os.path.join(CACHE, f"rb_{name}.parquet")
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    df = builder()
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(path)
    return df


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c = sqlite3.connect(DB)
    px = pd.read_sql_query(
        "SELECT trade_date, ts_code, close, pct_chg, amount FROM stock_daily "
        "WHERE trade_date >= '20200101'", c)
    c.close()
    close = px.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    pct = px.pivot(index="trade_date", columns="ts_code", values="pct_chg").sort_index() / 100.0
    amount = px.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    # 收益用 pct_chg 累乘；不走 raw close，避免除权跳空
    adj = (1 + pct.fillna(0)).cumprod()
    return close, pct, amount, adj


def _build_cap(close: pd.DataFrame) -> pd.DataFrame:
    """PIT 市值：快照日 total_mv / close = 隐含股本，前推后逐日 cap = 股本 × close。"""
    snap = pd.read_csv(CAP_CSV, dtype={"trade_date": str})
    snap = snap[snap["trade_date"].isin(close.index)]
    shares = []
    for d, g in snap.groupby("trade_date"):
        g = g.set_index("ts_code")["total_mv"]
        cl = close.loc[d]
        s = g / cl.reindex(g.index)
        s.name = d
        shares.append(s)
    # shares: index=ts_code, columns=快照日 → 转成 行=日期、列=股票
    sh = pd.concat(shares, axis=1).T.sort_index()
    # 只在快照日有值；前推到所有交易日（不后填，避免未来函数）
    sh_daily = sh.reindex(close.index).ffill().reindex(columns=close.columns)
    cap = sh_daily * close
    return cap


def _board_limit(ts: str) -> float:
    if ts.startswith(("30", "68")):
        return 0.198
    if ts.startswith(("8", "4", "9")):
        return 0.298
    return 0.098


def _build_market(close: pd.DataFrame, pct: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    c = sqlite3.connect(DB)
    idx = pd.read_sql_query(
        "SELECT trade_date, close FROM sw_index_daily WHERE ts_code='000300.SH' "
        "AND trade_date >= '20190101' ORDER BY trade_date", c).set_index("trade_date")["close"]
    c.close()
    idx = idx.reindex(close.index).astype(float)
    m = pd.DataFrame(index=close.index)
    m["idx"] = idx
    m["g20"] = idx / idx.shift(20) - 1
    m["g60"] = idx / idx.shift(60) - 1
    m["ma20"] = idx.rolling(20).mean()
    m["ma60"] = idx.rolling(60).mean()
    ir = idx.pct_change()
    m["ivol20"] = ir.rolling(20).std()
    m["ivol60"] = ir.rolling(60).std()
    m["ivol_ratio"] = m["ivol20"] / m["ivol60"]

    ma20 = close.rolling(20).mean()
    valid = close.notna() & ma20.notna()
    n_valid = valid.sum(axis=1).replace(0, np.nan)
    m["above_ma20"] = (close > ma20).where(valid).sum(axis=1) / n_valid
    n_pct = pct.notna().sum(axis=1).replace(0, np.nan)
    m["up_ratio"] = (pct > 0).where(pct.notna()).sum(axis=1) / n_pct
    m["down_ratio"] = (pct < 0).where(pct.notna()).sum(axis=1) / n_pct

    # 涨跌停家数（按板块阈值近似）
    lim = pd.Series({x: _board_limit(x) for x in pct.columns})
    m["lu"] = (pct >= lim - 1e-9).sum(axis=1)
    m["ld"] = (pct <= -lim + 1e-9).sum(axis=1)

    # 强势篮子（近20日涨幅前20%）的「过去5日」相对收益；全部股票口径
    # ‼️ 必须是过去 5 日（t-5→t），不能用未来 5 日 —— 否则是未来函数
    r20 = close / close.shift(20) - 1
    rank = r20.rank(axis=1, pct=True)
    strong = rank >= 0.8
    past5_ret = adj / adj.shift(5) - 1
    m["strong_r5"] = past5_ret.where(strong).mean(axis=1)
    m["all_r5"] = past5_ret.mean(axis=1)
    m["strong_ex5"] = m["strong_r5"] - m["all_r5"]

    # 冲击计数（收盘可得）
    m["shock"] = (m["down_ratio"] >= 0.80).astype(int)
    m["shock5"] = m["shock"].rolling(5).sum()
    m["shock10"] = m["shock"].rolling(10).sum()

    # 脆弱度分（初版，阈值待校准；报告里做敏感性）
    score = pd.Series(0.0, index=m.index)
    score += np.where(m["g60"] < -0.05, 1, 0)
    score += np.where(m["g60"] < -0.10, 1, 0)
    score += np.where(m["above_ma20"] < 0.30, 1, 0)
    score += np.where(m["above_ma20"] < 0.20, 1, 0)
    score += np.where(m["shock"] == 1, 1, 0)
    score += np.where(m["shock10"] >= 2, 1, 0)
    score += np.where(m["strong_ex5"] < 0, 1, 0)
    score += np.where(m["ivol_ratio"] > 1.3, 1, 0)
    m["frag_score"] = score
    m["frag_state"] = pd.cut(
        score, bins=[-0.1, 1.5, 3.5, 5.5, 10],
        labels=["normal", "warning", "fragile", "crash"]).astype(str)
    return m


def load(force: bool = False) -> dict[str, pd.DataFrame]:
    close = _cached("close", lambda: _load_raw()[0], force)
    pct = _cached("pct", lambda: _load_raw()[1], force)
    amount = _cached("amount", lambda: _load_raw()[2], force)
    adj = _cached("adj", lambda: _load_raw()[3], force)
    cap = _cached("cap", lambda: _build_cap(close), force)
    market = _cached("market", lambda: _build_market(close, pct, adj), force)
    band = (cap >= CAP_LO) & (cap <= CAP_HI) & cap.notna()
    return dict(close=close, pct=pct, amount=amount, adj=adj, cap=cap,
                band=band, market=market)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = load(force=a.force)
    print("close", d["close"].shape, "band", int(d["band"].sum().sum()),
          "band per day tail:\n", d["band"].sum(axis=1).tail(8).to_string())
    print("\nmarket tail:\n", d["market"].tail(8)[
        ["g60", "above_ma20", "down_ratio", "shock5", "strong_ex5",
         "ivol_ratio", "frag_score", "frag_state"]].to_string())
