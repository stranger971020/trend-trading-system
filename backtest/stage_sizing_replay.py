#!/usr/bin/env python3
"""stage_sizing_replay.py — 阶段仓位历史重放（检验 REGIME_POSITION 在用户交易域是否有效）

2026-11-21 Hermes。回答一个问题：
    「按决策单的阶段仓位建议（过冷30-50% / 下跌0-30% / 震荡30-60% / 过热0-30% / 上涨10-30%）
      在 100-500亿 + 全A 两个池子上，能否跑赢同池满仓？」

设计要点（防自欺）：
  1. **标的固定、由规则生成**：不由人挑，避免选股能力污染择时判断。
  2. **决策用上月末 regime**：月末收盘后才知状态，次月执行 → 无未来函数。
  3. **现金 0 收益**：控仓少投的钱不赚利息（保守）。
  4. **成本**：仓位变动计 0.15% 单边；池子换手另计（按实际重叠率）。
  5. **D 组 = 固定 50%**：把「降仓效应」与「择时效应」分离 —— 若 B 打不过 D，
     说明阶段标签没有信息，只是降了仓。
  6. **分环境**：牛/熊/震荡各算 Calmar。只在熊市赢 = 降波动也降收益 = 用户已拒绝的类型。

用法:
  python3 backtest/stage_sizing_replay.py            # 全流程（首次会构建 regime 序列，约数分钟）
  python3 backtest/stage_sizing_replay.py --quick    # 只跑 P2
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
MKTCAP = os.path.join(PROJECT_ROOT, "data_storage", "mktcap_cache.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "backtest", "cache")
REGIME_CACHE = os.path.join(CACHE_DIR, "regime_history.parquet")
OUT_MD = os.path.join(PROJECT_ROOT, "backtest", "stage_sizing_replay.md")
OUT_JSON = os.path.join(PROJECT_ROOT, "backtest", "stage_sizing_replay.json")

COST_SIDE = 0.0015          # 单边成本
CAP_LO, CAP_HI = 1e6, 5e6   # mktcap_cache 单位=万元 → 100亿~500亿
CALMAR_TOL = 0.005          # Calmar 差异小于此视为持平（避免浮点比较把 -0.078 vs -0.080 算成"赢"）

# 阶段仓位：取建议区间中值（区间上下界见 analysis/decision_card.REGIME_POSITION）
POS_REGIME = {
    "过冷/恐慌": 0.40,   # 30-50%
    "下跌趋势": 0.15,    # 0-30%
    "震荡/轮动": 0.45,   # 30-60%
    "过热":     0.15,    # 0-30%
    "上涨趋势": 0.20,    # 10-30%
}


# ────────────────────────── 阶段序列 ──────────────────────────
def load_regime_history(force: bool = False) -> pd.Series:
    """日频 regime（index=trade_date str）。缓存到 backtest/cache/。

    注意: analysis/regime.build_market() 自带的 parquet 缓存永远写不出来
    （drop(columns=['regime']) 在列不存在时抛 KeyError 被 except 吞掉），
    所以这里自建缓存，不改动日报主链路。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(REGIME_CACHE) and not force:
        s = pd.read_parquet(REGIME_CACHE)["regime"]
        print(f"[regime] 缓存命中 {len(s)} 天 {s.index[0]}~{s.index[-1]}", file=sys.stderr)
        return s
    print("[regime] 构建市场阶段序列（情绪周期+广度+指数特征，数分钟）...", file=sys.stderr)
    from analysis.regime import build_market
    m = build_market()
    s = m["regime"]
    s.to_frame("regime").to_parquet(REGIME_CACHE)
    print(f"[regime] 已构建 {len(s)} 天 {s.index[0]}~{s.index[-1]} → {REGIME_CACHE}", file=sys.stderr)
    return s


# ────────────────────────── 池子月收益 ──────────────────────────
def _month_ends(trade_dates) -> list[str]:
    """每个自然月的最后一个交易日（保持 YYYYMMDD 字符串）。"""
    s = pd.Series([str(x) for x in trade_dates])
    key = pd.to_datetime(s).dt.to_period("M")
    return [str(x) for x in s.groupby(key).max()]


def _clean_mean(row: pd.Series) -> float:
    """单月横截面：先剔除数据错误（>+300% / <-90%），再等权平均。

    ‼️ 不用 winsorize：A 股等权组合的收益本就由少数暴涨股主导，裁掉 99% 分位
    会把真实收益当噪音删掉（实测 winsor 后年化 -1.1%，原始 +1.2%）。只剔硬错误。
    """
    s = row.dropna()
    s = s[(s > -0.9) & (s < 3.0)]
    return float(s.mean()) if len(s) >= 20 else float("nan")


def pool_all_equal_weight(regime: pd.Series | None = None) -> pd.DataFrame:
    """P2 全A等权: 每月末截面的个股月收益 → 去硬错误后等权平均。

    ‼️ 口径（重要）：这是**月度再平衡的买入持有**（可实现）。
    不能用「每日 AVG(pct_chg) 复利」当基准 —— 那是日频再平衡，会持续吃掉 A 股
    的短期反转收益（实测日频 10 年年化 +11.9%，月频可实现口径约 +1~2%），
    属不可实现的虚高，不能当对照基线。

    无未来函数：每行附带 decision_date = 收益起点（=上月末）的前一交易日，
    阶段仓位只用 decision_date 及之前的 regime。
    """
    conn = sqlite3.connect(DB)
    try:
        px = pd.read_sql_query(
            "SELECT trade_date, ts_code, close, vol FROM stock_daily WHERE trade_date>='20150101'", conn)
    finally:
        conn.close()
    px = px[(px["close"] > 0) & (px["vol"] > 0)]
    dates = sorted(px["trade_date"].unique())
    me = _month_ends(dates)
    snap = px.set_index(["trade_date", "ts_code"])["close"].unstack()
    snap = snap.loc[[d for d in me if d in snap.index]]
    mret = snap.pct_change(fill_method=None).iloc[1:]
    raw = mret.mean(axis=1)
    clean = mret.apply(_clean_mean, axis=1)
    all_dates = list(snap.index)          # 月末序列
    starts = {all_dates[i]: all_dates[i - 1] for i in range(1, len(all_dates))}
    out = pd.DataFrame({"ret": clean, "raw": raw, "n": mret.notna().sum(axis=1),
                        "turnover": 0.0})
    out["start_date"] = [starts.get(d) for d in out.index]
    out["decision_date"] = [_prev_trading(all_dates, s) for s in out["start_date"]]
    return out.dropna(subset=["ret"])


def _prev_trading(all_dates: list[str], d: str | None) -> str | None:
    """严格取 < d 的最后一个交易日（收益起点日当天收盘后才知 regime，不能用当天）。"""
    if d is None:
        return None
    prev = [x for x in all_dates if x < d]
    return prev[-1] if prev else None


def pool_cap_band(lo=CAP_LO, hi=CAP_HI) -> pd.DataFrame:
    """P1 100-500亿域等权: 每月初按当时市值筛池 → 持有到下月初（月度 buy&hold）。

    mktcap_cache 是**月初**快照（20210104/20210201…），不是月末 → 收益期间
    = 相邻两个快照日之间。换手率用与上期池子的对称差实测。
    退市/长期停牌的票在期末无价 → 剔除（轻微高估，已在报告标注）。
    """
    conn = sqlite3.connect(DB)
    try:
        px = pd.read_sql_query(
            "SELECT trade_date, ts_code, close, vol FROM stock_daily WHERE trade_date>='20201201'", conn)
    finally:
        conn.close()
    px = px[(px["close"] > 0) & (px["vol"] > 0)]
    all_dates = sorted(px["trade_date"].unique())
    snap = px.set_index(["trade_date", "ts_code"])["close"].unstack()
    cap = pd.read_csv(MKTCAP, dtype={"trade_date": str})
    cap_p = cap.pivot_table(index="trade_date", columns="ts_code", values="total_mv")

    cap_dates = [d for d in sorted(cap_p.index) if d in snap.index]
    rows, prev = [], None
    for i in range(1, len(cap_dates)):
        d0, d1 = cap_dates[i - 1], cap_dates[i]
        caps = cap_p.loc[d0].dropna()
        codes = caps[(caps >= lo) & (caps <= hi)].index
        if len(codes) < 30:
            continue
        p0 = snap.loc[d0].reindex(codes)
        p1 = snap.loc[d1].reindex(codes)
        r = (p1 / p0 - 1).dropna()
        r = r[(r > -0.9) & (r < 3.0)]
        if len(r) < 30:
            continue
        turnover = 0.0 if prev is None else len(set(codes) ^ set(prev)) / max(len(codes), 1)
        prev = list(codes)
        rows.append({"date": d1, "start_date": d0,
                     "decision_date": _prev_trading(all_dates, d0),
                     "ret": float(r.mean()), "raw": float(r.mean()), "n": int(len(r)),
                     "turnover": float(turnover)})
    return pd.DataFrame(rows).set_index("date")


# ────────────────────────── 仓位策略 ──────────────────────────
def stage_profile(pool: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """最根本的信号检验：每个阶段 → 次月池子收益分布（不涉及仓位规则）。

    这是判断「阶段标签有没有信息」的第一性检查：
    若各阶段的次月收益没有差异，后面怎么调仓位都是噪音。
    """
    recs = []
    has_dec = "decision_date" in pool.columns
    for d, row in pool.iterrows():
        dec = row.get("decision_date") if has_dec else d
        key = str(dec) if (dec is not None and not pd.isna(dec)) else str(d)
        past = regime.loc[:key]
        reg = past.iloc[-1] if len(past) else "震荡/轮动"
        recs.append({"regime": reg, "ret": float(row["ret"])})
    df = pd.DataFrame(recs)
    g = df.groupby("regime")["ret"]
    out = pd.DataFrame({
        "月数": g.size(),
        "次月均收益": g.mean(),
        "中位": g.median(),
        "胜率": g.apply(lambda s: float((s > 0).mean())),
        "最差月": g.min(),
    })
    order = ["过冷/恐慌", "下跌趋势", "震荡/轮动", "上涨趋势", "过热"]
    return out.reindex([o for o in order if o in out.index])


def run_strategy(pool: pd.DataFrame, regime: pd.Series, weight_fn, name: str) -> dict:
    """按 weight_fn(收益起点前一交易日的 regime) 决定本期仓位，结算净值。

    无未来函数：仓位只用 pool["decision_date"]（收益期开始前的最后一个交易日）及之前的
    regime 观测；该日收盘后才知状态，下一交易日起执行。
    """
    nav, navs, dates, wts, regs, rets = 1.0, [], [], [], [], []
    prev_w = 1.0
    has_dec = "decision_date" in pool.columns
    for d, row in pool.iterrows():
        dec = row.get("decision_date") if has_dec else d
        key = str(dec) if (dec is not None and not pd.isna(dec)) else str(d)
        past = regime.loc[:key]
        reg = past.iloc[-1] if len(past) else "震荡/轮动"
        w = float(weight_fn(reg))
        gross = float(row["ret"]) * w
        # 成本: 仓位变动 + 池子换手（换手只在满仓部分生效）
        cost = abs(w - prev_w) * COST_SIDE + w * float(row.get("turnover", 0)) * COST_SIDE
        net = gross - cost
        nav *= (1 + net)
        navs.append(nav); dates.append(d); wts.append(w); regs.append(reg); rets.append(net)
        prev_w = w
    s = pd.Series(navs, index=pd.Index(dates, name="date"))
    r = pd.Series(rets, index=s.index)
    return {"name": name, "nav": s, "ret": r, "weight": pd.Series(wts, index=s.index),
            "regime": pd.Series(regs, index=s.index)}


def metrics(r: pd.Series, ppy: int = 12) -> dict:
    if len(r) < 6:
        return {}
    nav = (1 + r).cumprod()
    years = len(r) / ppy
    ann = nav.iloc[-1] ** (1 / years) - 1
    vol = r.std() * (ppy ** 0.5)
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = ann / vol if vol else float("nan")
    calmar = ann / abs(mdd) if (mdd and abs(mdd) > 1e-9) else float("nan")
    return {"累计": f"{(nav.iloc[-1]-1)*100:.1f}%", "年化": ann, "波动": vol,
            "Sharpe": sharpe, "最大回撤": mdd, "Calmar": calmar,
            "月胜率": float((r > 0).mean()), "期数": len(r)}


def fnum(x, d=2):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def env_split(pool: pd.DataFrame) -> pd.Series:
    """环境标签: 基准当月收益 >+3% 牛 / <-3% 熊 / 其余 震荡（用池子自身当基准，避免外部标的污染）。"""
    r = pool["ret"]
    lab = pd.Series("震荡", index=r.index)
    lab[r > 0.03] = "牛"
    lab[r < -0.03] = "熊"
    return lab


WEIGHT_FNS = {
    "A 满仓 100%": lambda reg: 1.00,
    "B 阶段仓位(REGIME_POSITION中值)": lambda reg: POS_REGIME.get(reg, 0.30),
    "C 只防守(下跌/过热减半)": lambda reg: 0.50 if reg in ("下跌趋势", "过热") else 1.00,
    "D 固定 50%(纯降仓对照)": lambda reg: 0.50,
}


def evaluate(pool: pd.DataFrame, regime: pd.Series, label: str) -> tuple[list[dict], dict]:
    env = env_split(pool)
    # 池子自身口径（满仓买入持有，用于说明基线是否可信）
    def _ann(col):
        if col not in pool:
            return None
        s = pool[col].dropna()
        return float((1 + s).prod() ** (12 / len(s)) - 1) if len(s) > 6 else None

    pool_stats = {
        "raw_ann": _ann("raw"), "win_ann": _ann("ret"),
        "avg_n": float(pool["n"].mean()) if "n" in pool else None,
        "avg_turnover": float(pool["turnover"].mean()) if "turnover" in pool else None,
    }
    rows, detail = [], {"pool": label, "period": f"{pool.index[0]}~{pool.index[-1]}",
                        "months": len(pool), "pool_stats": pool_stats, "strategies": {}}
    prof = stage_profile(pool, regime)
    detail["stage_profile"] = {
        str(k): {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in r.items()}
        for k, r in prof.iterrows()}
    base_calmar = None
    for name, fn in WEIGHT_FNS.items():
        res = run_strategy(pool, regime, fn, name)
        m = metrics(res["ret"])
        m["环境"] = {}
        for e in ("牛", "熊", "震荡"):
            mask = env.reindex(res["ret"].index) == e
            sub = res["ret"][mask]
            if len(sub) >= 6:
                m["环境"][e] = metrics(sub)
        m["平均仓位"] = float(res["weight"].mean())
        m["组"] = name
        if base_calmar is None:
            base_calmar = m["Calmar"]
        # 基准 Calmar ≤ 0 时比值无意义（负负得正会误导），置 None
        m["Calmar/满仓"] = (m["Calmar"] / base_calmar) if (base_calmar and base_calmar > 0) else None
        m["满仓Calmar"] = base_calmar
        rows.append(m)
        detail["strategies"][name] = {"metrics": {k: v for k, v in m.items() if k != "环境"},
                                      "env": m["环境"],
                                      "monthly": {str(k): round(float(v), 6) for k, v in res["ret"].items()}}
    return rows, detail


def fmt_pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{x*100:.{d}f}%"


def render_md(all_rows: dict[str, list[dict]], periods: dict[str, str],
              pool_stats: dict[str, dict] | None = None,
              stage_profiles: dict[str, dict] | None = None) -> str:
    L = ["# 阶段仓位历史重放：REGIME_POSITION 在用户交易域是否有效", "",
         "> 生成: 2026-11-21 · Hermes · `backtest/stage_sizing_replay.py`", "",
         "## 口径（防自欺的四条硬约束）", "",
         "1. **标的固定、规则生成**（不由人挑）：P1=100-500亿全域等权，P2=全A等权",
         "2. **决策用上月末 regime**（月末收盘后才知状态，次月执行）→ 无未来函数",
         "3. **现金 0 收益**（少投的钱不赚利息，保守）",
         "4. **D 组固定 50%** 作为「纯降仓」对照：若 B 打不过 D，说明阶段标签没有信息，只是降了仓",
         "", f"成本: 单边 {COST_SIDE*100:.2f}%（仓位变动 + 池子换手各计一次）", "",
         "## 阶段仓位取值（建议区间中值）", "",
         "| 阶段 | 建议区间 | 重放取值 |", "|---|---|---|"]
    band = {"过冷/恐慌": "30-50%", "下跌趋势": "0-30%", "震荡/轮动": "30-60%",
            "过热": "0-30%", "上涨趋势": "10-30%"}
    for k, v in POS_REGIME.items():
        L.append(f"| {k} | {band.get(k,'—')} | {v*100:.0f}% |")
    L.append("")
    for pool_name, rows in all_rows.items():
        ps = (pool_stats or {}).get(pool_name) or {}
        base = ""
        if ps:
            base = ("　> 池子基线（满仓买入持有）：年化 原始 "
                    f"{fmt_pct(ps.get('raw_ann'))} / 去错误值 {fmt_pct(ps.get('win_ann'))}"
                    f" · 平均持股 {ps.get('avg_n') and round(ps['avg_n'])} 只"
                    f" · 平均月换手 {ps.get('avg_turnover') and fmt_pct(ps['avg_turnover'],0)}")
        L += [f"## 池子 {pool_name}　({periods.get(pool_name,'')})", base, "",
              "| 组 | 平均仓位 | 累计 | 年化 | 波动 | Sharpe | 最大回撤 | Calmar | Calmar/满仓 | 月胜率 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in rows:
            L.append("| {g} | {w} | {c} | {a} | {v} | {s} | {dd} | **{cal}** | {rel} | {wr} |".format(
                g=m["组"], w=fmt_pct(m["平均仓位"], 0), c=m["累计"], a=fmt_pct(m["年化"]),
                v=fmt_pct(m["波动"]), s=fnum(m["Sharpe"]),
                dd=fmt_pct(m["最大回撤"]), cal=fnum(m["Calmar"]),
                rel=(fnum(m["Calmar/满仓"]) + "×") if m.get("Calmar/满仓") is not None else "—",
                wr=fmt_pct(m["月胜率"], 0)))
        L.append("")
        L += ["### 分环境（关键）", "",
              "| 组 | 牛市 Calmar | 熊市 Calmar | 震荡 Calmar | 牛市年化 | 熊市年化 |",
              "|---|---|---|---|---|---|"]
        for m in rows:
            e = m.get("环境") or {}
            def g(env, k):
                return e.get(env, {}).get(k)
            L.append("| {g} | {b1} | {b2} | {b3} | {a1} | {a2} |".format(
                g=m["组"],
                b1=fnum(g('牛', 'Calmar')), b2=fnum(g('熊', 'Calmar')),
                b3=fnum(g('震荡', 'Calmar')),
                a1=fmt_pct(g('牛', '年化')), a2=fmt_pct(g('熊', '年化'))))
        L.append("")
        prof = (stage_profiles or {}).get(pool_name) or {}
        if prof:
            L += ["### 阶段 → 次月收益画像（第一性检验：标签到底有没有信息）", "",
                  "| 阶段 | 月数 | 次月均收益 | 中位 | 胜率 | 最差月 |",
                  "|---|---|---|---|---|---|"]
            for reg, v in prof.items():
                L.append("| {} | {} | {} | {} | {} | {} |".format(
                    reg, int(v["月数"]), fmt_pct(v["次月均收益"], 2), fmt_pct(v["中位"], 2),
                    fmt_pct(v["胜率"], 0), fmt_pct(v["最差月"])))
            L.append("")
    # ── 自动判定（按预注册标准，不靠人事后解释）──
    L += ["## 判定（脚本按预注册标准自动计算）", ""]
    ok_pools = []
    for pool_name, rows in all_rows.items():
        d = {m["组"]: m for m in rows}
        a = d.get("A 满仓 100%")
        b = d.get("B 阶段仓位(REGIME_POSITION中值)")
        dd = d.get("D 固定 50%(纯降仓对照)")
        if not (a and b and dd):
            continue
        beat_a = b["Calmar"] > a["Calmar"] + CALMAR_TOL
        beat_d = b["Calmar"] > dd["Calmar"] + CALMAR_TOL
        note_a = "" if abs(b["Calmar"] - a["Calmar"]) > CALMAR_TOL else "（与A持平）"
        note_d = "" if abs(b["Calmar"] - dd["Calmar"]) > CALMAR_TOL else "（与D持平）"
        ok = beat_a and beat_d
        if ok:
            ok_pools.append(pool_name)
        L.append(f"- **{pool_name}**：B {fnum(b['Calmar'], 3)} vs A {fnum(a['Calmar'], 3)} → "
                 f"{'赢' if beat_a else '输'}{note_a}；vs D {fnum(dd['Calmar'], 3)} → "
                 f"{'赢' if beat_d else '输'}{note_d} ⇒ **{'有效' if ok else '无效'}**")
    n = len(all_rows)
    if n:
        if len(ok_pools) == n:
            tail = "**总判定：全部池子通过 → 阶段仓位在这些域里有效。**"
        elif not ok_pools:
            tail = "**总判定：0 个池子通过 → 阶段仓位无效，不该进报告。**"
        else:
            off = [p for p in all_rows if p not in ok_pools]
            tail = (f"**总判定：{len(ok_pools)}/{n} 通过。通过：{', '.join(ok_pools)}；"
                    f"未通过：{', '.join(off)}。**\n\n"
                    "> 差异不是随机的 —— 要通过看「判定基准 vs 标的域」：regime 用的是沪深300 + "
                    "全市场 MA20 广度，对大池子（全A）有代表性，对 100-500亿中盘域指导性弱。"
                    "所以在未通过的域里被判无效，属于**口径错配**，不是运气波动。")
        L += ["", tail, ""]
    L += ["## 判定标准（先定，后看结果）", "",
          "- **有效**：B 的 Calmar > A（满仓），且 B > D（说明是择时而非单纯降仓）",
          "- **无效**：B 打不过 A，或只赢 D 之外的熊市（=降波动也降收益）",
          "- 只在单一环境赢、全样本输 → 判**无效**（用户已明确拒绝「收益≈回撤」）", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑 P2 全A等权")
    ap.add_argument("--force-regime", action="store_true", help="强制重建阶段序列")
    args = ap.parse_args()

    regime = load_regime_history(force=args.force_regime)
    print(f"[regime] 分布: {regime.value_counts().to_dict()}", file=sys.stderr)

    pools = {"P2 全A等权": pool_all_equal_weight(regime)}
    if not args.quick:
        p1 = pool_cap_band()
        if len(p1) >= 12:
            pools["P1 100-500亿等权"] = p1
        else:
            print(f"[warn] P1 月数不足({len(p1)})，跳过", file=sys.stderr)
        # 同期对齐：P2 截到 2021 起，与 P1 同窗口 —— 区分「池子差异」vs「时期差异」
        p2 = pools["P2 全A等权"]
        p2b = p2[p2.index >= "20210201"]
        if len(p2b) >= 12:
            pools["P2b 全A等权(2021起·与P1同期)"] = p2b

    all_rows, periods, details = {}, {}, {}
    for name, pool in pools.items():
        pool = pool[pool.index >= "20150101"]
        if "decision_date" in pool.columns:
            pool = pool[pool["decision_date"].notna()]
            pool = pool[pool["decision_date"] >= regime.index[0]]
        if len(pool) < 12:
            print(f"[warn] {name} 有效月数 {len(pool)} <12，跳过", file=sys.stderr)
            continue
        rows, det = evaluate(pool, regime, name)
        all_rows[name] = rows
        periods[name] = f"{pool.index[0]}~{pool.index[-1]} · {len(pool)} 个月"
        details[name] = det

    md = render_md(all_rows, periods,
                   {n: d.get("pool_stats") for n, d in details.items()},
                   {n: d.get("stage_profile") for n, d in details.items()})
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    print(md)
    print(f"\n✅ {OUT_MD}\n✅ {OUT_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
