# -*- coding: utf-8 -*-
"""
波段决策单 (Decision Card) — 模型输出如何指导交易

设计原则: 模型不直接给"买/卖", 而是输出**分层决策单**, 每一行对应一个明确动作:
  ① 市场阶段   → 决定总仓位
  ② 行业选择   → 决定买哪个行业(按阶段选目标热度分位)
  ③ 个股候选   → 模型评分排序(当前用规则占位, 待波段模型替换)
  ④ 持仓卖出   → 止盈/止损信号
  ⑤ 纪律提醒   → 固定不变

Changelog
─────────
2026-09-25 Hermes: 新建。定义模型输出的使用接口。
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.mainline import STOCK_DB, INDUSTRY_CSV  # noqa: E402
from analysis.regime import current_regime  # noqa: E402

# 各阶段 → 目标行业热度分位区间 (来自 industry_heat_report.md 实证)
REGIME_TARGET_BAND = {
    "过冷/恐慌": (0, 30, "买最冷行业超跌反弹 (+5.2%/月, 胜率62%)"),
    "下跌趋势": (0, 30, "买最冷行业超跌 或 空仓 (+1.6%/月)"),
    "震荡/轮动": (70, 90, "买次热行业(热但不过热) (+1.0%/月)"),
    "过热": (30, 50, "买中间热度行业, 回避top10%最热 (+0.7%/月)"),
    "上涨趋势": (50, 90, "行业分位普遍为负, 取相对最优(偏热段); 靠模型选股+低仓位"),
}

# 阶段自适应行业臂 — 依据 backtest/report_signal_compare.py 实测(超额口径):
#   主线单独+0.39% | 全行业+2.46% | 震荡期「主线∩热度带」+2.88% | 过热期「主线」+2.77%
REGIME_ARM = {
    "过冷/恐慌": ("all", "全行业（本阶段行业筛选无增益）"),
    "下跌趋势": ("all", "全行业（本阶段行业筛选无增益）"),
    "震荡/轮动": ("band_ml", "主线∩热度带（实测超额+2.88%）"),
    "上涨趋势": ("all", "全行业（本阶段行业筛选无增益）"),
    "过热": ("mainline", "主线（过热期实测最优+2.77%）"),
}


def mainline_inds(asof, topk=12):
    """当前主线行业(set) — 复用 analysis.mainline 的 ml20 topk。"""
    try:
        from analysis.mainline import current_mainlines
        r = current_mainlines(asof, level="L2", topk=topk)
        return {x["ind"] for x in r.get("ranking", [])}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] mainline 失败: {str(e)[:80]}", file=sys.stderr)
        return set()
REGIME_POSITION = {"过冷/恐慌": "30-50%（试探建仓）", "下跌趋势": "0-30%（防守）",
                   "震荡/轮动": "30-60%", "过热": "0-30%（防守）",
                   "上涨趋势": "10-30%（低仓+模型选股）"}

# 业内标准口径(技术性牛熊 ±20%)下, 反转引擎的历史**超额**(跑赢沪深300) — 来自 TASK-20260925-022 实证
STD_EXCESS = {
    "技术性牛市": {"excess": "+2.54%/月", "rel_win": "64%", "n": 28},
    "技术性震荡": {"excess": "+2.37%/月", "rel_win": "53%", "n": 30},
    "技术性熊市": {"excess": "+2.71%/月", "rel_win": "100%", "n": 2},
}


def standard_state(asof):
    """业内标准口径: 技术性牛熊(距250日低点涨20%/高点跌20%) + 年线/半年线位置."""
    import sqlite3
    c = sqlite3.connect(STOCK_DB)
    try:
        d = pd.read_sql_query(
            "SELECT trade_date,close FROM sw_index_daily WHERE ts_code='000300.SH' ORDER BY trade_date", c)
    finally:
        c.close()
    idx = d.index[d["trade_date"] == str(asof)]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i < 250:
        return None
    close = float(d["close"].iloc[i])
    win = d["close"].iloc[i - 249:i + 1]
    lo, hi = float(win.min()), float(win.max())
    from_low, from_high = close / lo - 1, close / hi - 1
    ma120 = float(d["close"].rolling(120).mean().iloc[i])
    ma250 = float(d["close"].rolling(250).mean().iloc[i])
    if from_high <= -0.20:
        st = "技术性熊市"
    elif from_low >= 0.20:
        st = "技术性牛市"
    else:
        st = "技术性震荡"
    if close > ma120 and close > ma250:
        ma_pos = "年线+半年线上方"
    elif close < ma120 and close < ma250:
        ma_pos = "年线+半年线下方"
    else:
        ma_pos = "均线混合"
    return {"state": st, "from_low": from_low, "from_high": from_high,
            "ma_pos": ma_pos, "excess": STD_EXCESS.get(st)}


MATRIX = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
SWING_MODEL = os.path.join(PROJECT_ROOT, "data_storage", "swing_models", "swing_20241231.pkl")
CAP_CACHE = os.path.join(PROJECT_ROOT, "data_storage", "mktcap_cache.csv")
REPR = [-0.15, -0.02, 0.12, 0.30]
# 用户真实可交易域: 市值 100亿~500亿 (total_mv 单位=万元) — 回测证明 alpha 在此域最强且无生存者偏差
CAP_LO, CAP_HI = 1e6, 5e6


def market_caps(asof):
    """取 asof 日全部股票市值(Tushare daily_basic, 万元), 缓存累积."""
    cache = (pd.read_csv(CAP_CACHE, dtype={"trade_date": str})
             if os.path.exists(CAP_CACHE) else pd.DataFrame(columns=["trade_date", "ts_code", "total_mv"]))
    if str(asof) not in set(cache["trade_date"].unique()):
        try:
            import tushare as ts
            d = ts.pro_api().daily_basic(trade_date=str(asof), fields="ts_code,total_mv")
            d["trade_date"] = str(asof)
            cache = pd.concat([cache, d], ignore_index=True).drop_duplicates(subset=["trade_date", "ts_code"])
            cache.to_csv(CAP_CACHE, index=False)
        except Exception as e:
            print(f"[warn] 市值拉取失败({asof}): {str(e)[:80]}", file=sys.stderr)
    return cache[cache["trade_date"] == str(asof)][["ts_code", "total_mv"]]


def swing_scores(asof):
    """波段引擎输出: expect_ret / p_win / swing_score (截面分位). 模型缺失则返回 None."""
    if not os.path.exists(SWING_MODEL):
        return None
    import pickle
    import pyarrow.parquet as pq
    with open(SWING_MODEL, "rb") as f:
        art = pickle.load(f)
    m, feats = art["model"], art["features"]
    schema = set(pq.ParquetFile(MATRIX).schema_arrow.names)
    cols = ["ts_code"] + [c for c in feats if c in schema]
    if "beta_60d" in schema and "beta_60d" not in cols:
        cols.append("beta_60d")
    tbl = pq.read_table(MATRIX, columns=cols,
                        filters=[("trade_date", "==", str(asof))])
    X = tbl.to_pandas()
    if X.empty:
        return None
    proba = m.predict_proba(X[[c for c in feats if c in schema]].astype("float32"))
    out = pd.DataFrame({"ts_code": X["ts_code"].to_numpy(),
                        "expect_ret": proba @ np.array(REPR),
                        "p_win": proba[:, 3]})
    out["swing_score"] = out["expect_ret"].rank(pct=True) * 100
    if "beta_60d" in X.columns:
        out["beta_60d"] = X["beta_60d"].to_numpy()
    return out


def industry_heat(asof, lookback=400):
    cut = (pd.Timestamp(str(asof)) - pd.Timedelta(days=int(lookback * 1.6))).strftime("%Y%m%d")
    m = pd.read_csv(INDUSTRY_CSV, dtype=str)
    c2i = dict(zip(m["ts_code"], m["l2_name"]))
    conn = sqlite3.connect(STOCK_DB)
    try:
        d = pd.read_sql_query(
            "SELECT trade_date, ts_code, close FROM stock_daily WHERE trade_date BETWEEN ? AND ?",
            conn, params=[cut, str(asof)])
    finally:
        conn.close()
    d["ind"] = d["ts_code"].map(c2i)
    d = d.dropna(subset=["ind"]).sort_values(["ts_code", "trade_date"], kind="mergesort")
    g = d.groupby("ts_code", sort=False)["close"]
    d["r20"] = g.transform(lambda s: s / s.shift(20) - 1)
    d["vol20"] = g.transform(lambda s: s.pct_change().rolling(20).std())
    d["dd250"] = g.transform(lambda s: s / s.rolling(250).max() - 1)
    last = d[d["trade_date"] == str(asof)]
    ind = last.groupby("ind").agg(r20=("r20", "median"), n=("r20", "size")).dropna()
    ind["rank"] = ind["r20"].rank(pct=True) * 100
    return last, ind.sort_values("rank")


def decision_card(asof):
    rg = current_regime(asof)
    reg = rg["regime"]
    last, ind = industry_heat(asof)
    lo, hi, note = REGIME_TARGET_BAND.get(reg, (None, None, ""))
    out = {"asof": rg["asof"], "regime": reg, "position": REGIME_POSITION.get(reg, "—"),
           "action_note": note, "breadth": rg["breadth"], "dd250": rg["dd250"],
           "std": standard_state(asof)}

    arm, arm_note = REGIME_ARM.get(reg, ("all", ""))
    out["arm"] = arm
    out["arm_note"] = arm_note

    if arm == "all":
        out["industries_buy"] = []
    else:
        mls = mainline_inds(asof) if arm in ("mainline", "band_ml") else set()
        if arm == "mainline":
            sel = ind[ind.index.isin(mls)] if mls else ind.head(0)
        else:  # band_ml
            lo, hi = REGIME_TARGET_BAND.get(reg, (70, 90, ""))[:2]
            sel = ind[(ind["rank"] >= lo) & (ind["rank"] <= hi)]
            if mls:
                inter = sel[sel.index.isin(mls)]
                if len(inter) >= 3:
                    sel = inter
        out["industries_buy"] = [{"ind": i, "rank": round(float(r["rank"]), 0),
                                  "r20": round(float(r["r20"]) * 100, 1), "n": int(r["n"])}
                                 for i, r in sel.iterrows()]
    # 回避: 最热 top10%
    hot = ind[ind["rank"] >= 90]
    out["industries_avoid"] = [{"ind": i, "rank": round(float(r["rank"]), 0),
                                "r20": round(float(r["r20"]) * 100, 1)} for i, r in hot.iterrows()][:8]

    # 个股候选: 目标行业内, 按【波段模型 swing_score】排序
    sw = swing_scores(asof)
    out["model"] = "swing_engine(20241231)" if sw is not None else "规则占位(模型缺失)"
    if sw is not None:
        names = set(x["ind"] for x in out["industries_buy"])
        _base = last if (out.get("arm") == "all" or not names) else last[last["ind"].isin(names)]
        pool = _base.merge(sw, on="ts_code", how="inner")
        pool = pool.dropna(subset=["dd250", "vol20"])
        # 用户可交易域: 市值 100-500 亿
        caps = market_caps(asof)
        if not caps.empty:
            pool = pool.merge(caps, on="ts_code", how="inner")
            pool = pool[(pool["total_mv"] >= CAP_LO) & (pool["total_mv"] <= CAP_HI)]
        # 阶段自适应 Beta 倾斜 (T6: 仅震荡/过热压低β) — 回测 Calmar 0.88→1.33
        TILT = {"震荡/轮动": -1.0, "过热": -1.0}
        lam = TILT.get(reg, 0.0)
        if lam != 0.0 and "beta_60d" in pool.columns:
            pool = pool.copy()
            pool["sel_score"] = (pool["expect_ret"].rank(pct=True)
                                 + 0.2 * lam * pool["beta_60d"].rank(pct=True))
        else:
            pool = pool.assign(sel_score=pool["expect_ret"].rank(pct=True))
        cand = pool.sort_values("sel_score", ascending=False).head(10)
        out["_tilt_note"] = " · 震荡/过热压β" if reg in ("震荡/轮动", "过热") else ""
        _m = pd.read_csv(INDUSTRY_CSV, dtype=str)
        nmap = dict(zip(_m["ts_code"], _m["stock_name"]))
        out["stocks"] = [{"ts_code": r["ts_code"], "name": nmap.get(r["ts_code"], ""), "ind": r["ind"],
                          "score": round(float(r["swing_score"]), 1),
                          "p_win": round(float(r["p_win"]) * 100, 1),
                          "expect": round(float(r["expect_ret"]) * 100, 1),
                          "mv": round(float(r["total_mv"]) / 1e4, 0) if "total_mv" in r and pd.notna(r["total_mv"]) else None,
                          "dd250": round(float(r["dd250"]) * 100, 1),
                          "vol20": round(float(r["vol20"]) * 100, 2)}
                         for _, r in cand.iterrows()]
    else:
        out["stocks"] = []
    return out


def render(c):
    L = []
    L.append(f"📋 波段决策单 {c['asof']}")
    L.append("")
    L.append(f"【① 市场阶段】{c['regime']}")
    L.append(f"   广度 {c['breadth']*100:.0f}% · 距250日高点 {c['dd250']*100:+.1f}%")
    st = c.get("std")
    if st:
        L.append(f"   标准口径: {st['state']} · {st['ma_pos']} · "
                 f"距250日低点{st['from_low']*100:+.0f}% / 高点{st['from_high']*100:+.0f}%")
        if st.get("excess"):
            ex = st["excess"]
            L.append(f"   ↳ 该状态历史超额(跑赢沪深300): {ex['excess']} · 相对胜率 {ex['rel_win']} · n={ex['n']}期")
    L.append(f"   → 建议总仓位: {c['position']}")
    L.append(f"   → 本阶段最优动作: {c['action_note']}")
    L.append("")
    L.append(f"【② 行业选择】{c.get('arm_note','')}")
    if c["industries_buy"]:
        L.append(f"   ✅ 优先（{len(c['industries_buy'])} 个行业）:")
        for x in c["industries_buy"][:6]:
            L.append(f"      {x['ind']}　热度分位{x['rank']:.0f}%　20日{x['r20']:+.1f}%　{x['n']}只")
    else:
        L.append("   ○ 不限行业（全行业选股）")
    if c["industries_avoid"]:
        L.append(f"   🚫 回避（最热 top10%，局部过热）: {'、'.join(x['ind'] for x in c['industries_avoid'][:6])}")
    L.append("")
    tn = "· 震荡/过热压β" if c.get("regime") in ("震荡/轮动", "过热") else ""
    L.append(f"【③ 个股候选】（波段模型 {c.get('model','')} 排序 · 市值限100-500亿{tn}）")
    for s in c["stocks"]:
        pw = f"　P(≥20%){s['p_win']:.0f}%" if "p_win" in s else ""
        ex = f"　预期{s['expect']:+.1f}%" if "expect" in s else ""
        mv = f"　市值{s['mv']:.0f}亿" if s.get("mv") else ""
        L.append(f"   {s['name']}({s['ts_code']}) {s['ind']}{mv}　评分{s['score']}{pw}{ex}　距高点{s['dd250']:+.0f}%")
    if c["stocks"] and max(s.get("expect", 0) for s in c["stocks"]) < 0:
        L.append("   ⚠️ 当前候选预期收益全为负 → 模型建议观望，不建仓")
    L.append("")
    L.append("【④ 持仓卖出信号】（对已持仓，逐日检查）")
    L.append("   浮盈≥+10% → 移动止盈激活，从最高点回撤8%即卖")
    L.append("   浮亏≤-10% → 无条件止损")
    L.append("   持有>120交易日 → 时间止盈")
    L.append("")
    L.append("【⑤ 纪律】首仓≤1/3 · 同股间隔≥20日 · 单票≤25% · 行业≥4")
    return "\n".join(L)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_push(c, link=None):
    """Telegram HTML 推送格式（parse_mode=HTML，< > 已转义）"""
    L = [f"🎯 <b>波段决策单 {c['asof']}</b>", ""]
    L.append(f"<b>① 市场阶段</b> {c['regime']}")
    st = c.get("std")
    if st:
        L.append(f"　标准口径: {st['state']} · {st['ma_pos']}")
        if st.get("excess"):
            ex = st["excess"]
            L.append(f"　历史超额: <b>{ex['excess']}</b> · 相对胜率 {ex['rel_win']} (n={ex['n']}期)")
    L.append(f"　建议仓位: <b>{c['position']}</b>")
    if c.get("action_note"):
        L.append(f"　{c['action_note']}")
    L.append("")
    L.append(f"<b>② 行业</b> {c.get('arm_note','')}")
    if c["industries_buy"]:
        L.append("　✅ 优先: " + "、".join(f"{x['ind']}({x['rank']:.0f}%)" for x in c["industries_buy"][:5]))
    else:
        L.append("　○ 不限行业（全行业选股）")
    if c["industries_avoid"]:
        L.append("　🚫 回避: " + "、".join(x["ind"] for x in c["industries_avoid"][:5]))
    L.append("")
    L.append(f"<b>③ 个股候选</b>（市值100-500亿{c.get('_tilt_note','')}）")
    for i, s in enumerate(c["stocks"][:8], 1):
        mv = f" {s['mv']:.0f}亿" if s.get("mv") else ""
        pw = f" P≧20%:{s['p_win']:.0f}%" if "p_win" in s else ""
        ex = f" 预期{s['expect']:+.1f}%" if "expect" in s else ""
        L.append(f"　{i}. {_esc(s['name'])}({s['ts_code']}){mv}{pw}{ex}")
    if c["stocks"] and max(s.get("expect", 0) for s in c["stocks"]) < 0:
        L.append("　⚠️ 候选预期全为负 → 建议观望不建仓")
    L.append("")
    L.append("<b>④ 出场</b> 浮盈≥+10%后回撤8%卖 · 浮亏-10%止损（兜底）")
    L.append("<b>⑤ 纪律</b> 首仓≤1/3 · 同股间隔≥20日 · 单票≤25% · 行业≥4")
    if link:
        L.append("")
        L.append(f'📄 <a href="{link}">完整报告</a>')
    return "\n".join(L)


if __name__ == "__main__":
    asof = sys.argv[1] if len(sys.argv) > 1 else "20260924"
    c = decision_card(asof)
    print(render(c))
