# -*- coding: utf-8 -*-
"""
影子组合 + 策略归因 (复盘闭环)

两块:
  A. 策略归因 (回测): 读 backtest/v6_audit_full_*.json + *_daily_trades.csv,
     输出 按引擎 / Regime / win_prob 分档 的真实胜率与收益, 及组合累计收益/最大回撤。
     → 回答 "模型的建议历史上到底赚不赚钱、哪个引擎/环境有效、PWin 是否校准"。
  B. 实盘影子组合 (前向): 日报每日把"按建议应持有的 Top-N"记入 JSONL,
     次日用真实收盘价结算 T+1/D+5 收益 → 建立真实 track record, 不依赖用户手记。
     → 回答 "报告最近给的建议实际表现如何"。

Changelog
─────────
2026-09-25 Hermes: 新建。服务 V6 日报"我做得怎么样"层(复盘/归因)。
"""

import glob
import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
INDUSTRY_CSV = os.path.join(PROJECT_ROOT, "data_storage", "stock_industry_mapping.csv")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "backtest")
SHADOW_LOG = os.path.join(PROJECT_ROOT, "data_storage", "shadow_portfolio_log.jsonl")


# ═══════════════════════════════════════════════════════════════
# A. 策略归因 (回测)
# ═══════════════════════════════════════════════════════════════

def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def attribute_backtest():
    """从最新 walk-forward 审计产物生成策略归因。"""
    jpath = _latest(os.path.join(BACKTEST_DIR, "v6_audit_full_*.json"))
    cpath = _latest(os.path.join(BACKTEST_DIR, "v6_audit_full_*_daily_trades.csv"))
    if not jpath or not cpath or not os.path.exists(cpath):
        return None
    with open(jpath) as f:
        audit = json.load(f)
    tr = pd.read_csv(cpath)
    if tr.empty:
        return None

    def _wr(s):
        v = s.dropna()
        return float((v > 0).mean()) if len(v) else None

    out = {"source": os.path.basename(jpath), "n_trades": len(tr),
           "date_range": f"{tr['trade_date'].min()}~{tr['trade_date'].max()}",
           "summary": audit.get("summary", {})}

    # 按引擎
    eng = []
    for e, g in tr.groupby("gate_engine"):
        eng.append({"engine": e, "n": len(g),
                    "t1_wr": _wr(g["t1_ret_pct"]), "d5_wr": _wr(g["d5_ret_pct"]),
                    "t1_ret": float(g["t1_ret_pct"].mean()), "d5_ret": float(g["d5_ret_pct"].mean())})
    out["by_engine"] = sorted(eng, key=lambda x: -x["n"])

    # 按 Regime
    reg = []
    for r, g in tr.groupby("regime"):
        reg.append({"regime": r, "n": len(g),
                    "t1_wr": _wr(g["t1_ret_pct"]), "d5_wr": _wr(g["d5_ret_pct"]),
                    "d5_ret": float(g["d5_ret_pct"].mean())})
    out["by_regime"] = sorted(reg, key=lambda x: -x["n"])

    # PWin 校准: 模型说的概率 vs 实际胜率
    tr = tr.copy()
    tr["_pb"] = pd.cut(tr["win_prob"], bins=[0, 0.45, 0.55, 0.65, 0.75, 1.0],
                       labels=["<0.45", "0.45-0.55", "0.55-0.65", "0.65-0.75", "≥0.75"])
    cal = []
    for b, g in tr.groupby("_pb", observed=True):
        cal.append({"bucket": str(b), "n": len(g), "stated": float(g["win_prob"].mean()),
                    "actual_t1": _wr(g["t1_ret_pct"]), "actual_d5": _wr(g["d5_ret_pct"])})
    out["calibration"] = cal

    # 组合累计 (每日等权 T+1)
    day = tr.groupby("trade_date")["t1_ret_pct"].mean().sort_index() / 100.0
    cum = (1 + day).cumprod()
    out["portfolio"] = {
        "n_days": int(len(day)),
        "total_ret": float(cum.iloc[-1] - 1) if len(cum) else None,
        "win_days": float((day > 0).mean()) if len(day) else None,
        "max_dd": float((cum / cum.cummax() - 1).min()) if len(cum) else None,
        "avg_daily_bp": float(day.mean() * 10000) if len(day) else None,
    }
    return out


# ═══════════════════════════════════════════════════════════════
# B. 实盘影子组合 (前向跟踪)
# ═══════════════════════════════════════════════════════════════

def _name_map():
    try:
        m = pd.read_csv(INDUSTRY_CSV, dtype=str)
        return dict(zip(m["ts_code"], m["stock_name"])), dict(zip(m["ts_code"], m["l1_name"]))
    except Exception:
        return {}, {}


def record_picks(asof_date, ranking, size_label, max_per_industry=2):
    """把当日'按建议应持有'的 Top-N 记入影子日志 (幂等: 同日已记则跳过)。

    ranking: DataFrame(index=ts_code, columns 含 pwin_today, l1_name) — 来自 compute_decay["ranking"]
    size_label: 如 "3只 (半仓)" → N=3
    """
    if ranking is None or ranking.empty:
        return {"status": "skip", "reason": "empty_ranking"}
    import re
    m = re.search(r"(\d+)", str(size_label))
    n = int(m.group(1)) if m else 3
    n = max(1, min(n, 10))

    ranked = ranking.sort_values("pwin_today", ascending=False)
    picks, per_ind = [], {}
    for code, row in ranked.iterrows():
        ind = row.get("l1_name", "-")
        if per_ind.get(ind, 0) >= max_per_industry:
            continue
        picks.append({"ts_code": str(code), "name": row.get("name", ""), "l1_name": ind,
                      "pwin": round(float(row["pwin_today"]), 4)})
        per_ind[ind] = per_ind.get(ind, 0) + 1
        if len(picks) >= n:
            break

    rec = {"asof": str(asof_date), "recorded_at": datetime.now().isoformat(timespec="seconds"),
           "size_label": str(size_label), "picks": picks}
    # 幂等
    if os.path.exists(SHADOW_LOG):
        with open(SHADOW_LOG) as f:
            for line in f:
                try:
                    if json.loads(line).get("asof") == str(asof_date):
                        return {"status": "exists", "asof": str(asof_date), "n": len(picks)}
                except Exception:
                    continue
    with open(SHADOW_LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "ok", "asof": str(asof_date), "n": len(picks), "picks": picks}


def _forward_returns(db_path=STOCK_DB):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT trade_date, ts_code, close FROM stock_daily", conn)
    finally:
        conn.close()
    df = df.sort_values(["ts_code", "trade_date"])
    g = df.groupby("ts_code", sort=False)["close"]
    df["close_t1"] = g.shift(-1)
    df["close_t5"] = g.shift(-5)
    return df.set_index(["trade_date", "ts_code"])


def score_log(asof_date=None):
    """用真实收盘价结算影子日志, 返回聚合表现。"""
    if not os.path.exists(SHADOW_LOG):
        return None
    recs = []
    with open(SHADOW_LOG) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    if not recs:
        return None
    fwd = _forward_returns()
    rows = []
    for rec in recs:
        d = rec["asof"]
        for p in rec["picks"]:
            key = (d, p["ts_code"])
            if key not in fwd.index:
                continue
            c0, c1, c5 = fwd.loc[key, ["close", "close_t1", "close_t5"]]
            if pd.isna(c0):
                continue
            rows.append({"asof": d, "ts_code": p["ts_code"], "name": p.get("name", ""),
                         "l1_name": p.get("l1_name", "-"),
                         "t1": (c1 / c0 - 1) * 100 if pd.notna(c1) else np.nan,
                         "d5": (c5 / c0 - 1) * 100 if pd.notna(c5) else np.nan})
    if not rows:
        return {"n_picks": 0, "n_settled": 0, "recent_days": []}
    tr = pd.DataFrame(rows)
    settled_t1 = tr.dropna(subset=["t1"])
    settled_d5 = tr.dropna(subset=["d5"])
    # 按日等权
    daily = settled_t1.groupby("asof")["t1"].mean().sort_index()
    return {
        "n_picks": int(len(tr)),
        "n_settled_t1": int(len(settled_t1)),
        "n_settled_d5": int(len(settled_d5)),
        "t1_wr": float((settled_t1["t1"] > 0).mean()) if len(settled_t1) else None,
        "t1_avg": float(settled_t1["t1"].mean()) if len(settled_t1) else None,
        "d5_wr": float((settled_d5["d5"] > 0).mean()) if len(settled_d5) else None,
        "d5_avg": float(settled_d5["d5"].mean()) if len(settled_d5) else None,
        "cum_t1": float(((1 + daily / 100).cumprod().iloc[-1] - 1) * 100) if len(daily) else None,
        "recent_days": [{"asof": d, "t1_avg": round(float(v), 2)} for d, v in daily.tail(8).items()],
    }


if __name__ == "__main__":
    a = attribute_backtest()
    if a:
        print(f"=== 策略归因 ({a['source']}) ===")
        print(f"交易 {a['n_trades']} 笔 · {a['date_range']}")
        print("\n按引擎:")
        for e in a["by_engine"]:
            print(f"  {e['engine']:10s} n={e['n']:4d} T+1胜率 {e['t1_wr']*100:5.1f}% D+5胜率 {e['d5_wr']*100:5.1f}% "
                  f"D+5均 {e['d5_ret']:+.2f}%")
        print("\n按Regime:")
        for r in a["by_regime"]:
            print(f"  {r['regime']:6s} n={r['n']:4d} T+1胜率 {r['t1_wr']*100:5.1f}% D+5均 {r['d5_ret']:+.2f}%")
        print("\nPWin 校准:")
        for c in a["calibration"]:
            print(f"  说 {c['bucket']:10s} n={c['n']:4d} 实T+1 {c['actual_t1']*100:5.1f}% 实D+5 {c['actual_d5']*100:5.1f}%")
        p = a["portfolio"]
        print(f"\n组合: {p['n_days']} 日, 累计 {p['total_ret']*100:+.1f}%, 日胜率 {p['win_days']*100:.1f}%, "
              f"最大回撤 {p['max_dd']*100:.1f}%, 日均 {p['avg_daily_bp']:+.1f}bp")
    s = score_log()
    print("\n=== 实盘影子组合 ===")
    print(json.dumps(s, ensure_ascii=False, indent=2))
