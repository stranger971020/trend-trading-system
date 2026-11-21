"""每日风险状态 · 大盘脆弱度 + 持仓动作建议

2026-11-21 Claude。与回测同口径（PIT）：
  - 市场脆弱度：趋势/广度/冲击/拥挤/波动 五维 0~8 分
  - 状态：normal / warning / fragile / crash
  - 动作：初减 0 / 0 / 25% / 40%；弱市+第二次冲击 → 50%
  - 剩余仓位：峰值回撤 8% 移动止损（不预测顶，不因单日下跌清仓）

输出 JSON 给 hermes-dashboard ⑦ 段消费。用法：
  python3 analysis/risk_state.py --out ~/projects/hermes-dashboard/risk_state.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backtest"))
import rb_data  # noqa: E402

TRAIL = 0.08
LABEL = {"normal": "正常", "warning": "警戒", "fragile": "脆弱", "crash": "崩跌"}
ACTION = {"normal": 0.0, "warning": 0.0, "fragile": 0.25, "crash": 0.40}


def latest_db_date() -> str:
    c = sqlite3.connect(rb_data.DB)
    d = c.execute("SELECT MAX(trade_date) FROM stock_daily").fetchone()[0]
    c.close()
    return d


def load_fresh() -> dict:
    latest = latest_db_date()
    p = os.path.join(rb_data.CACHE, "rb_market.parquet")
    stale = True
    if os.path.exists(p):
        try:
            stale = pd.read_parquet(p).index.max() != latest
        except Exception:
            stale = True
    return rb_data.load(force=stale)


def build(out_path: str | None = None) -> dict:
    d = load_fresh()
    m = d["market"]
    asof = str(m.index.max())
    r = m.loc[asof]

    comp = {
        "trend_60d_lt_-5": bool(r["g60"] < -0.05),
        "trend_60d_lt_-10": bool(r["g60"] < -0.10),
        "breadth_ma20_lt_30": bool(r["above_ma20"] < 0.30),
        "breadth_ma20_lt_20": bool(r["above_ma20"] < 0.20),
        "shock_today": bool(r["down_ratio"] >= 0.80),
        "second_shock_10d": bool(r["shock10"] >= 2),
        "strong_basket_rolling": bool(r["strong_ex5"] < 0),
        "vol_expansion": bool(r["ivol_ratio"] > 1.3),
    }
    score = int(sum(comp.values()))
    state = ("normal" if score <= 1 else "warning" if score <= 3
             else "fragile" if score <= 5 else "crash")

    weak = bool(r["g60"] < -0.05)
    second = bool(r["shock10"] >= 2)
    trim = ACTION[state]
    if weak and second:
        trim = max(trim, 0.50)  # R7：第二次冲击 → 至少减半
    trim = min(trim, 0.50)

    hist = m.loc[m.index[-20:], ["frag_score", "frag_state", "down_ratio",
                                 "above_ma20", "g60", "shock10"]].reset_index()
    hist["trade_date"] = hist["trade_date"].astype(str)
    history = hist.to_dict(orient="records")

    val = {}
    vp = os.path.join(ROOT, "backtest", "risk_budget_metrics.json")
    if os.path.exists(vp):
        try:
            with open(vp) as f:
                val = json.load(f)
        except Exception:
            val = {}

    payload = {
        "schema": 1,
        "asof": asof,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "state": state,
        "state_label": LABEL[state],
        "score": score,
        "components": comp,
        "metrics": {
            "g60": round(float(r["g60"]), 4),
            "g20": round(float(r["g20"]), 4),
            "above_ma20": round(float(r["above_ma20"]), 4),
            "down_ratio": round(float(r["down_ratio"]), 4),
            "shock5": int(r["shock5"]),
            "shock10": int(r["shock10"]),
            "strong_ex5": round(float(r["strong_ex5"]), 4)
            if pd.notna(r["strong_ex5"]) else None,
            "ivol_ratio": round(float(r["ivol_ratio"]), 4),
            "lu": int(r["lu"]), "ld": int(r["ld"]),
        },
        "action": {
            "initial_trim": trim,
            "initial_trim_reason": (
                "弱市 + 10日内第二次广泛下跌 → 至少减半" if weak and second
                else f"{LABEL[state]}状态 → 初减 {trim:.0%}" if trim > 0
                else "正常/警戒 → 不初减"),
            "trail_stop": TRAIL,
            "note": "初减后剩余仓位按「持仓峰值回撤 8%」移动止损；不预测顶，不因单日下跌砍掉全部仓位。",
        },
        "history": history,
        "validation": val,
    }

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out_path)),
                                   prefix=".risk_state.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser(
        "~/projects/hermes-dashboard/risk_state.json"))
    a = ap.parse_args()
    p = build(a.out)
    print(f"asof {p['asof']} · {p['state_label']}({p['state']}) · score {p['score']}/8")
    print(f"  初减 {p['action']['initial_trim']:.0%} · 移动止损 {p['action']['trail_stop']:.0%}")
    print(f"  下跌占比 {p['metrics']['down_ratio']:.1%} · MA20上方 {p['metrics']['above_ma20']:.1%} "
          f"· 60日 {p['metrics']['g60']:+.1%} · 10日冲击 {p['metrics']['shock10']}")
    print(f"  → {a.out}")


if __name__ == "__main__":
    main()
