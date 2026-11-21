"""每日风险状态 · 大盘脆弱度 + 持仓动作建议

2026-11-21 Claude。与回测同口径（PIT）：
  - 市场脆弱度：趋势/广度/冲击/拥挤/波动 五维 0~8 分
  - 状态：normal / warning / fragile / crash
  - 动作：初减 0 / 0 / 25% / 40%；弱市+第二次冲击 → 50%
  - 剩余仓位：峰值回撤 8% 移动止损（不预测顶，不因单日下跌砍掉全部仓位）

2026-11-21 接入 V6：新增 --asof / --expected-date / --stale-days / --mirror，
  输出 freshness 与「前日是否已初减」，供 V6 日报和看板共用。

用法：
  python3 analysis/risk_state.py --asof 20261119 \
      --out data_storage/risk_state.json \
      --mirror ~/projects/hermes-dashboard/risk_state.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backtest"))
import rb_data  # noqa: E402

SCHEMA = 1
TRAIL = 0.08
LABEL = {"normal": "正常", "warning": "警戒", "fragile": "脆弱", "crash": "崩跌"}
ACTION = {"normal": 0.0, "warning": 0.0, "fragile": 0.25, "crash": 0.40}
KNOWN_STATES = tuple(LABEL)


def latest_db_date() -> str:
    c = sqlite3.connect(rb_data.DB)
    d = c.execute("SELECT MAX(trade_date) FROM stock_daily").fetchone()[0]
    c.close()
    return d


def _trade_open_days(start: str, end: str) -> list[str] | None:
    """SSE 开市日；失败返回 None（不抛）。"""
    try:
        import tushare as ts
        from config import TUSHARE_TOKEN
        pro = ts.pro_api(TUSHARE_TOKEN)
        df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        return sorted(df[df["is_open"] == 1]["cal_date"].astype(str).tolist())
    except Exception:  # noqa: BLE001
        return None


def expected_trade_date(now: datetime | None = None) -> str | None:
    """02:45 运行时期望的「最新已完成交易日」= 上一交易日；非交易日=最近交易日。"""
    now = now or datetime.now()
    today = now.strftime("%Y%m%d")
    start = (now - timedelta(days=20)).strftime("%Y%m%d")
    days = _trade_open_days(start, today)
    if not days:
        return None
    latest = days[-1]
    now_min = now.hour * 60 + now.minute
    if latest == today and now_min < 22 * 60 + 30:
        return days[-2] if len(days) >= 2 else latest
    return latest


def count_trade_days(start_exclusive: str, end_inclusive: str) -> int | None:
    start = (datetime.strptime(start_exclusive, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    days = _trade_open_days(start, end_inclusive)
    if days is None:
        return None
    return len([d for d in days if d > str(start_exclusive) and d <= str(end_inclusive)])


def load_fresh(asof: str | None = None) -> dict:
    latest = latest_db_date()
    p = os.path.join(rb_data.CACHE, "rb_market.parquet")
    stale = True
    if os.path.exists(p):
        try:
            stale = pd.read_parquet(p).index.max() != latest
        except Exception:
            stale = True
    return rb_data.load(force=stale)


def _prev_trim(m: pd.DataFrame, asof: str) -> tuple[str | None, float]:
    """前一个交易日的状态与其建议初减（用于避免连续两日重复减仓）。"""
    try:
        idx = list(m.index).index(str(asof))
    except ValueError:
        return None, 0.0
    if idx < 1:
        return None, 0.0
    pr = m.iloc[idx - 1]
    ps = str(pr["frag_state"]) if pd.notna(pr["frag_state"]) else None
    if ps not in ACTION:
        return ps, 0.0
    pt = ACTION[ps]
    if bool(pr["g60"] < -0.05) and int(pr["shock10"]) >= 2:
        pt = max(pt, 0.50)
    return ps, min(pt, 0.50)


def _atomic_write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=".risk_state.", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build(out_path: str | None = None, asof: str | None = None,
          expected_date: str | None = None, stale_days: int | None = None,
          mirror_path: str | None = None) -> dict:
    d = load_fresh(asof)
    m = d["market"]
    if asof:
        m = m.loc[m.index <= str(asof)]
    if m.empty:
        raise RuntimeError(f"risk_state: 无可用市场数据（asof={asof}）")
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

    prev_state, prev_trim = _prev_trim(m, asof)
    already = prev_trim + 1e-9 >= trim and trim > 0
    incremental = max(0.0, trim - prev_trim) if trim > prev_trim else 0.0

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

    expected = expected_date or expected_trade_date()
    if stale_days is None:
        if expected and str(expected) > str(asof):
            n = count_trade_days(str(asof), str(expected))
            stale_days = 1 if n is None else int(n)
        else:
            stale_days = 0
    stale_days = int(stale_days or 0)
    fresh_ok = ((expected is None or str(expected) == str(asof)) and stale_days == 0)

    if already:
        trim_text = (f"前日建议初减 {prev_trim:.0%}；若已执行，今日不重复；"
                     f"若未执行，今日补执行。剩余仓位执行 {TRAIL:.0%} 移动止损")
    elif incremental > 0 and prev_trim > 0:
        trim_text = (f"今日较前日新增初减 {incremental:.0%}（累计 {trim:.0%}，若尚未执行）；"
                     f"剩余仓位执行 {TRAIL:.0%} 移动止损")
    elif trim > 0:
        trim_text = (f"今日初减 {trim:.0%}（若尚未执行）；"
                     f"剩余仓位自持有期最高收盘价回撤 {TRAIL:.0%} 即减/退")
    else:
        trim_text = f"不初减；持仓沿用「浮盈≥+10% 后回撤 {TRAIL:.0%}」"

    payload = {
        "schema": SCHEMA,
        "asof": asof,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_date": expected,
        "stale_days": stale_days,
        "freshness": {
            "asof": asof,
            "expected": expected,
            "stale_days": stale_days,
            "ok": fresh_ok,
            "source": "arg" if expected_date else ("trade_cal" if expected else "unknown"),
        },
        "entry_advice_ok": fresh_ok,
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
            "initial_trim_already_active": bool(already),
            "incremental_trim": incremental,
            "previous_state": prev_state,
            "previous_trim": prev_trim,
            "trim_text": trim_text,
            "trail_stop": TRAIL,
            "note": "初减后剩余仓位按「持仓峰值回撤 8%」移动止损；不预测顶，不因单日下跌砍掉全部仓位。",
        },
        "history": history,
        "validation": val,
    }

    if out_path:
        _atomic_write(out_path, payload)
    if mirror_path:
        _atomic_write(mirror_path, payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser(
        "~/projects/hermes-dashboard/risk_state.json"))
    ap.add_argument("--asof", default=None, help="基准交易日 YYYYMMDD（默认库内最新）")
    ap.add_argument("--expected-date", default=None, help="期望最新交易日 YYYYMMDD")
    ap.add_argument("--stale-days", type=int, default=None, help="数据滞后交易日数")
    ap.add_argument("--mirror", default=None, help="额外写一份镜像（看板兼容）")
    a = ap.parse_args()
    p = build(a.out, a.asof, a.expected_date, a.stale_days, a.mirror)
    print(f"asof {p['asof']} · {p['state_label']}({p['state']}) · score {p['score']}/8")
    print(f"  初减 {p['action']['initial_trim']:.0%} · {p['action']['trim_text']}")
    print(f"  下跌占比 {p['metrics']['down_ratio']:.1%} · MA20上方 {p['metrics']['above_ma20']:.1%} "
          f"· 60日 {p['metrics']['g60']:+.1%} · 10日冲击 {p['metrics']['shock10']}")
    f = p["freshness"]
    print(f"  freshness: expected {f['expected']} · stale_days {f['stale_days']} · ok {f['ok']}")
    print(f"  → {a.out}" + (f" / {a.mirror}" if a.mirror else ""))


if __name__ == "__main__":
    main()
