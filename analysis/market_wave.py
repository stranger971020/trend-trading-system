# -*- coding: utf-8 -*-
"""
主升浪识别器（波段/趋势交易用）

背景: 用户交易风格 = 一年抓 2-3 次主升浪、每次目标 30%+、重仓参与、偏好左侧。
      与 V6 日报的 T+1/D+5 短周期选股尺度完全不同 → 本模块服务"阶段择时"。

方法: ZigZag 摆动点识别（对收盘价序列, 反向回撤 ≥ zigzag_pct 确认一次转折）,
      相邻 低→高 腿幅度 ≥ min_amp 者判为"主升浪"。
      输出历史主升浪清单/频率/幅度分布 + 当前状态。

数据: data_storage/sw_index_data.db -> sw_index_daily (指数) / stock_daily (个股)

Changelog
─────────
2026-09-25 Hermes: 新建。
"""

import os
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")

# 默认参数（可由回测/用户风格调整）
ZIGZAG_PCT = 0.15      # 反向回撤阈值: 确认一次摆动转折
MIN_AMP = 0.20         # 腿幅度 ≥ 此值 → 记为主升浪（指数级）
INDEX = "000300.SH"    # 主基准: 沪深300


def load_index(code=INDEX, db_path=STOCK_DB, start="20050101", end="20991231"):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, close, high, low FROM sw_index_daily "
            "WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            conn, params=[code, start, end])
    finally:
        conn.close()
    return df


def zigzag(close, pct=ZIGZAG_PCT):
    """标准 ZigZag 摆动点。返回 list[(pos, price, 'H'|'L')]。

    状态机: 上行时追踪新高, 回撤 > pct 确认高点并转下行; 下行时追踪新低,
    反弹 > pct 确认低点并转上行。首段方向默认按"上行"起步。
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    if n < 2:
        return []
    pivots = []
    trend = 1                       # 默认先按上行处理
    ext_i, ext_p = 0, c[0]
    for i in range(1, n):
        if trend == 1:
            if c[i] > ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] <= ext_p * (1 - pct):
                pivots.append((ext_i, ext_p, "H"))
                trend = -1
                ext_i, ext_p = i, c[i]
        else:
            if c[i] < ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] >= ext_p * (1 + pct):
                pivots.append((ext_i, ext_p, "L"))
                trend = 1
                ext_i, ext_p = i, c[i]
    return pivots


def waves(df, pct=ZIGZAG_PCT, min_amp=MIN_AMP):
    """从 ZigZag 摆动点提取"腿"，标记主升浪。"""
    piv = zigzag(df["close"].to_numpy(), pct)
    dates = df["trade_date"].tolist()
    rows = []
    for a, b in zip(piv, piv[1:]):
        (i0, p0, t0), (i1, p1, t1) = a, b
        if t0 == "L" and t1 == "H":
            amp = p1 / p0 - 1
            rows.append({"start": dates[i0], "end": dates[i1],
                         "start_px": round(p0, 2), "end_px": round(p1, 2),
                         "amp": amp, "days": i1 - i0,
                         "calendar_days": (pd.Timestamp(dates[i1]) - pd.Timestamp(dates[i0])).days,
                         "is_main": amp >= min_amp})
    return pd.DataFrame(rows), piv


def stats(code=INDEX, pct=ZIGZAG_PCT, min_amp=MIN_AMP, start="20050101", end="20991231"):
    df = load_index(code, start=start, end=end)
    if df.empty:
        return None
    w, piv = waves(df, pct, min_amp)
    if w.empty:
        return None
    mains = w[w["is_main"]].copy()
    yrs = (pd.Timestamp(df["trade_date"].iloc[-1]) - pd.Timestamp(df["trade_date"].iloc[0])).days / 365.25
    # 当前状态
    last = piv[-1] if piv else None
    cur_px = df["close"].iloc[-1]
    cur_date = df["trade_date"].iloc[-1]
    state = {}
    if last:
        li, lp, lt = last
        state = {"last_pivot_type": lt, "last_pivot_date": df["trade_date"].iloc[li],
                 "last_pivot_px": round(lp, 2),
                 "since_pivot": round(cur_px / lp - 1, 4),
                 "underwater_from_last_high": None}
    # 距最近高点回撤
    peak = df["close"].cummax()
    dd = (df["close"].iloc[-1] / peak.iloc[-1] - 1)
    return {
        "code": code, "range": f"{df['trade_date'].iloc[0]}~{cur_date}", "years": round(yrs, 1),
        "n_legs": len(w), "n_main": len(mains),
        "mains_per_year": round(len(mains) / yrs, 2),
        "amp_stats": {
            "mean": round(mains["amp"].mean(), 3) if len(mains) else None,
            "median": round(mains["amp"].median(), 3) if len(mains) else None,
            "min": round(mains["amp"].min(), 3) if len(mains) else None,
            "max": round(mains["amp"].max(), 3) if len(mains) else None,
        },
        "days_stats": {
            "median_trading_days": int(mains["days"].median()) if len(mains) else None,
            "median_calendar_days": int(mains["calendar_days"].median()) if len(mains) else None,
        },
        "recent_mains": mains.tail(8)[["start", "end", "amp", "days", "calendar_days"]].to_dict("records"),
        "current": {**state, "date": cur_date, "close": round(float(cur_px), 2),
                    "drawdown_from_peak": round(float(dd), 4)},
        "all_waves": w,
    }


def wave_state(code=INDEX, pct=ZIGZAG_PCT, min_amp=MIN_AMP):
    """紧凑的"主升浪状态"判定, 供日报/雷达调用。

    逻辑: 最近已确认摆动点若为低点 L, 则"当前=该低点以来的上升腿":
      - 腿内高点回撤 ≥10% 且腿幅≥min_amp → 主升浪中·回调
      - 腿幅 ≥ min_amp                      → 主升浪进行中
      - 5% ~ min_amp                        → 主升浪初期/待确认
      - < 5%                                → 底部区域(未确认)
    最近摆动点为高点 H → 回调中(等回撤到位)。回撤一律相对"本轮腿内高点", 非历史最高。
    """
    df = load_index(code)
    if df.empty:
        return None
    close = df["close"].to_numpy()
    dates = df["trade_date"].tolist()
    piv = zigzag(close, pct)
    if not piv:
        return None
    li, lp, lt = piv[-1]
    cur, curd = float(close[-1]), dates[-1]
    since = cur / lp - 1.0
    leg_peak = float(close[li:].max())          # 本轮腿内高点(自最后摆动点起)
    dd_leg = cur / leg_peak - 1.0
    if lt == "L":
        if since >= min_amp and dd_leg <= -0.10:
            status, tone = "主升浪中·回调", "pullback"
        elif since >= min_amp:
            status, tone = "主升浪进行中", "up"
        elif since > 0.05:
            status, tone = "主升浪初期/待确认", "early"
        else:
            status, tone = "底部区域(未确认)", "bottom"
    else:
        status, tone = "回调中", "down"
    return {"code": code, "date": curd, "close": round(cur, 2),
            "last_pivot": lt, "last_pivot_date": dates[li], "last_pivot_px": round(float(lp), 2),
            "since_pivot": round(since, 4), "leg_peak": round(leg_peak, 2),
            "dd_from_leg_peak": round(dd_leg, 4),
            "status": status, "tone": tone,
            "leg_trading_days": len(close) - 1 - li}


PRIMARY_CODES = [("000300.SH", "沪深300"), ("000001.SH", "上证"), ("399006.SZ", "创业板")]


def wave_radar(codes=PRIMARY_CODES, min_amp=MIN_AMP):
    """主升浪雷达: 多指数当前状态 + 历史频率/幅度/持续参考。"""
    out = {"asof": None, "states": [], "stats": {}}
    for code, name in codes:
        st = wave_state(code, min_amp=min_amp)
        if not st:
            continue
        st["name"] = name
        out["states"].append(st)
        out["asof"] = st["date"]
    for code, name in codes:
        s = stats(code, min_amp=min_amp)
        if s:
            out["stats"][name] = {
                "mains_per_year": s["mains_per_year"], "n_main": s["n_main"],
                "amp_median": s["amp_stats"]["median"],
                "days_median": s["days_stats"]["median_calendar_days"],
            }
    return out


if __name__ == "__main__":
    import sys
    for code in (sys.argv[1:] or ["000300.SH", "000001.SH", "399006.SZ"]):
        s = stats(code)
        if not s:
            print(f"{code}: 无数据"); continue
        print(f"\n===== {code} ({s['range']}, {s['years']}年) =====")
        print(f"主升浪(≥{int(MIN_AMP*100)}%): {s['n_main']} 次 · {s['mains_per_year']} 次/年")
        print(f"幅度 mean/median/min/max: {s['amp_stats']}")
        print(f"持续 median: {s['days_stats']['median_trading_days']} 交易日 / {s['days_stats']['median_calendar_days']} 自然日")
        print("近 8 次主升浪:")
        for m in s["recent_mains"]:
            print(f"  {m['start']}→{m['end']}  {m['amp']*100:+.1f}%  {m['days']}日/{m['calendar_days']}天")
        c = s["current"]
        print(f"当前: {c['date']} 收 {c['close']} · 最近摆动={c.get('last_pivot_type')}@{c.get('last_pivot_date')} "
              f"({c.get('since_pivot')}) · 距峰值 {c['drawdown_from_peak']*100:+.1f}%")
