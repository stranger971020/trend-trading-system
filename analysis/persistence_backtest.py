"""
板块持续性评分阈值回测模块（含行业宽度 + 资金流因子）
===========================================================
验证现有的 HIGH_PERSISTENCE (7.0) 和 MEDIUM_PERSISTENCE (5.0) 是否合理，
通过历史数据找到最优分档线，并测试新因子的预测能力。

新增因子：
  5. 行业宽度：站上MA20的个股比例（反映行业内部共识度）
  6. 资金流方向：主力净流入/流出（反映聪明钱态度）

用法：
  python3 analysis/persistence_backtest.py              # 完整回测
  python3 analysis/persistence_backtest.py --quick       # 快速（60 天）
  python3 analysis/persistence_backtest.py --thresholds-only
"""

import argparse
import csv as _csv
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DB_PATH,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_VOL,
    COL_NAME,
    COL_HIGH,
    COL_LOW,
    MOMENTUM_LOOKBACK,
    HIGH_PERSISTENCE as CURRENT_HIGH,
    MEDIUM_PERSISTENCE as CURRENT_MEDIUM,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)

BACKTEST_DAYS = 120
PREDICT_HORIZON = 5
MIN_HISTORY = 30

W_MOMENTUM = 0.30
W_SLOPE = 0.25
W_TURNOVER = 0.20
W_STRENGTH = 0.25


# ============================================================
# 数据加载
# ============================================================

def load_industry_data(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT trade_date, ts_code, name, open, high, low, close, "
        "vol, amount FROM sw_index_daily ORDER BY trade_date",
        conn,
    )
    conn.close()
    df[COL_TRADE_DATE] = df[COL_TRADE_DATE].astype(str)
    return df


# ============================================================
# 简化版持续性评分
# ============================================================

def compute_industry_persistence(df_hist, date):
    df = df_hist[df_hist[COL_TRADE_DATE] <= date].copy()
    all_dates = sorted(df[COL_TRADE_DATE].unique())
    if len(all_dates) < MIN_HISTORY:
        return {}
    start_date = all_dates[-min(len(all_dates), MOMENTUM_LOOKBACK + 10)]
    df = df[df[COL_TRADE_DATE] >= start_date]
    results = {}

    for ts_code in df[COL_TS_CODE].unique():
        grp = df[df[COL_TS_CODE] == ts_code].sort_values(COL_TRADE_DATE)
        if len(grp) < MOMENTUM_LOOKBACK + 1:
            continue
        closes = grp[COL_CLOSE].values
        volumes = grp[COL_VOL].values
        name = grp[COL_NAME].iloc[0]
        prices = grp[COL_CLOSE]
        ma20 = prices.iloc[-20:].mean()
        std20 = prices.iloc[-20:].std()
        current_close = closes[-1]
        bb_pos = 0.5
        if std20 > 0:
            bb_pos = (current_close - (ma20 - 2 * std20)) / (4 * std20)
            bb_pos = max(0, min(1, bb_pos))
        bb_score = bb_pos * 10
        mom_20d = (closes[-1] / closes[-(MOMENTUM_LOOKBACK + 1)] - 1) * 100
        mom_score = max(0, min(10, (mom_20d + 20) / 4))
        momentum_score = bb_score * 0.4 + mom_score * 0.6

        if len(closes) >= MOMENTUM_LOOKBACK:
            y = closes[-MOMENTUM_LOOKBACK:]
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        else:
            slope = 0
        slope_norm = max(0, min(10, (slope / (closes[-1] * 0.005) + 5)))
        slope_score = slope_norm

        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_vol
        turnover_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        if turnover_ratio < 0.5:
            to_score = 2
        elif turnover_ratio < 1.0:
            to_score = 5 + (turnover_ratio - 0.5) * 10
        elif turnover_ratio < 1.5:
            to_score = 10 - (turnover_ratio - 1.0) * 4
        elif turnover_ratio < 3:
            to_score = 8 - (turnover_ratio - 1.5) * 2
        else:
            to_score = max(0, 5 - (turnover_ratio - 3) * 2)
        turnover_score = max(0, min(10, to_score))

        # ---- 52周高点距离 ----
        lookback_52w = min(len(closes), 252)
        if lookback_52w >= 20:
            high_52w = np.max(closes[-lookback_52w:])
            pct_from_high = current_close / high_52w
            proximity_score = max(0, min(10, (pct_from_high - 0.5) * 20))
        else:
            proximity_score = 5.0

        # ---- 动量稳定性（20日回归R²） ----
        if len(closes) >= MOMENTUM_LOOKBACK:
            y = closes[-MOMENTUM_LOOKBACK:]
            x = np.arange(len(y))
            if np.std(y) > 0 and len(y) > 1:
                r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
            else:
                r2 = 0
            stability_score = r2 * 10
        else:
            stability_score = 5.0

        # ---- 波动率变化 ----
        if len(closes) >= 60:
            rets = np.diff(closes) / closes[:-1]
            vol_20d = np.std(rets[-20:]) if len(rets) >= 20 else 0.01
            vol_60d = np.std(rets[-60:]) if len(rets) >= 60 else vol_20d
            vr = vol_20d / vol_60d if vol_60d > 0 else 1.0
            if vr < 0.5: vcs = 3
            elif vr < 1.0: vcs = 5 + (vr - 0.5) * 8
            elif vr < 1.5: vcs = 9 - (vr - 1.0) * 4
            elif vr < 2.5: vcs = 7 - (vr - 1.5) * 3
            else: vcs = max(1, 4 - (vr - 2.5) * 2)
            vol_change_score = max(0, min(10, vcs))
        else:
            vol_change_score = 5.0

        results[ts_code] = {
            "name": name,
            "momentum_score": round(momentum_score, 2),
            "slope_score": round(slope_score, 2),
            "turnover_score": round(turnover_score, 2),
            "relative_strength": 5.0,
            "proximity_score": round(proximity_score, 2),
            "stability_score": round(stability_score, 2),
            "vol_change_score": round(vol_change_score, 2),
            "mom_20d": round(mom_20d, 2),
        }

    if not results:
        return {}
    slopes = np.array([r["slope_score"] for r in results.values()])
    avg_slope = np.mean(slopes) if len(slopes) > 0 else 1
    for ts_code, r in results.items():
        rel = r["slope_score"] / (avg_slope + 0.01)
        r["relative_strength"] = max(0, min(10, rel * 5))

    for ts_code, r in results.items():
        r["score"] = round(
            r["momentum_score"] * W_MOMENTUM
            + r["slope_score"] * W_SLOPE
            + r["turnover_score"] * W_TURNOVER
            + r["relative_strength"] * W_STRENGTH,
            2,
        )
    return results


def get_forward_return(df, ts_code, date, horizon=PREDICT_HORIZON):
    grp = df[(df[COL_TS_CODE] == ts_code) & (df[COL_TRADE_DATE] >= date)].sort_values(COL_TRADE_DATE)
    dates = grp[COL_TRADE_DATE].tolist()
    try:
        idx = dates.index(date)
    except ValueError:
        return np.nan
    if idx + horizon >= len(dates):
        return np.nan
    c_now = grp[COL_CLOSE].iloc[idx]
    c_fut = grp[COL_CLOSE].iloc[idx + horizon]
    if c_now <= 0:
        return np.nan
    return (c_fut / c_now - 1) * 100


# ============================================================
# 回测主逻辑
# ============================================================

def run_persistence_backtest(db_path=DB_PATH, lookback_days=BACKTEST_DAYS, verbose=True):
    t0 = time.time()

    df = load_industry_data(db_path)
    all_dates = sorted(df[COL_TRADE_DATE].unique())
    logger.info("加载 %d 个行业, %d 个交易日", df[COL_TS_CODE].nunique(), len(all_dates))

    # ---- 个股数据（行业宽度） ----
    logger.info("加载个股数据...")
    mapping_path = os.path.join(os.path.dirname(DB_PATH), "stock_industry_mapping.csv")
    stock_to_l1 = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                l1c = row.get("l1_code", "")
                if l1c:
                    stock_to_l1[row["ts_code"]] = l1c

    sconn = sqlite3.connect(db_path)
    stock_df = pd.read_sql_query(
        "SELECT trade_date, ts_code, close FROM stock_daily ORDER BY trade_date", sconn)
    stock_df[COL_TRADE_DATE] = stock_df[COL_TRADE_DATE].astype(str)
    stock_df["ma20"] = stock_df.groupby(COL_TS_CODE)[COL_CLOSE].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    stock_df["above_ma20"] = stock_df[COL_CLOSE] > stock_df["ma20"]
    stock_df["_l1"] = stock_df[COL_TS_CODE].map(stock_to_l1)
    stock_df = stock_df.dropna(subset=["_l1"])
    logger.info("个股: %d 行, %d 只", len(stock_df), stock_df[COL_TS_CODE].nunique())

    # ---- 资金流 ----
    mf_df = pd.read_sql_query(
        "SELECT trade_date, ts_code, net_mf_amount FROM moneyflow_cache ORDER BY trade_date", sconn)
    sconn.close()
    mf_df[COL_TRADE_DATE] = mf_df[COL_TRADE_DATE].astype(str)
    mf_df["_l1"] = mf_df[COL_TS_CODE].map(stock_to_l1)
    mf_df = mf_df.dropna(subset=["_l1"])
    logger.info("资金流: %d 行", len(mf_df))

    # ---- 市场指数 ----
    logger.info("预计算市场状态...")
    market_idx = df.groupby(COL_TRADE_DATE)[COL_CLOSE].mean()
    market_df = market_idx.to_frame(name="market_close")
    market_df["ma50"] = market_df["market_close"].rolling(50, min_periods=30).mean()
    market_df["ma200"] = market_df["market_close"].rolling(200, min_periods=120).mean()
    market_df["_roc"] = market_df["market_close"].pct_change()
    market_df["_up"] = (market_df["_roc"] > 0).astype(float)
    market_df["_up_ema"] = market_df["_up"].ewm(span=14, adjust=False).mean()
    market_df["adx"] = market_df["_up_ema"].ewm(span=14, adjust=False).mean() * 100

    def _get_regime(date_str):
        try:
            loc = market_df.index.get_loc(date_str)
            row = market_df.iloc[loc]
            above = row["market_close"] > row["ma200"] if pd.notna(row["ma200"]) else True
            strong = row["adx"] >= 25 if pd.notna(row["adx"]) else False
            if strong and above: return "BULL"
            if strong and not above: return "BEAR"
            return "RANGE"
        except: return "UNKNOWN"

    if len(all_dates) < lookback_days + MIN_HISTORY:
        lookback_days = len(all_dates) - MIN_HISTORY - 1
    test_end = len(all_dates) - PREDICT_HORIZON
    test_start = max(MIN_HISTORY, test_end - lookback_days)
    test_dates = all_dates[test_start:test_end]

    regime_cache = {}
    for date in test_dates:
        regime_cache[date] = _get_regime(date) if date in market_df.index else "UNKNOWN"
    rc = pd.Series(regime_cache).value_counts()
    logger.info("Regime: %s", {k: int(v) for k, v in rc.items()})
    logger.info("回测: %s -> %s (%d 天)", test_dates[0], test_dates[-1], len(test_dates))

    records = []

    for i, date in enumerate(test_dates):
        scores = compute_industry_persistence(df, date)
        if not scores:
            continue
        regime = regime_cache.get(date, "UNKNOWN")

        # 行业宽度
        sd = stock_df[stock_df[COL_TRADE_DATE] == date]
        bmap = {}
        if not sd.empty:
            for l1c, grp in sd.groupby("_l1"):
                if len(grp) >= 3:
                    bmap[l1c] = grp["above_ma20"].mean() * 10

        # 资金流
        mfd = mf_df[mf_df[COL_TRADE_DATE] == date]
        mmap = {}
        if not mfd.empty:
            for l1c, grp in mfd.groupby("_l1"):
                net = grp["net_mf_amount"].sum()
                mmap[l1c] = (np.tanh(net / 1e8) + 1) * 5

        for ts_code, r in scores.items():
            fwd = get_forward_return(df, ts_code, date, PREDICT_HORIZON)
            if np.isnan(fwd):
                continue
            records.append({
                "date": date, "ts_code": ts_code, "name": r["name"],
                "score": r["score"], "forward_return": round(fwd, 2),
                "mom_20d": r["mom_20d"], "regime": regime,
                "high": int(r["score"] >= CURRENT_HIGH),
                "medium": int(CURRENT_MEDIUM <= r["score"] < CURRENT_HIGH),
                "low": int(r["score"] < CURRENT_MEDIUM),
                "mom_score": r.get("momentum_score", 0),
                "slope_score": r.get("slope_score", 0),
                "turnover_score": r.get("turnover_score", 0),
                "strength_score": r.get("relative_strength", 0),
                "breadth_score": round(bmap.get(ts_code, 5.0), 2),
                "mf_score": round(mmap.get(ts_code, 5.0), 2),
                "proximity_score": r.get("proximity_score", 5.0),
                "stability_score": r.get("stability_score", 5.0),
                "vol_change_score": r.get("vol_change_score", 5.0),
            })

        if verbose and (i + 1) % 30 == 0:
            logger.info("进度: %d / %d", i + 1, len(test_dates))

    elapsed = time.time() - t0
    logger.info("完成: %d 记录, %.1f 秒", len(records), elapsed)
    if not records:
        return {"status": "failed", "error": "无数据"}

    pred_df = pd.DataFrame(records)

    # ---- 分桶 ----
    pred_df["score_bucket"] = pd.cut(pred_df["score"], bins=20, precision=1)
    bucket_stats = []
    for label, grp in pred_df.groupby("score_bucket", observed=False):
        fwd = grp["forward_return"].values
        bucket_stats.append({
            "bucket": str(label), "mid": (label.left + label.right) / 2,
            "count": len(grp), "avg_return": round(float(np.mean(fwd)), 2),
            "median_return": round(float(np.median(fwd)), 2),
            "win_rate": round(float(np.mean(fwd > 0) * 100), 1),
            "win_rate_gt5": round(float(np.mean(fwd > 5) * 100), 1),
            "win_rate_gt10": round(float(np.mean(fwd > 10) * 100), 1),
        })

    # ---- 当前阈值 ----
    current_groups = {}
    for label, mask in [
        ("高持续性 (>=7.0)", pred_df["score"] >= CURRENT_HIGH),
        ("中等持续性 (5.0~7.0)", (pred_df["score"] >= CURRENT_MEDIUM) & (pred_df["score"] < CURRENT_HIGH)),
        ("低持续性 (<5.0)", pred_df["score"] < CURRENT_MEDIUM),
    ]:
        grp = pred_df[mask]
        if len(grp) > 0:
            fwd = grp["forward_return"].values
            current_groups[label] = {
                "count": len(grp), "avg_return": round(float(np.mean(fwd)), 2),
                "median_return": round(float(np.median(fwd)), 2),
                "win_rate": round(float(np.mean(fwd > 0) * 100), 1),
                "win_rate_gt5": round(float(np.mean(fwd > 5) * 100), 1),
                "win_rate_gt10": round(float(np.mean(fwd > 10) * 100), 1),
            }

    # ---- Regime 分组 ----
    regime_groups = {}
    for rl in ["BULL", "BEAR", "RANGE"]:
        grp = pred_df[pred_df["regime"] == rl]
        if len(grp) < 20:
            continue
        subs = {}
        for sl, sm in [
            ("高持续性 (>=7.0)", grp["score"] >= CURRENT_HIGH),
            ("中等持续性 (5.0~7.0)", (grp["score"] >= CURRENT_MEDIUM) & (grp["score"] < CURRENT_HIGH)),
            ("低持续性 (<5.0)", grp["score"] < CURRENT_MEDIUM),
        ]:
            sg = grp[sm]
            if len(sg) > 0:
                sf = sg["forward_return"].values
                subs[sl] = {"count": len(sg), "avg_return": round(float(np.mean(sf)), 2),
                            "win_rate": round(float(np.mean(sf > 0) * 100), 1)}
        regime_groups[rl] = subs

    # ---- Regime 最优阈值 ----
    regime_best = {}
    for rl in ["BULL", "BEAR", "RANGE"]:
        grp = pred_df[pred_df["regime"] == rl]
        if len(grp) < 30:
            continue
        best_h, best_m, best_sep = CURRENT_HIGH, CURRENT_MEDIUM, 0
        for hc in np.arange(6.0, 9.5, 0.5):
            for mc in np.arange(3.0, hc - 0.5, 0.5):
                hg = grp[grp["score"] >= hc]
                lg = grp[grp["score"] < mc]
                if len(hg) < 5 or len(lg) < 10:
                    continue
                sep = hg["forward_return"].mean() - lg["forward_return"].mean()
                if sep > best_sep:
                    best_sep, best_h, best_m = sep, hc, mc
        if best_sep > 0:
            regime_best[rl] = {"high": round(best_h, 1), "medium": round(best_m, 1),
                               "separation": round(best_sep, 2)}

    # ---- 6因子网格搜索 ----
    logger.info("网格搜索因子权重...")

    def _eval_6f(w, sub_df, regime_filter=None):
        df2 = sub_df.copy()
        if regime_filter is not None:
            df2 = df2[df2["regime"] == regime_filter]
            if len(df2) < 50:
                return None
        df2["_cs"] = (w[0] * df2["mom_score"] + w[1] * df2["slope_score"]
                      + w[2] * df2["turnover_score"] + w[3] * df2["strength_score"]
                      + w[4] * df2["breadth_score"] + w[5] * df2["mf_score"]
                      + w[6] * df2["proximity_score"] + w[7] * df2["stability_score"]
                      + w[8] * df2["vol_change_score"])
        df2["_rk"] = df2.groupby("date")["_cs"].rank(ascending=False, pct=True)
        top = df2[df2["_rk"] <= 0.25]
        bot = df2[df2["_rk"] >= 0.75]
        if len(top) < 20 or len(bot) < 20:
            return None
        return {"top_avg": round(float(top["forward_return"].mean()), 2),
                "bot_avg": round(float(bot["forward_return"].mean()), 2),
                "separation": round(float(top["forward_return"].mean() - bot["forward_return"].mean()), 2),
                "top_win": round(float((top["forward_return"] > 0).mean() * 100), 1)}

    # 单因子区分力
    factor_names = ["mom_score", "slope_score", "turnover_score", "strength_score",
                     "breadth_score", "mf_score", "proximity_score", "stability_score", "vol_change_score"]
    factor_labels = ["动量", "斜率", "换手", "强度",
                     "行业宽度", "资金流", "52周高点", "动量稳定性", "波动率变化"]
    single_factors = {}
    for fn, fl in zip(factor_names, factor_labels):
        w9 = [0] * 9
        w9[factor_names.index(fn)] = 1
        res = _eval_6f(w9, pred_df, None)
        if res:
            single_factors[fl] = res["separation"]

    grid_results = {}
    for regime_key, regime_filter in [("ALL", None), ("BULL", "BULL")]:
        orig_w = [W_MOMENTUM, W_SLOPE, W_TURNOVER, W_STRENGTH, 0, 0, 0, 0, 0]
        blended_w = [0.10, 0.10, 0.10, 0.10, 0.25, 0, 0.15, 0.10, 0.10]
        entry = {"current": None, "blended": None, "best": None, "singles": single_factors}
        cur = _eval_6f(orig_w, pred_df, regime_filter)
        if cur:
            entry["current"] = {"weights": [round(x, 2) for x in orig_w], **cur}
        bl = _eval_6f(blended_w, pred_df, regime_filter)
        if bl:
            entry["blended"] = {"weights": [round(x, 2) for x in blended_w], **bl}
        best_sep, best_w = -999, orig_w[:]
        for w1 in np.arange(0, 1.01, 0.2):
            for w2 in np.arange(0, 1.01, 0.2):
                if w1 + w2 > 1: continue
                for w3 in np.arange(0, 1.01, 0.2):
                    if w1 + w2 + w3 > 1: continue
                    for w4 in np.arange(0, 1.01, 0.2):
                        if w1 + w2 + w3 + w4 > 1: continue
                        for w5 in np.arange(0, 1.01, 0.2):
                            if w1 + w2 + w3 + w4 + w5 > 1: continue
                            w6 = round(1 - w1 - w2 - w3 - w4 - w5, 1)
                            if w6 < 0: continue
                            nz = sum(1 for w in [w1,w2,w3,w4,w5,w6] if w >= 0.1)
                            if nz < 2: continue
                            w_all = [w1,w2,w3,w4,w5,w6,0,0,0]
                            res = _eval_6f(w_all, pred_df, regime_filter)
                            if res and res["separation"] > best_sep:
                                best_sep, best_w = res["separation"], w_all[:]
        if best_sep > -999:
            entry["best"] = {"weights": [round(x, 1) for x in best_w], "separation": round(best_sep, 2)}
        grid_results[regime_key] = entry

    logger.info("网格搜索完成")

    # ---- 全样本最优阈值 ----
    best_high, best_medium, best_separation = CURRENT_HIGH, CURRENT_MEDIUM, 0
    for hc in np.arange(6.0, 9.5, 0.5):
        for mc in np.arange(3.0, hc - 0.5, 0.5):
            hg = pred_df[pred_df["score"] >= hc]
            lg = pred_df[pred_df["score"] < mc]
            if len(hg) < 10 or len(lg) < 30:
                continue
            sep = hg["forward_return"].mean() - lg["forward_return"].mean()
            if sep > best_separation:
                best_separation = sep
                best_high, best_medium = hc, mc
    cur_sep = (pred_df[pred_df["score"] >= CURRENT_HIGH]["forward_return"].mean()
               - pred_df[pred_df["score"] < CURRENT_MEDIUM]["forward_return"].mean())

    return {
        "status": "success",
        "metadata": {"backtest_period": f"{test_dates[0]} -> {test_dates[-1]}",
                     "trading_days": len(test_dates), "total_records": len(records),
                     "regime_distribution": {k: int(v) for k, v in rc.items()},
                     "computation_time_sec": round(elapsed, 2)},
        "current_thresholds": {"high": CURRENT_HIGH, "medium": CURRENT_MEDIUM},
        "current_separation": round(cur_sep, 2),
        "recommended_thresholds": {"high": round(best_high, 1), "medium": round(best_medium, 1),
                                    "separation": round(best_separation, 2)},
        "regime_thresholds": regime_best,
        "current_performance": current_groups,
        "regime_performance": regime_groups,
        "bucket_analysis": sorted(bucket_stats, key=lambda x: x["mid"]),
        "weight_grid": grid_results,
        "_raw_data": pred_df,
    }


# ============================================================
# 打印
# ============================================================

def print_report(report):
    if report.get("status") != "success":
        print(f"\n❌ 失败: {report.get('error', '')}")
        return
    meta = report["metadata"]
    cur = report["current_thresholds"]
    rec = report["recommended_thresholds"]
    bands = report["current_performance"]
    print("\n" + "=" * 68)
    print("  板块持续性评分 · 6因子回测报告")
    print("=" * 68)
    print(f"  区间: {meta['backtest_period']}  |  {meta['trading_days']}天  |  {meta['total_records']}条")
    print(f"  耗时: {meta['computation_time_sec']}秒")
    print("-" * 68)
    print(f"  当前阈值表现")
    for label, stats in bands.items():
        print(f"  {label:<20} {stats['count']:>5}条  {stats['avg_return']:>+7.2f}% 胜率{stats['win_rate']:>5.1f}%")
    print(f"  区分度: {report['current_separation']:+.2f}% (当前) → {rec['separation']:+.2f}% (推荐)")
    print(f"  推荐阈值: HIGH {cur['high']:.0f}→{rec['high']:.1f}  MEDIUM {cur['medium']:.0f}→{rec['medium']:.1f}")

    # Regime
    rg = report.get("regime_performance", {})
    if rg:
        print("-" * 68)
        print(f"  按市场状态")
        for rl in ["BULL", "BEAR", "RANGE"]:
            subs = rg.get(rl, {})
            if not subs:
                continue
            for sl, stats in subs.items():
                print(f"  {rl:>6} {sl:<20} {stats['count']:>4}条 {stats['avg_return']:>+7.2f}% 胜率{stats['win_rate']:>5.1f}%")

    # Regime 最佳阈值
    rt = report.get("regime_thresholds", {})
    if rt:
        print("-" * 68)
        print(f"  Regime最优阈值")
        for rl in ["BULL", "BEAR", "RANGE"]:
            info = rt.get(rl)
            if info:
                print(f"  {rl:>6}: HIGH>={info['high']:.1f} MEDIUM>={info['medium']:.1f} 区分度{info['separation']:+.2f}%")

    # 因子权重
    wg = report.get("weight_grid", {})
    if wg:
        print("-" * 68)
        print(f"  6因子分析")
        # 单因子
        singles = {}
        for rk in ["ALL", "BULL"]:
            ent = wg.get(rk, {})
            if ent.get("singles"):
                singles = ent["singles"]
        if singles:
            print(f"  单因子区分力:")
            for fname, sep in sorted(singles.items(), key=lambda x: -x[1]):
                print(f"    {fname}: {sep:+.2f}%")

        for rk in ["ALL", "BULL"]:
            label = "全市场" if rk == "ALL" else "BULL市场"
            ent = wg.get(rk, {})
            if not ent:
                continue
            print(f"  {label}:")
            cur_e = ent.get("current")
            if cur_e:
                w = cur_e["weights"]
                print(f"    当前(4因子): {cur_e['separation']:+.2f}%")
            bl_e = ent.get("blended")
            if bl_e:
                w = bl_e["weights"]
                print(f"    混合(9因子 含宽度+52周+稳定性): → {bl_e['separation']:+.2f}%")
            best_e = ent.get("best")
            if best_e:
                w = best_e["weights"]
                print(f"    最优: 动量{w[0]*100:.0f} 斜率{w[1]*100:.0f} 换手{w[2]*100:.0f} 强度{w[3]*100:.0f}")
                print(f"          宽度{w[4]*100:.0f} 资金流{w[5]*100:.0f} 52周{w[6]*100:.0f} 稳定性{w[7]*100:.0f} 波动率{w[8]*100:.0f}")
                print(f"    → 区分度 {best_e['separation']:+.2f}%")

    print("-" * 68)
    print("  分数分桶")
    print(f"  {'范围':>10} | {'量':>4} | {'平均收益':>8} | {'胜率':>5} | {'>5%':>5} | {'>10%':>5}")
    for b in report.get("bucket_analysis", []):
        if b["count"] < 5: continue
        print(f"  {b['bucket']:>10} | {b['count']:>4} | {b['avg_return']:>+7.2f}% | {b['win_rate']:>4.0f}% | {b['win_rate_gt5']:>4.0f}% | {b['win_rate_gt10']:>4.0f}%")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--thresholds-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    lookback = args.days or (60 if args.quick else BACKTEST_DAYS)
    report = run_persistence_backtest(lookback_days=lookback)
    if args.thresholds_only:
        if report.get("status") == "success":
            rec = report["recommended_thresholds"]
            print(f"HIGH: {rec['high']:.1f}  MEDIUM: {rec['medium']:.1f}  区分度: {rec['separation']:+.2f}%")
        return
    print_report(report)
    raw = report.get("_raw_data")
    if raw is not None and not raw.empty:
        raw.to_csv(f"persistence_backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)


if __name__ == "__main__":
    main()
