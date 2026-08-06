#!/usr/bin/env python3
"""V6 生产模拟回测（HERMES-20260802-001）——完全模拟真实环境。

生产逻辑：
  ① 首次训练：用截至 T0 的数据训练 3 引擎 Ordinal 模型 → 持久化 v6_{engine}_{T0}.pkl
  ② 每周重训：每 RETRAIN_INTERVAL 个交易日，用截至当日的全量数据重训 → 保存新版本
  ③ 逐日选股：每个交易日收盘，用当前活跃模型 + 截至当日特征打分 → route_portfolio 选 Top-N
  ④ 每日明细：覆盖 T0 之后全部交易日，含股票/分数/regime/次日与5日收益

模型持久化：每版保留、不覆盖（v6_{engine}_{YYYYMMDD}.pkl）。

用法:
  python3 backtest/v6_production_backtest.py
      --train-start 20240801 --train-end 20250731   # 首次训练窗
      --since 20250801                              # 明细覆盖起点（=T0后首个交易日）
      --out backtest/v6_prod_20260802
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, COL_TS_CODE, COL_TRADE_DATE

from analysis import strategy_feature_masker as sfm
from analysis import winrate_engine as wre
from analysis import t1_gap_risk as t1
from analysis import dynamic_allocation as da
from analysis.alpha_decay_risk import load_t1_daily, compute_volume_shock_batch
from analysis.retrain_ml import load_stock_daily, load_industry_data
from analysis.ml_model import build_full_feature_matrix

logger = logging.getLogger("v6_prod")

FORWARD_DAYS = 20
GATE_THRESHOLD = 0.55
RETRAIN_INTERVAL = 5  # 每周（5交易日）重训一次


def alloc_engine(regime: str) -> str:
    return "momentum" if regime == "bull" else "reversion"


def regime_upto(l1_daily: pd.DataFrame, upto_date: str) -> str:
    if l1_daily is None or l1_daily.empty:
        return "range"
    sub = l1_daily[l1_daily[COL_TRADE_DATE] <= upto_date]
    if sub.empty:
        return "range"
    try:
        from analysis.market_regime import determine_regime
        rr = determine_regime(sub)
        return da.normalize_regime(rr.get("regime"))
    except Exception:
        return "range"


def run_production_backtest(
    feat: pd.DataFrame,
    l1_daily: pd.DataFrame,
    train_start: str,
    train_end: str,
    since: str,
    top_n: int = 5,
    verbose: bool = False,
) -> dict:
    """生产模拟主循环。"""
    t_start = time.time()
    feat = feat.dropna(subset=[COL_TRADE_DATE]).sort_values([COL_TS_CODE, COL_TRADE_DATE])
    dates = sorted(feat[COL_TRADE_DATE].unique())
    if train_end not in dates:
        train_end = max(d for d in dates if d <= train_end)
    if since not in dates:
        since = min(d for d in dates if d >= since)

    # 预构建三引擎掩码特征（逐日切片复用）
    engines = {}
    for eng in sfm.ALL_ENGINES:
        masked = sfm.mask_feature_matrix(feat, eng, include_close=True)
        masked = wre.build_winrate_labels(masked, forward_days=FORWARD_DAYS)
        feats = [c for c in sfm.get_engine_features(eng) if c in masked.columns]
        engines[eng] = {"masked": masked, "feats": feats}

    # 价格面板
    price_wide = feat[[COL_TS_CODE, COL_TRADE_DATE, "close", "atr_pct"]].copy()
    meta_cols = ["mom5", "value_score", "earnings_yield", "quality_score", "vol_ratio"]

    daily_trades = []
    model_versions = []
    regime_counts = {"bull": 0, "range": 0, "bear": 0}
    n_folds = 0
    n_dates_scored = 0

    # 首次训练（截至 train_end）
    logger.info("① 首次训练（截至 %s）...", train_end)
    active_models = {}
    for eng, info in engines.items():
        masked = info["masked"]
        tr = masked[masked[COL_TRADE_DATE] <= train_end].dropna(subset=["y_label"])
        model, _ = wre.train_winrate_model(tr, feature_cols=info["feats"],
                                           forward_days=FORWARD_DAYS,
                                           engine=eng, asof_date=train_end)
        if model is not None:
            active_models[eng] = model
            model_versions.append({"engine": eng, "asof": train_end, "kind": "initial",
                                   "n_train": len(tr)})
    if not active_models:
        return {"error": "首次训练无可用模型"}

    # 逐日生产循环
    last_scorable = dates[-1 - FORWARD_DAYS]  # 末尾留 FORWARD_DAYS 天算收益
    scoring_dates = [d for d in dates if d >= since and d <= last_scorable]
    for day_idx, day in enumerate(scoring_dates):
        n_folds += 1
        n_dates_scored += 1

        # 每周重训（每 RETRAIN_INTERVAL 天，用截至昨日的全部数据）
        retrain_done = False
        if day_idx > 0 and day_idx % RETRAIN_INTERVAL == 0:
            prev_date = dates[dates.index(day) - 1]
            logger.info("② 每周重训（截至 %s）...", prev_date)
            for eng, info in engines.items():
                masked = info["masked"]
                tr = masked[masked[COL_TRADE_DATE] <= prev_date].dropna(subset=["y_label"])
                model, _ = wre.train_winrate_model(tr, feature_cols=info["feats"],
                                                   forward_days=FORWARD_DAYS,
                                                   engine=eng, asof_date=prev_date)
                if model is not None:
                    active_models[eng] = model
                    model_versions.append({"engine": eng, "asof": prev_date, "kind": "weekly",
                                           "n_train": len(tr)})
            retrain_done = True

        # 当日候选：用截至 day 的特征打分
        day_feat = feat[feat[COL_TRADE_DATE] == day]
        if day_feat.empty:
            continue
        candidates = day_feat[[COL_TS_CODE, "close"]].copy()
        for extra in meta_cols + ["atr_pct"]:
            if extra in day_feat.columns:
                candidates[extra] = day_feat[extra].values

        for eng, model in active_models.items():
            info = engines[eng]
            masked_day = sfm.mask_feature_matrix(day_feat, eng, include_close=True)
            prob = wre.predict_win_probability(model, masked_day[info["feats"]].fillna(0))
            candidates[f"win_prob_{eng}"] = prob

        candidates = candidates.dropna(subset=[f"win_prob_{e}" for e in active_models])
        if candidates.empty:
            continue

        # 动量加成
        if "mom5" in candidates.columns and len(candidates) > 0:
            mom_top_n = max(int(np.ceil(len(candidates) * 0.03)), 1)
            candidates["momentum_boost"] = (
                candidates["mom5"].fillna(-1e9).rank(ascending=False, method="first") <= mom_top_n
            ).values
        else:
            candidates["momentum_boost"] = False

        # Regime（截至当日，防泄漏）
        regime = regime_upto(l1_daily, day)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # T+1 Gap-Risk
        codes = candidates[COL_TS_CODE].unique().tolist()
        t1_risk = t1.compute_t1_lock_risk_batch(
            t1.load_stock_daily(codes=codes, lookback_days=60))

        # 路由选股
        routed = da.route_portfolio(
            candidates, regime,
            winrate_gate=GATE_THRESHOLD,
            t1_risk_scores=t1_risk,
            momentum_boost_col="momentum_boost",
        )
        port = routed.get("portfolio", pd.DataFrame())

        # 每日明细（含真实收益）
        if not port.empty:
            for _, row in port.iterrows():
                px = pd.read_sql_query(
                    "SELECT ts_code, trade_date, close FROM stock_daily "
                    "WHERE ts_code = ? AND trade_date >= ? ORDER BY trade_date LIMIT 6",
                    __import__("sqlite3").connect(DB_PATH),
                    params=[row[COL_TS_CODE], day])
                if px.empty:
                    continue
                entry = float(px.iloc[0]["close"])
                t1px = float(px.iloc[1]["close"]) if len(px) >= 2 else None
                d5px = float(px["close"].iloc[min(5, len(px) - 1)]) if len(px) >= 2 else None
                daily_trades.append({
                    "trade_date": day,
                    "ts_code": row[COL_TS_CODE],
                    "regime": regime,
                    "gate_engine": alloc_engine(regime),
                    "win_prob": round(float(row[f"win_prob_{alloc_engine(regime)}"]), 4),
                    "composite_score": round(float(row.get("composite_score", 0)), 4),
                    "mom5": round(float(row["mom5"]), 4) if "mom5" in row.index else None,
                    "entry_close": round(entry, 2),
                    "t1_close": round(t1px, 2) if t1px else None,
                    "d5_close": round(d5px, 2) if d5px else None,
                    "t1_ret_pct": round((t1px / entry - 1) * 100, 2) if entry and t1px else None,
                    "d5_ret_pct": round((d5px / entry - 1) * 100, 2) if entry and d5px else None,
                })

        if verbose:
            logger.info("日 %s regime=%s 组合%d只", day, regime, len(port))

    daily_df = pd.DataFrame(daily_trades)
    summary = {
        "n_days_scored": n_dates_scored,
        "n_daily_trades": len(daily_df),
        "date_range": f"{scoring_dates[0] if scoring_dates else '-'} ~ {scoring_dates[-1] if scoring_dates else '-'}",
        "regime_distribution": regime_counts,
        "n_model_versions": len(model_versions),
        "model_versions": model_versions[-10:],  # 最近10个版本
        "t1_win_rate": round(float((daily_df["t1_ret_pct"] > 0).mean()), 4) if not daily_df.empty and daily_df["t1_ret_pct"].notna().any() else None,
        "d5_win_rate": round(float((daily_df["d5_ret_pct"] > 0).mean()), 4) if not daily_df.empty and daily_df["d5_ret_pct"].notna().any() else None,
        "elapsed_sec": round(time.time() - t_start, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"summary": summary, "daily_trades": daily_df, "model_versions": model_versions}


def main():
    parser = argparse.ArgumentParser(description="V6 生产模拟回测")
    parser.add_argument("--train-start", type=str, default="20240801", help="首次训练起点")
    parser.add_argument("--train-end", type=str, default="20250731", help="首次训练截止")
    parser.add_argument("--since", type=str, default="20250801", help="明细覆盖起点")
    parser.add_argument("--stocks", type=int, default=None, help="股票数（默认 None=全量池约4999）")
    parser.add_argument("--out", type=str, default="backtest/v6_prod_20260802", help="输出前缀")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    t0 = time.time()
    logger.info("① 加载日线（%s 起，%s 只）...", args.train_start, ("全量" if args.stocks is None else args.stocks))
    from data.stock_industry_mapping import load_stock_universe  # 全量池扩池: universe 过滤
    universe = {u["ts_code"] for u in load_stock_universe()}
    stock = load_stock_daily(max_stocks=args.stocks, min_days=120, since_date=args.train_start, universe=universe)
    logger.info("   %d 行", len(stock))

    logger.info("② 加载行业数据...")
    l1_daily, l2_daily, l3_daily = load_industry_data()

    logger.info("③ 构建全量特征矩阵...")
    feat = build_full_feature_matrix(
        stock_daily_df=stock, l1_daily=l1_daily, l2_daily=l2_daily, l3_daily=l3_daily,
        stock_mapping={}, persistence_scores={"l1": {}, "l2": {}})
    logger.info("   %d 行 × %d 列", len(feat), len(feat.columns))

    logger.info("④ 生产模拟回测...")
    result = run_production_backtest(
        feat, l1_daily,
        train_start=args.train_start, train_end=args.train_end, since=args.since,
        verbose=args.verbose,
    )
    if "error" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    daily = result["daily_trades"]
    if not daily.empty:
        daily.to_csv(f"{args.out}_daily_trades.csv", index=False)
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": result["summary"],
                   "daily_trades": daily.to_dict("records") if not daily.empty else []},
                  f, ensure_ascii=False, indent=2)
    logger.info("报告已保存: %s", args.out)
    if not daily.empty:
        print(f"📊 每日明细: {args.out}_daily_trades.csv ({len(daily)} 条, 覆盖 {daily['trade_date'].nunique()} 交易日)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
