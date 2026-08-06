#!/usr/bin/env python3
"""V6 三层串联生产模拟（大盘 → 行业 → 个股）。

架构:
  Layer 1 (市场): market_ml.py → 每日预测 up_prob → 转 regime (bull/range/bear)
  Layer 2 (行业): sector_ml.py → 每日预测行业跑赢概率 → 选 Top-N 行业
  Layer 3 (个股): V6 三引擎 → 在 Top-N 行业内打分 → route_portfolio 选股

模型持久化: market_{asof}.pkl / sector_{asof}.pkl / v6_{engine}_{asof}.pkl
每周重训三层全部模型。

用法:
  python3 backtest/v6_three_layer_backtest.py
      --train-start 20240801 --train-end 20250731
      --since 20250801
      --out backtest/v6_three_layer_20260802
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, COL_TS_CODE, COL_TRADE_DATE, DATA_DIR
from analysis import strategy_feature_masker as sfm
from analysis import winrate_engine as wre
from analysis import t1_gap_risk as t1
from analysis import dynamic_allocation as da
from analysis.retrain_ml import load_stock_daily, load_industry_data
from analysis.ml_model import build_full_feature_matrix
from analysis import market_ml, sector_ml

logger = logging.getLogger("v6_3layer")
MODEL_DIR = os.path.join(DATA_DIR, "lgb_models")
FORWARD_DAYS = 20
RETRAIN_INTERVAL = 5
TOP_SECTORS = 5    # 行业模型选 Top-N 行业
GATE_THRESHOLD = 0.55


def alloc_engine(regime: str) -> str:
    return "momentum" if regime == "bull" else "reversion"


def save_model_versioned(model, prefix: str, asof_date: str) -> str:
    """保存版本化模型。"""
    path = os.path.join(MODEL_DIR, f"{prefix}_{asof_date}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("已保存: %s", path)
    return path


def load_model_latest(prefix: str):
    """加载最新版本。"""
    import glob
    files = sorted(glob.glob(os.path.join(MODEL_DIR, f"{prefix}_*.pkl")))
    if not files:
        return None
    with open(files[-1], "rb") as f:
        return pickle.load(f)


def build_market_features_daily(feat: pd.DataFrame, l1_daily: pd.DataFrame) -> pd.DataFrame:
    """从个股特征聚合为大盘每日特征（复用 market_ml.build_market_features 逻辑）。"""
    try:
        return market_ml.build_market_features(feat, l1_daily)
    except Exception:
        # 退化：用等权均值
        date_features = feat.groupby(COL_TRADE_DATE).agg(["mean", "std"]).fillna(0)
        date_features.columns = [f"{c[0]}_{c[1]}" for c in date_features.columns]
        return date_features.reset_index()


def build_sector_features_daily(feat: pd.DataFrame, sector_mapping: dict) -> pd.DataFrame:
    """从个股特征聚合为行业每日特征。"""
    from analysis.sector_ml import build_sector_features
    return build_sector_features(feat, sector_mapping)


def run_three_layer(
    feat: pd.DataFrame,
    l1_daily: pd.DataFrame,
    sector_mapping: dict,
    train_start: str,
    train_end: str,
    since: str,
    verbose: bool = False,
) -> dict:
    t_start = time.time()
    feat = feat.dropna(subset=[COL_TRADE_DATE]).sort_values([COL_TS_CODE, COL_TRADE_DATE])
    dates = sorted(feat[COL_TRADE_DATE].unique())
    if train_end not in dates:
        train_end = max(d for d in dates if d <= train_end)

    # ── Layer 1+2 特征预构建 ──
    logger.info("构建大盘特征...")
    mkt_feat = build_market_features_daily(feat, l1_daily)
    logger.info("构建行业特征...")
    sec_feat = build_sector_features_daily(feat, sector_mapping)

    # ── Layer 3 引擎特征 ──
    engines = {}
    for eng in sfm.ALL_ENGINES:
        masked = sfm.mask_feature_matrix(feat, eng, include_close=True)
        masked = wre.build_winrate_labels(masked, forward_days=FORWARD_DAYS)
        feats = [c for c in sfm.get_engine_features(eng) if c in masked.columns]
        engines[eng] = {"masked": masked, "feats": feats}

    meta_cols = ["mom5", "value_score", "earnings_yield", "quality_score", "vol_ratio"]

    # ── Layer 1+2: 用已有的 walk-forward 验证过的模型 ──
    logger.info("加载大盘模型...")
    mkt_model = market_ml.load_market_model()
    if mkt_model:
        save_model_versioned(mkt_model, "market", train_end)
    # 读特征列信息
    # 特征列自动推断（报告不存 feature_cols，从聚合矩阵推导）
    mkt_feat_cols = [c for c in mkt_feat.columns if c.endswith("_mean") or c.endswith("_std")]

    logger.info("加载行业模型...")
    sec_model = sector_ml.load_sector_model()
    if sec_model:
        save_model_versioned(sec_model, "sector", train_end)
    sec_feat_cols = [c for c in sec_feat.columns if c.endswith("_mean") or c.endswith("_std")]

    # Layer 3: 个股
    active_models = {}
    for eng, info in engines.items():
        tr = info["masked"][info["masked"][COL_TRADE_DATE] <= train_end].dropna(subset=["y_label"])
        model, _ = wre.train_winrate_model(tr, feature_cols=info["feats"],
                                           forward_days=FORWARD_DAYS,
                                           engine=eng, asof_date=train_end)
        if model is not None:
            active_models[eng] = model

    # ── 逐日生产 ──
    daily_trades = []
    model_versions = []
    regime_counts = {"bull": 0, "range": 0, "bear": 0}
    scoring_dates = [d for d in dates if d >= since and d <= dates[-1 - FORWARD_DAYS]]

    for day_idx, day in enumerate(scoring_dates):
        # 每周重训（仅个股层，大盘/行业用已验证的 walk-forward 模型）
        if day_idx > 0 and day_idx % RETRAIN_INTERVAL == 0:
            prev_date = dates[dates.index(day) - 1]
            logger.info("=== 每周重训个股（截至 %s）===", prev_date)
            for eng, info in engines.items():
                tr = info["masked"][info["masked"][COL_TRADE_DATE] <= prev_date].dropna(subset=["y_label"])
                model, _ = wre.train_winrate_model(tr, feature_cols=info["feats"],
                                                   forward_days=FORWARD_DAYS,
                                                   engine=eng, asof_date=prev_date)
                if model is not None:
                    active_models[eng] = model

        # ── Layer 1: 大盘预测 → regime ──
        mkt_day = mkt_feat[mkt_feat[COL_TRADE_DATE] == day]
        if mkt_model and mkt_feat_cols and not mkt_day.empty:
            avail = [c for c in mkt_feat_cols if c in mkt_day.columns]
            if avail:
                up_prob = float(mkt_model.predict_proba(mkt_day[avail].fillna(0))[:, 1][0])
            else:
                up_prob = 0.5
        else:
            up_prob = 0.5

        regime = "bull" if up_prob > 0.6 else ("bear" if up_prob < 0.4 else "range")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # ── Layer 2: 行业预测 → Top-N ──
        sec_day = sec_feat[sec_feat[COL_TRADE_DATE] == day]
        top_sector_codes = []
        if sec_model and sec_feat_cols and not sec_day.empty:
            avail = [c for c in sec_feat_cols if c in sec_day.columns]
            if avail:
                probs = sec_model.predict_proba(sec_day[avail].fillna(0))
                sec_day = sec_day.copy()
                sec_day["sector_prob"] = probs[:, 1] if probs.ndim == 2 else probs
                top_sector_codes = sec_day.nlargest(TOP_SECTORS, "sector_prob")["sector"].tolist()

        # ── Layer 3: 个股打分 ──
        day_feat = feat[feat[COL_TRADE_DATE] == day]
        if day_feat.empty:
            continue

        # 行业过滤（用映射表，不在特征矩阵中存字符串列）
        if top_sector_codes:
            day_feat["_l1"] = day_feat[COL_TS_CODE].map(sector_mapping)
            day_feat = day_feat[day_feat["_l1"].isin(top_sector_codes)]
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

        # T+1 Gap-Risk
        codes = candidates[COL_TS_CODE].unique().tolist()
        if codes:
            t1_risk = t1.compute_t1_lock_risk_batch(
                t1.load_stock_daily(codes=codes, lookback_days=60))
        else:
            t1_risk = pd.DataFrame()

        # 路由
        routed = da.route_portfolio(
            candidates, regime,
            winrate_gate=GATE_THRESHOLD,
            t1_risk_scores=t1_risk,
            momentum_boost_col="momentum_boost",
        )
        port = routed.get("portfolio", pd.DataFrame())

        if not port.empty:
            conn = sqlite3.connect(DB_PATH)
            for _, row in port.iterrows():
                try:
                    px = pd.read_sql_query(
                        "SELECT ts_code, trade_date, close FROM stock_daily "
                        "WHERE ts_code = ? AND trade_date >= ? ORDER BY trade_date LIMIT 6",
                        conn, params=[row[COL_TS_CODE], day])
                except Exception:
                    continue
                if px.empty:
                    continue
                entry = float(px.iloc[0]["close"])
                t1px = float(px.iloc[1]["close"]) if len(px) >= 2 else None
                d5px = float(px["close"].iloc[min(5, len(px) - 1)]) if len(px) >= 2 else None
                daily_trades.append({
                    "trade_date": day,
                    "ts_code": row[COL_TS_CODE],
                    "regime": regime,
                    "up_prob": round(up_prob, 3),
                    "top_sectors": ",".join(top_sector_codes[:3]) if top_sector_codes else "",
                    "win_prob": round(float(row.get(f"win_prob_{alloc_engine(regime)}", 0)), 4),
                    "composite_score": round(float(row.get("composite_score", 0)), 4),
                    "entry_close": round(entry, 2),
                    "t1_close": round(t1px, 2) if t1px else None,
                    "d5_close": round(d5px, 2) if d5px else None,
                    "t1_ret_pct": round((t1px / entry - 1) * 100, 2) if entry and t1px else None,
                    "d5_ret_pct": round((d5px / entry - 1) * 100, 2) if entry and d5px else None,
                })
            conn.close()

        if verbose and not port.empty:
            logger.info("日 %s regime=%s up=%.2f sectors=%s 组合%d只",
                        day, regime, up_prob, top_sector_codes[:3], len(port))

    daily_df = pd.DataFrame(daily_trades)
    summary = {
        "n_days": len(scoring_dates),
        "n_trades": len(daily_df),
        "date_range": f"{scoring_dates[0]}_{scoring_dates[-1]}" if scoring_dates else "-",
        "regime_distribution": regime_counts,
        "layer": "market+sector+stock",
        "t1_win_rate": round(float((daily_df["t1_ret_pct"] > 0).mean()), 4) if not daily_df.empty and daily_df["t1_ret_pct"].notna().any() else None,
        "d5_win_rate": round(float((daily_df["d5_ret_pct"] > 0).mean()), 4) if not daily_df.empty and daily_df["d5_ret_pct"].notna().any() else None,
        "t1_avg_pct": round(float(daily_df["t1_ret_pct"].mean()), 2) if not daily_df.empty else None,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    return {"summary": summary, "daily_trades": daily_df}


def _train_market_model(mkt_feat: pd.DataFrame) -> tuple:
    """训练市场模型（用全量历史单向训练）。"""
    if mkt_feat is None or mkt_feat.empty or len(mkt_feat) < 100:
        return None, []

    import lightgbm as lgb
    mkt_feat = mkt_feat.sort_values(COL_TRADE_DATE).reset_index(drop=True)

    # 用「当日所有特征的均值」作为市场代理价格，计算 fwd 20日"涨跌"
    # 找最能代表市场方向的列：优先 mom20_mean（20日动量均值）
    price_col = None
    for c in ["mom20_mean", "close_mean", "close"]:
        if c in mkt_feat.columns:
            price_col = c
            break
    if price_col is None:
        # 退化为第一列数值列
        for c in mkt_feat.columns:
            if c.endswith("_mean") and mkt_feat[c].dtype in (float, int):
                price_col = c
                break
    if price_col is None:
        return None, []

    mkt_feat["fwd_val"] = mkt_feat[price_col].shift(-FORWARD_DAYS)
    mkt_feat["y_label"] = (mkt_feat["fwd_val"] > mkt_feat[price_col]).astype(int)
    train = mkt_feat.dropna(subset=["y_label"])
    if len(train) < 50:
        return None, []

    exclude = {"trade_date", "ts_code", "fwd_val", "y_label"}
    feats = [c for c in train.columns if c not in exclude and train[c].dtype in (float, int)]
    if not feats:
        return None, []

    X = train[feats].fillna(0)
    y = train["y_label"].values
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, verbosity=-1,
                               objective="binary", metric="binary_logloss")
    model.fit(X, y)
    logger.info("大盘模型训练: %d 样本, %d 特征, pos_rate=%.2f", len(train), len(feats), y.mean())
    return model, feats


def _train_sector_model(sec_feat: pd.DataFrame) -> tuple:
    """训练行业模型（简化版）。"""
    if sec_feat is None or sec_feat.empty:
        return None, []
    if "sector_ret" not in sec_feat.columns:
        return None, []

    import lightgbm as lgb
    # 构建目标：行业收益率 > 0
    sec_feat = sec_feat.copy()
    sec_feat["y_label"] = (sec_feat["sector_ret"] > 0).astype(int)
    train = sec_feat.dropna(subset=["y_label"])
    if len(train) < 100:
        return None, []

    exclude = {"trade_date", "sector", "y_label", "sector_ret", "fwd_ret",
               "fwd_rank", "fwd_outperform", "rule_score", "rule_rank_3",
               "rule_rsi", "rule_mom", "rule_stability", "sector_prob"}
    feats = [c for c in train.columns if c not in exclude and train[c].dtype in (float, int)]
    if not feats:
        return None, []

    X = train[feats].fillna(0)
    y = train["y_label"].values
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=12, verbosity=-1,
                               objective="binary", metric="binary_logloss")
    model.fit(X, y)
    logger.info("行业模型训练: %d 样本, %d 特征, pos_rate=%.2f", len(train), len(feats), y.mean())
    return model, feats


def main():
    parser = argparse.ArgumentParser(description="V6 三层串联生产模拟")
    parser.add_argument("--train-start", type=str, default="20240801")
    parser.add_argument("--train-end", type=str, default="20250731")
    parser.add_argument("--since", type=str, default="20250801")
    parser.add_argument("--stocks", type=int, default=None, help="股票数（默认 None=全量池约4999）")
    parser.add_argument("--out", type=str, default="backtest/v6_three_layer_20260802")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    t0 = time.time()
    logger.info("① 加载数据...")
    from data.stock_industry_mapping import load_stock_industry_mapping, load_stock_universe  # 全量池扩池
    universe = {u["ts_code"] for u in load_stock_universe()}
    stock = load_stock_daily(max_stocks=args.stocks, min_days=120, since_date=args.train_start, universe=universe)
    l1_daily, l2_daily, l3_daily = load_industry_data()

    raw_mapping = load_stock_industry_mapping()
    # build_full_feature_matrix 需要嵌套 dict（{ts_code: {l1_code: ..., l1_name: ...}}）
    # build_sector_features 需要扁平 dict（{ts_code: l1_code}）
    sector_mapping_flat = {k: v["l1_code"] for k, v in raw_mapping.items()}

    logger.info("② 构建全量特征矩阵...")
    feat = build_full_feature_matrix(
        stock_daily_df=stock, l1_daily=l1_daily, l2_daily=l2_daily, l3_daily=l3_daily,
        stock_mapping=raw_mapping, persistence_scores={"l1": {}, "l2": {}})

    logger.info("③ 三层串联模拟...")
    result = run_three_layer(
        feat, l1_daily, sector_mapping_flat,
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
        print(f"📊 每日明细: {args.out}_daily_trades.csv ({len(daily)} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
