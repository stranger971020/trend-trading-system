#!/usr/bin/env python3
"""V6 Master Plan — 多策略独立验证流水线（编排入口）。

三步走架构（HERMES-20260801-001）：
  策略拆解 → Alpha 验证 → 组合风控
  ① strategy_feature_masker.py   特征分段（三引擎互斥掩码）
  ② winrate_engine.py            胜率预测头（二分类 LightGBM + P(Win)<55% 一票否决）
  ③ t1_gap_risk.py               T+1 Gap-Risk（大跌次日跳空低开概率，>40% 罚 -15）
  ④ dynamic_allocation.py        Regime 动态路由（bull/range/bear）

# ── Changelog ──
# 2026-08-01 Claude: HERMES-20260801-003 — smoke 路径接入 Fix-2 动量加成：
#   today_candidates 补回 mom5 列，计算 momentum_boost 布尔列（全市场前 3%），
#   route_portfolio 传入 momentum_boost_col，使冒烟路径与 walk-forward 行为一致。
# 告警: 无——momentum_boost 列缺失时 route_portfolio 自动跳过加成。
# ─────────────

用法：
  python3 analysis/ml_v6_pipeline.py --self-test    # 四模块单元自检
  python3 analysis/ml_v6_pipeline.py --smoke        # 真实数据冒烟（快，数分钟内）
  python3 analysis/ml_v6_pipeline.py --train        # 全量 walk-forward 验证（重）
  python3 analysis/ml_v6_pipeline.py --predict      # 按当前 regime 输出今日组合
  python3 analysis/ml_v6_pipeline.py --all          # self-test + smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# 项目根路径引导
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, COL_TS_CODE, COL_TRADE_DATE

from analysis import strategy_feature_masker as sfm
from analysis import winrate_engine as wre
from analysis import t1_gap_risk as t1
from analysis import dynamic_allocation as da

logger = logging.getLogger("ml_v6")

FEATURE_MATRIX_PATH = os.path.join(DATA_DIR, "feature_matrix_v5.parquet")
DB_PATH = os.path.join(DATA_DIR, "sw_index_data.db")

# 冒烟参数
SMOKE_MAX_STOCKS = 150          # 候选股票数
SMOKE_LOOKBACK_DATES = 70       # 回看交易日
SMOKE_TRAIN_DATES = 40          # 训练交易日
SMOKE_VAL_DATES = 10            # 验证交易日
FORWARD_DAYS = 20

# ── Fix-2 动量加成参数 (HERMES-20260801-003) ──
MOMENTUM_LOOKBACK = 5           # 过去 N 日收益（mom5）
MOMENTUM_TOP_PCT = 0.03         # 全市场前 3% 动量 → 加成
MOMENTUM_BOOST_MULT = 1.1       # 加成倍数


def apply_momentum_boost(
    scored_df: pd.DataFrame,
    mom_col: str = "mom5",
    top_pct: float = MOMENTUM_TOP_PCT,
    boost_mult: float = MOMENTUM_BOOST_MULT,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """动量加成 (Fix-2)：过去 MOMENTUM_LOOKBACK 日收益排进全市场前 top_pct 的
    候选，composite_score 额外 ×boost_mult，补偿短期动能属性。

    Args:
        scored_df: 候选池（含 mom5 列与 composite_score）
        mom_col: 5日动量列名（无则跳过加成）
        top_pct: 动量前百分之几获得加成（默认 3%）
        boost_mult: 加成倍数（默认 1.1）
        score_col: 被加成的总分列

    Returns:
        新增 momentum_boost 布尔列、已加成 composite_score 的 DataFrame。
    """
    df = scored_df.copy()
    if mom_col not in df.columns or score_col not in df.columns or df.empty:
        df["momentum_boost"] = False
        return df
    mom = df[mom_col].fillna(-1e9)
    top_n = max(int(np.ceil(len(df) * top_pct)), 1)
    is_top = mom.rank(ascending=False, method="first") <= top_n
    df["momentum_boost"] = is_top.values
    df.loc[is_top, score_col] = df.loc[is_top, score_col] * boost_mult
    return df


# ═══════════════════════════════════════════════════════════════
# 数据装载（只读所需列，避免全量加载 6.8GB）
# ═══════════════════════════════════════════════════════════════

def _top_codes_by_amount(n: int = SMOKE_MAX_STOCKS) -> list[str]:
    """按日均成交额取流动性最好的 n 只股票（避免选到僵尸股）。"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT ts_code FROM stock_daily WHERE trade_date >= '20260101' "
            "GROUP BY ts_code HAVING AVG(amount) > 0 "
            "ORDER BY AVG(amount) DESC LIMIT ?", (n,)
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def load_feature_slice(codes: list[str], lookback_dates: int = SMOKE_LOOKBACK_DATES) -> pd.DataFrame:
    """加载特征矩阵子集（指定股票 + 最近 N 个交易日）。

    列裁剪：只读 ts_code/trade_date/close + 三引擎特征池中存在的列。
    """
    import pyarrow.parquet as pq
    schema = pq.read_schema(FEATURE_MATRIX_PATH)
    all_cols = schema.names
    wanted = {"ts_code", "trade_date", "close"}
    # 熊市防御路由需要价值/红利维度
    wanted.update({"value_score", "earnings_yield", "quality_score"})
    for feats in sfm.ENGINE_FEATURE_MASKS.values():
        wanted.update(feats)
    wanted = sorted(wanted & set(all_cols))
    df = pq.read_table(FEATURE_MATRIX_PATH, columns=wanted).to_pandas()
    df = df[df[COL_TS_CODE].isin(codes)].copy()

    dates = sorted(df[COL_TRADE_DATE].unique())
    if len(dates) > lookback_dates:
        keep = set(dates[-lookback_dates:])
        df = df[df[COL_TRADE_DATE].isin(keep)].copy()
    df = df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).reset_index(drop=True)
    logger.info("特征子集: %d 只 × %d 天 = %d 行", df[COL_TS_CODE].nunique(),
                df[COL_TRADE_DATE].nunique(), len(df))
    return df


# ═══════════════════════════════════════════════════════════════
# 冒烟流水线
# ═══════════════════════════════════════════════════════════════

def run_smoke() -> dict:
    """真实数据冒烟：三引擎各自训练+预测 → T+1 风控 → Regime 路由。"""
    t_start = time.time()
    result: dict = {"step": "smoke", "forward_days": FORWARD_DAYS}

    # ① 候选股票 + 特征切片
    codes = _top_codes_by_amount(SMOKE_MAX_STOCKS)
    result["candidate_stocks"] = len(codes)
    feat = load_feature_slice(codes, SMOKE_LOOKBACK_DATES)

    dates = sorted(feat[COL_TRADE_DATE].unique())
    n_dates = len(dates)
    if n_dates < SMOKE_TRAIN_DATES + SMOKE_VAL_DATES + FORWARD_DAYS:
        result["error"] = f"交易日不足: {n_dates}"
        return result

    # ② 每引擎：掩码 → 标签 → 训练 → 预测（训练窗口）
    train_dates = set(dates[:SMOKE_TRAIN_DATES])
    val_dates = set(dates[SMOKE_TRAIN_DATES:SMOKE_TRAIN_DATES + SMOKE_VAL_DATES])
    engines = {}

    vol_carrier = feat[["ts_code", "trade_date", "atr_pct"]].drop_duplicates(
        ["ts_code", "trade_date"]) if "atr_pct" in feat.columns else None

    for engine in sfm.ALL_ENGINES:
        masked = sfm.mask_feature_matrix(feat, engine, include_close=True)
        masked = wre.build_winrate_labels(masked, forward_days=FORWARD_DAYS)
        train_df = masked[masked[COL_TRADE_DATE].isin(train_dates)].dropna(subset=["y_label"])
        val_df = masked[masked[COL_TRADE_DATE].isin(val_dates)].dropna(subset=["y_label"])
        if len(train_df) < 300 or val_df.empty:
            result[f"engine_{engine}"] = {"error": "数据不足"}
            continue
        feats = sfm.get_engine_features(engine)
        feats = [c for c in feats if c in train_df.columns]
        model, imp = wre.train_winrate_model(train_df, feature_cols=feats, forward_days=FORWARD_DAYS)
        if model is None:
            result[f"engine_{engine}"] = {"error": "训练失败"}
            continue
        val_df = val_df.copy()
        val_df[f"win_prob_{engine}"] = wre.predict_win_probability(model, val_df[feats].fillna(0))
        # 补充 atr_pct 以便 Avg_Volatility_of_Top5 有真实波动度量（避免掩码内已有时产生 _x/_y）
        if vol_carrier is not None and "atr_pct" not in val_df.columns:
            val_df = val_df.merge(vol_carrier, on=["ts_code", "trade_date"], how="left")
        panel = wre.evaluate_panel(val_df, f"win_prob_{engine}", top_n=5)
        engines[engine] = {
            "n_train": len(train_df), "n_val": len(val_df),
            "n_features": len(feats),
            "panel": panel,
            "top_features": imp["feature"].head(5).tolist(),
            "model": model,
            "feats": feats,
        }
        result[f"engine_{engine}"] = {
            "n_train": len(train_df), "n_val": len(val_df), "n_features": len(feats),
            "panel": panel, "top_features": imp["feature"].head(5).tolist(),
        }

    # ③ 最新一日候选（今日选股池）
    latest_date = dates[-1]
    latest_feat = feat[feat[COL_TRADE_DATE] == latest_date]
    keep_cols = ["ts_code", "close"]
    for extra in ("atr_pct", "value_score", "earnings_yield", "quality_score", "mom5"):
        if extra in latest_feat.columns:
            keep_cols.append(extra)
    today_candidates = latest_feat[keep_cols].copy()
    for engine, info in engines.items():
        if "model" not in info:
            continue
        masked = sfm.mask_feature_matrix(latest_feat, engine, include_close=True)
        today_candidates[f"win_prob_{engine}"] = wre.predict_win_probability(
            info["model"], masked[info["feats"]].fillna(0))
    today_candidates = today_candidates.dropna(subset=[f"win_prob_{e}" for e in engines])

    # Fix-2 (HERMES-20260801-003): 动量加成列（mom5 全市场前 3%）
    if "mom5" in today_candidates.columns and not today_candidates.empty:
        mom_top_n = max(int(np.ceil(len(today_candidates) * MOMENTUM_TOP_PCT)), 1)
        today_candidates["momentum_boost"] = (
            today_candidates["mom5"].fillna(-1e9).rank(ascending=False, method="first") <= mom_top_n
        ).values
    else:
        today_candidates["momentum_boost"] = False

    # ④ T+1 Gap-Risk（真实 stock_daily）
    daily = t1.load_stock_daily(codes=today_candidates["ts_code"].tolist(),
                                lookback_days=t1.DEFAULT_LOOKBACK_DAYS)
    risk_scores = t1.compute_t1_lock_risk_batch(daily)

    # ⑤ Regime 路由（软权重 + 动量加成）
    regime_info = da.get_regime_from_market_ml()
    routed = da.route_portfolio(
        today_candidates, regime_info["regime"],
        winrate_gate=wre.DEFAULT_WINRATE_GATE,
        t1_risk_scores=risk_scores,
        momentum_boost_col="momentum_boost",
    )

    result["latest_date"] = latest_date
    result["regime"] = routed.get("regime")
    result["regime_source"] = regime_info.get("source")
    result["raw_regime"] = regime_info.get("raw_regime")
    result["allocation"] = routed.get("allocation")
    result["gate_summary"] = routed.get("gate_summary")
    result["panel"] = routed.get("panel")
    result["t1_risk_summary"] = risk_scores[["ts_code", "score"]].rename(
        columns={"score": "t1_lock_risk"}).to_dict("records")

    port = routed.get("portfolio", pd.DataFrame())
    if not port.empty:
        result["portfolio"] = port.drop(columns=["engine_weights"], errors="ignore").to_dict("records")
    else:
        result["portfolio"] = []

    result["elapsed_sec"] = round(time.time() - t_start, 1)
    return result


# ═══════════════════════════════════════════════════════════════
# 自检（四模块单元测试）
# ═══════════════════════════════════════════════════════════════

def run_self_test() -> dict:
    t0 = time.time()
    return {
        "strategy_feature_masker": sfm._self_test(),
        "winrate_engine": wre._self_test(),
        "t1_gap_risk": t1._self_test(),
        "dynamic_allocation": da._self_test(),
        "elapsed_sec": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V6 Master Plan 多策略独立验证流水线")
    parser.add_argument("--self-test", action="store_true", help="四模块单元自检")
    parser.add_argument("--smoke", action="store_true", help="真实数据冒烟")
    parser.add_argument("--train", action="store_true", help="全量 walk-forward 验证（重）")
    parser.add_argument("--predict", action="store_true", help="按当前 regime 输出今日组合")
    parser.add_argument("--all", action="store_true", help="self-test + smoke")
    parser.add_argument("--out", type=str, default="", help="JSON 输出路径")
    args = parser.parse_args()

    output = {}
    if args.self_test or args.all:
        logger.info("▶ 四模块单元自检")
        output["self_test"] = run_self_test()
    if args.smoke or args.all:
        logger.info("▶ 真实数据冒烟")
        output["smoke"] = run_smoke()
    if args.train:
        logger.info("▶ 全量 walk-forward（各引擎）")
        output["train"] = _run_full_train()
    if args.predict:
        output["predict"] = _run_predict()

    if not output:
        parser.print_help()
        return

    text = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("输出已写入: %s", args.out)
    print(text)


def _run_predict() -> dict:
    """按当前 regime 输出今日组合（复用 smoke 逻辑但只出组合）。"""
    return run_smoke()


def _run_full_train() -> dict:
    """全量 walk-forward（任务步骤 2 的完整验证）——注意：重。"""
    from analysis.retrain_ml import load_stock_daily
    stock = load_stock_daily(max_stocks=1200, min_days=120)
    from analysis.ml_model import build_full_feature_matrix
    feat = build_full_feature_matrix(stock_daily_df=stock)
    results = {}
    for engine in sfm.ALL_ENGINES:
        masked = sfm.mask_feature_matrix(feat, engine, include_close=True)
        feats = sfm.get_engine_features(engine)
        feats = [c for c in feats if c in masked.columns]
        r = wre.walk_forward_winrate(masked, feats, initial_train_days=120,
                                     val_days=25, step_days=25, forward_days=20)
        results[engine] = {k: v for k, v in r.items() if k != "fold_models"}
        logger.info("引擎 %s: %s", engine, json.dumps(results[engine].get("overall", {}), ensure_ascii=False))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
