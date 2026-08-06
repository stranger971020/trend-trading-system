#!/usr/bin/env python3
"""
v6_weekly_retrain.py — V6 引擎模型周度重训练

轻量模式: 只训练最新 1 个 fold（12 周训练 + 1 周验证），不跑全量 98 folds。
全量 WFV 每月手动跑一次做评估。

用法:
  python3 backtest/v6_weekly_retrain.py              # 默认 12 周
  python3 backtest/v6_weekly_retrain.py --weeks 8    # 自定义训练窗口

输出:
  data_storage/lgb_models/v6_{engine}_{asof}.pkl  (3 个文件)
  logs/v6_weekly_retrain.log                       (训练日志)

── Changelog ──
# 2026-08-02 Claude: 初版，1-fold 轻量重训练
─────────────
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.winrate_engine import (
    train_winrate_model,
    predict_win_probability,
    save_engine_model,
    build_winrate_labels,
    rank_ic,
)
from analysis.strategy_feature_masker import mask_feature_matrix, get_engine_features

FEAT_PATH = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "v6_weekly_retrain.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("v6_retrain")

ENGINES = ["momentum", "reversion", "breakout"]
TRAIN_WEEKS = 12
VAL_WEEKS = 1
MIN_TRAIN_ROWS = 50000


def load_and_prepare():
    """加载特征矩阵，计算标签"""
    feat = pd.read_parquet(FEAT_PATH)
    feat["trade_date"] = pd.to_datetime(feat["trade_date"], format="%Y%m%d")
    logger.info("特征矩阵: %d 行 × %d 列, %s ~ %s",
                len(feat), len(feat.columns),
                feat["trade_date"].min().strftime("%Y%m%d"),
                feat["trade_date"].max().strftime("%Y%m%d"))

    return feat


def train_one_engine(feat, engine, train_start, train_end, val_start, val_end):
    """训练单个引擎的 1 个 fold"""
    logger.info("━" * 40)
    logger.info("引擎: %s", engine)
    logger.info("训练: %s ~ %s, 验证: %s ~ %s",
                train_start, train_end, val_start, val_end)

    # 特征准备
    masked = mask_feature_matrix(feat, engine, include_close=True)
    feats = [c for c in get_engine_features(engine) if c in masked.columns]
    logger.info("  特征数: %d", len(feats))

    # 切分
    train_mask = (masked["trade_date"] >= train_start) & (masked["trade_date"] <= train_end)
    val_mask = (masked["trade_date"] >= val_start) & (masked["trade_date"] <= val_end)

    train_df = masked[train_mask].copy()
    val_df = masked[val_mask].copy()

    if len(train_df) < MIN_TRAIN_ROWS:
        logger.warning("  训练数据不足: %d 行 < %d", len(train_df), MIN_TRAIN_ROWS)
        return None

    logger.info("  训练: %d 行, 验证: %d 行", len(train_df), len(val_df))

    # 构建标签
    train_df = build_winrate_labels(train_df, forward_days=20)
    val_df = build_winrate_labels(val_df, forward_days=20)

    # 训练
    t0 = time.time()
    model, importance = train_winrate_model(train_df, feature_cols=feats)
    elapsed = time.time() - t0
    logger.info("  训练耗时: %.1f 秒", elapsed)

    if model is None:
        return None

    # 验证
    val_X = val_df[feats].fillna(0)
    val_probs = predict_win_probability(model, val_X)
    val_df["win_prob"] = val_probs

    # IC
    ic = rank_ic(val_df, "win_prob", "fwd_return_20d", alpha_boost=2.0)
    ic_unboosted = rank_ic(val_df, "win_prob", "fwd_return_20d", alpha_boost=1.0)

    # 胜率 (PWin > 0.55 的股票中, fwd > 0 的占比)
    top = val_df[val_df["win_prob"] > 0.55]
    if len(top) > 0:
        wr = (top["fwd_return_20d"] > 0).mean()
    else:
        wr = 0

    logger.info("  验证 IC: %.4f (unboosted: %.4f), WR(PWin>0.55): %.1f%%",
                ic, ic_unboosted, wr * 100)

    return {
        "model": model,
        "engine": engine,
        "ic": ic,
        "ic_unboosted": ic_unboosted,
        "wr": wr,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "elapsed": elapsed,
        "top_features": list(importance.head(5).index) if importance is not None and len(importance) > 0 else [],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V6 引擎模型周度重训练")
    parser.add_argument("--weeks", type=int, default=TRAIN_WEEKS, help="训练窗口周数")
    args = parser.parse_args()

    logger.info("═" * 50)
    logger.info("V6 周度模型重训练 — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("═" * 50)

    feat = load_and_prepare()
    all_dates = sorted(feat["trade_date"].unique())
    latest_date = all_dates[-1]

    # 计算训练/验证窗口
    val_end = latest_date
    val_start = val_end - pd.DateOffset(weeks=VAL_WEEKS)
    train_end = val_start - timedelta(days=1)
    train_start = train_end - pd.DateOffset(weeks=args.weeks)

    # 确保是交易日
    train_start_str = train_start.strftime("%Y%m%d")
    train_end_str = train_end.strftime("%Y%m%d")
    val_start_str = val_start.strftime("%Y%m%d")
    val_end_str = val_end.strftime("%Y%m%d")

    # asof_date: 训练数据截止日 → 用作模型文件名
    asof_date = train_end_str

    logger.info("训练窗口: %d 周, 验证窗口: %d 周", args.weeks, VAL_WEEKS)

    results = {}
    for eng in ENGINES:
        result = train_one_engine(feat, eng, train_start, train_end, val_start, val_end)
        if result:
            save_engine_model(result["model"], eng, asof_date)
            logger.info("  模型已保存: v6_%s_%s.pkl", eng, asof_date)
            results[eng] = result

    # ── 摘要 ──
    logger.info("═" * 50)
    logger.info("训练摘要")
    logger.info("═" * 50)
    logger.info("  模型日期: %s", asof_date)
    for eng in ENGINES:
        if eng in results:
            r = results[eng]
            logger.info("  %s: IC=%.4f WR=%.1f%% %ds %d行",
                        eng, r["ic"], r["wr"] * 100, int(r["elapsed"]), r["n_train"])
    logger.info("═" * 50)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
