#!/usr/bin/env python3
"""大盘评估 ML 模型训练入口。

用法:
    python3 analysis/train_market_ml.py                    # 训练 20日方向模型
    python3 analysis/train_market_ml.py --target fwd_dir_5  # 训练 5日方向
    python3 analysis/train_market_ml.py --task regression   # 收益回归
    python3 analysis/train_market_ml.py --verbose           # 详细输出
"""
import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_market_ml")

from config import DATA_DIR, DB_PATH, COL_TS_CODE, COL_TRADE_DATE
from analysis.market_ml import (
    build_market_features,
    walk_forward_market,
    save_market_model,
    MARKET_MODEL_PATH,
)

def main():
    parser = argparse.ArgumentParser(description="训练大盘评估 ML 模型")
    parser.add_argument("--target", default="fwd_dir_20",
                        choices=["fwd_dir_5", "fwd_dir_20", "fwd_ret_5", "fwd_ret_20"],
                        help="预测目标")
    parser.add_argument("--task", default="binary",
                        choices=["binary", "regression"],
                        help="任务类型")
    parser.add_argument("--initial-train", type=int, default=500,
                        help="初始训练天数")
    parser.add_argument("--val-days", type=int, default=60,
                        help="验证天数")
    parser.add_argument("--step-days", type=int, default=60,
                        help="步进天数")
    parser.add_argument("--forward-days", type=int, default=20,
                        help="前向收益天数")
    parser.add_argument("--no-dispersion", action="store_true",
                        help="不使用离散度特征")
    args = parser.parse_args()

    # ── 加载数据 ──
    logger.info("加载特征矩阵...")
    cache_path = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
    if not os.path.exists(cache_path):
        logger.error("特征矩阵不存在，请先运行重建: --rebuild")
        sys.exit(1)

    feature_df = pd.read_parquet(cache_path)
    logger.info("特征矩阵: %d 行 × %d 列", len(feature_df), len(feature_df.columns))

    # ── 加载个股日线（用于合成市场指数）──
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    stock_daily_df = pd.read_sql_query(
        "SELECT ts_code, trade_date, pct_chg FROM stock_daily ORDER BY ts_code, trade_date",
        conn,
    )
    conn.close()
    logger.info("个股日线: %d 行", len(stock_daily_df))

    # ── 加载大盘指数（可选，回退到等权合成）──
    try:
        conn = sqlite3.connect(DB_PATH)
        market_index_df = pd.read_sql_query(
            "SELECT * FROM sw_index_daily WHERE ts_code = '000300.SH' ORDER BY trade_date",
            conn,
        )
        conn.close()
        if len(market_index_df) > 10:
            logger.info("沪深300指数: %d 行", len(market_index_df))
        else:
            market_index_df = None
    except Exception:
        market_index_df = None

    # ── 构建市场级特征 ──
    market_df = build_market_features(
        feature_df,
        market_index_df=market_index_df,
        stock_daily_df=stock_daily_df,
        use_dispersion=not args.no_dispersion,
    )
    logger.info("市场级特征: %d 天 × %d 列", len(market_df), len(market_df.columns))

    # ── Walk-Forward 训练 ──
    is_regression = args.task == "regression" or args.target.startswith("fwd_ret")
    result = walk_forward_market(
        market_df,
        target_col=args.target,
        initial_train_days=args.initial_train,
        val_days=args.val_days,
        step_days=args.step_days,
        forward_days=args.forward_days,
        task="regression" if is_regression else "binary",
    )

    # ── 保存 ──
    save_market_model(result)

    # ── 总结 ──
    ov = result["overall"]
    print()
    print("=" * 60)
    print(f"大盘评估模型训练完成")
    print(f"  目标: {args.target}")
    print(f"  折数: {ov['n_folds']}")
    print(f"  ML 平均准确率: {ov['mean_accuracy']:.2%}")
    print(f"  ML 平均 AUC:   {ov['mean_auc']:.4f}")
    print(f"  规则平均准确率: {ov['mean_regime_accuracy']:.2%}")
    print(f"  准确率提升:     {ov['accuracy_improvement']:+.2%}")
    print(f"  击败规则比例:   {ov['beat_regime_ratio']:.0%}")
    print(f"  模型: {MARKET_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
