#!/usr/bin/env python3
"""强势板块 ML 模型训练入口。

用法:
    python3 analysis/train_sector_ml.py
"""
import argparse
import logging
import os
import sys
import sqlite3

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_sector_ml")

from config import DATA_DIR, DB_PATH, COL_TS_CODE, COL_TRADE_DATE
from analysis.sector_ml import (
    load_sector_mapping, build_sector_features, add_sector_targets,
    add_rule_scores, train_sector_model, save_sector_model,
)


def main():
    parser = argparse.ArgumentParser(description="训练板块评估 ML 模型")
    parser.add_argument("--initial-train", type=int, default=500)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--forward-days", type=int, default=20)
    args = parser.parse_args()

    # 加载特征矩阵
    logger.info("加载特征矩阵...")
    cache_path = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
    feature_df = pd.read_parquet(cache_path)
    logger.info("特征矩阵: %d 行 × %d 列", len(feature_df), len(feature_df.columns))

    # 加载个股日线
    conn = sqlite3.connect(DB_PATH)
    stock_daily_df = pd.read_sql_query(
        "SELECT ts_code, trade_date, pct_chg FROM stock_daily ORDER BY ts_code, trade_date",
        conn,
    )
    conn.close()
    logger.info("个股日线: %d 行", len(stock_daily_df))

    # 加载板块映射
    sector_mapping = load_sector_mapping()
    logger.info("SW L1 板块: %d 个", len(set(sector_mapping.values())))

    # 构建板块级特征
    sector_df = build_sector_features(feature_df, sector_mapping)

    # 添加标签
    sector_df = add_sector_targets(sector_df, stock_daily_df, sector_mapping,
                                    forward_days=args.forward_days)

    # 添加规则评分（用于对比）
    sector_df = add_rule_scores(sector_df)
    logger.info("板块 DataFrame: %d 行, %d 列", len(sector_df), len(sector_df.columns))

    # Walk-Forward 训练
    result = train_sector_model(
        sector_df,
        initial_train_days=args.initial_train,
        val_days=args.val_days,
        step_days=args.step_days,
    )

    # 保存
    save_sector_model(result)

    # 总结
    ov = result["overall"]
    print()
    print("=" * 60)
    print(f"板块评估模型训练完成")
    print(f"  折数: {ov['n_folds']}")
    print(f"  ML 平均准确率: {ov['mean_accuracy']:.2%}")
    print(f"  ML 平均 AUC:   {ov['mean_auc']:.4f}")
    print(f"  规则平均准确率: {ov['mean_rule_accuracy']:.2%}")
    print(f"  准确率提升:     {ov['accuracy_improvement']:+.2%}")
    print(f"  击败规则比例:   {ov['beat_rule_ratio']:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
