#!/usr/bin/env python3
"""
ML 模型训练脚本
- 从历史数据构建特征矩阵
- 训练 LightGBM LambdaRank 模型
- 保存到 data_storage/lgb_model.pkl

用法: python3 train_ml_model.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.industry_daily_updater import load_daily_data, get_db_connection
from data.stock_daily_updater import load_stock_daily
from config import DB_PATH
from data.stock_industry_mapping import load_stock_industry_mapping
from analysis.ml_model import build_feature_matrix, train_model, save_model, FEATURE_COLS, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ml")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("ML 模型训练")

    # 加载数据
    conn = get_db_connection()
    l1_codes = [row[0] for row in conn.execute("SELECT DISTINCT ts_code FROM sw_index_daily").fetchall()]
    l1_df = load_daily_data(conn, l1_codes, min_rows=100)

    all_stocks = [row[0] for row in conn.execute("SELECT DISTINCT ts_code FROM stock_daily").fetchall()]
    stock_df = load_stock_daily(DB_PATH, all_stocks, min_rows=20)
    conn.close()

    stock_mapping = load_stock_industry_mapping()

    logger.info("L1 行业: %d, 个股: %d", len(l1_codes), len(all_stocks))

    # 构建特征矩阵（多日模式，用于训练）
    features = build_feature_matrix(stock_df, l1_df, stock_mapping, multi_day=True)
    logger.info("特征矩阵: %d 行 × %d 列", len(features), len(features.columns))

    # 训练
    model, importance = train_model(features, forward_days=20)

    if model is not None:
        save_model(model, MODEL_PATH)
        logger.info("\n特征重要性 Top 10:")
        for _, row in importance.head(10).iterrows():
            logger.info("  %-25s %.4f", row["feature"], row["importance"])
        logger.info("\n✅ 训练完成, 模型已保存")
    else:
        logger.error("训练失败：数据不足")
        sys.exit(1)
