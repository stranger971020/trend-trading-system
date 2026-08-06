"""ML 公共工具 — 供 market_ml / sector_ml / ml_model / retrain_ml 复用。

背景：2026-07-31 修复「每日增量训练从未生效」的根因——新鲜度检查读取的
feature_matrix_v4.parquet 无 cron 刷新，导致 parquet 日期与模型训练日期
恒差 0~1 天，永远触发不了重训练。

修复原则：权威数据源是 SQLite 的 stock_daily 表（每日由数据更新流程写入），
parquet 只是特征缓存，不作为日期判断依据。
"""
import logging
import os
import sqlite3

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)


def get_latest_stock_daily_date(db_path: str | None = None) -> pd.Timestamp | None:
    """从 SQLite stock_daily 表读取真实最新交易日。

    这是模型新鲜度检查的权威数据源。stock_daily 由数据更新流程每日写入，
    而 feature_matrix_v4.parquet 仅由增量更新任务刷新（可能滞后），
    故不可用 parquet 日期判断数据是否过期。
    """
    db_path = db_path or DB_PATH
    try:
        con = sqlite3.connect(db_path)
        try:
            r = con.execute("SELECT MAX(trade_date) FROM stock_daily").fetchone()
        finally:
            con.close()
        if r and r[0]:
            return pd.to_datetime(r[0])
    except Exception as e:
        logger.warning("读取 stock_daily 最新日期失败: %s", e)
    return None


def get_parquet_latest_date(cache_path: str) -> pd.Timestamp | None:
    """轻量读取 parquet 的 trade_date 最大值（pyarrow 只读一列，避免加载全量）。"""
    try:
        import pyarrow.parquet as pq
        if os.path.exists(cache_path):
            tab = pq.read_table(cache_path, columns=["trade_date"])
            return pd.to_datetime(tab.column("trade_date").to_pylist()).max()
    except Exception as e:
        logger.warning("读取 parquet 最新日期失败: %s", e)
    return None
