"""
申万一级行业日线数据获取与存储
- 通过 akshare index_hist_sw() 获取 31 个 SW L1 指数日线 OHLCV
- 增量更新：只插入数据库中尚不存在的日期
- 存储到 SQLite (sw_index_data.db → sw_index_daily)
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import akshare as ak
import pandas as pd

from config import (
    DB_PATH,
    API_RATE_LIMIT,
    API_RETRY_COUNT,
    API_RETRY_DELAY,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOL,
    COL_AMOUNT,
    COL_NAME,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

# akshare → 内部列名映射
_AKSHARE_COL_MAP = {
    "日期": COL_TRADE_DATE,
    "代码": "ak_code",
    "收盘": COL_CLOSE,
    "开盘": COL_OPEN,
    "最高": COL_HIGH,
    "最低": COL_LOW,
    "成交量": COL_VOL,
    "成交额": COL_AMOUNT,
}

# 建表 SQL
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS sw_index_daily (
    trade_date  TEXT    NOT NULL,
    ts_code     TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    vol         REAL,
    amount      REAL,
    PRIMARY KEY (trade_date, ts_code)
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_sw_ts_code ON sw_index_daily(ts_code);",
    "CREATE INDEX IF NOT EXISTS idx_sw_trade_date ON sw_index_daily(trade_date);",
]


def _ts_code_to_symbol(ts_code: str) -> str:
    """将 Tushare 格式的代码 (801010.SI) 转为 akshare 格式 (801010)。"""
    return ts_code.replace(".SI", "").replace(".SH", "").replace(".SZ", "")


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """初始化数据库，创建表和索引。"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_TABLE_SQL)
    for idx_sql in CREATE_INDEX_SQL:
        conn.execute(idx_sql)
    conn.commit()
    logger.info("数据库已初始化: %s", db_path)
    return conn


def get_latest_date_for_code(
    conn: sqlite3.Connection, ts_code: str
) -> str | None:
    """查询某个行业代码在数据库中的最新交易日。"""
    cur = conn.execute(
        "SELECT MAX(trade_date) FROM sw_index_daily WHERE ts_code = ?",
        (ts_code,),
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def fetch_single_code_akshare(ts_code: str, name: str) -> pd.DataFrame | None:
    """通过 akshare index_hist_sw 获取单个行业的历史日线数据。

    Args:
        ts_code: Tushare 格式代码 (如 "801010.SI")
        name: 行业中文名

    Returns:
        标准化后的 DataFrame，或 None
    """
    symbol = _ts_code_to_symbol(ts_code)

    for attempt in range(1, API_RETRY_COUNT + 1):
        try:
            df_raw = ak.index_hist_sw(symbol=symbol, period="day")
            if df_raw is None or df_raw.empty:
                logger.warning("%s (%s): akshare 返回空数据", ts_code, name)
                return None

            # 重命名列为内部标准
            df = df_raw.rename(columns=_AKSHARE_COL_MAP)
            df[COL_TS_CODE] = ts_code
            df[COL_NAME] = name

            # 只保留标准列
            std_cols = [COL_TRADE_DATE, COL_TS_CODE, COL_NAME,
                        COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOL, COL_AMOUNT]
            df = df[std_cols].copy()

            # 确保 trade_date 为字符串格式 YYYYMMDD
            df[COL_TRADE_DATE] = df[COL_TRADE_DATE].astype(str).str.replace("-", "")

            return df

        except Exception as e:
            logger.warning(
                "%s 第 %d/%d 次获取失败: %s", ts_code, attempt, API_RETRY_COUNT, e
            )
            if attempt < API_RETRY_COUNT:
                time.sleep(API_RETRY_DELAY)

    logger.error("%s 获取彻底失败（已重试 %d 次）", ts_code, API_RETRY_COUNT)
    return None


def fetch_and_store_incremental(
    conn: sqlite3.Connection,
    ts_codes: dict[str, str],
) -> dict[str, int]:
    """增量获取并存储所有 SW L1 行业的日线数据。

    策略（akshare 版本）：
    - akshare 返回全量历史数据，不做日期范围查询
    - 将返回的 DataFrame 与数据库已有数据比较
    - 只 INSERT 数据库中不存在的 (trade_date, ts_code) 组合

    Args:
        conn: SQLite 连接
        ts_codes: {"801010.SI": "农林牧渔", ...}

    Returns:
        dict: {"801010.SI": 新增记录数, ...}
    """
    summary: dict[str, int] = {}

    for idx, (ts_code, name) in enumerate(sorted(ts_codes.items()), 1):
        logger.info("[%d/%d] 获取 %s (%s)", idx, len(ts_codes), ts_code, name)

        df = fetch_single_code_akshare(ts_code, name)

        if df is None or df.empty:
            summary[ts_code] = 0
            logger.warning("  ✗ %s: 数据获取失败", ts_code)
        else:
            rows_before = _count_rows_for_code(conn, ts_code)

            # 只插入数据库中尚不存在的 (trade_date, ts_code)
            existing_dates = _get_existing_dates(conn, ts_code)
            df_new = df[~df[COL_TRADE_DATE].isin(existing_dates)]

            if df_new.empty:
                summary[ts_code] = 0
                logger.info("  ✓ %s: 数据已是最新 (%d 条历史)", ts_code, len(df))
            else:
                try:
                    df_new.to_sql(
                        "sw_index_daily", conn, if_exists="append", index=False
                    )
                except Exception as e:
                    logger.warning("  %s 批量写入失败: %s，逐行插入", ts_code, e)
                    _insert_rows_individually(conn, df_new)

                conn.commit()
                rows_after = _count_rows_for_code(conn, ts_code)
                new_rows = max(0, rows_after - rows_before)
                summary[ts_code] = new_rows
                logger.info(
                    "  ✓ %s: +%d 条新记录 (总计 %d 条)",
                    ts_code, new_rows, rows_after,
                )

        # 限速
        if idx < len(ts_codes):
            time.sleep(API_RATE_LIMIT)

    total_new = sum(summary.values())
    logger.info(
        "数据更新完成: %d/%d 个行业有新数据，共 %d 条新记录",
        sum(1 for v in summary.values() if v > 0),
        len(ts_codes),
        total_new,
    )
    return summary


def _get_existing_dates(conn: sqlite3.Connection, ts_code: str) -> set[str]:
    """获取数据库中某代码已有的所有 trade_date。"""
    cur = conn.execute(
        "SELECT trade_date FROM sw_index_daily WHERE ts_code = ?", (ts_code,)
    )
    return {row[0] for row in cur.fetchall()}


def _count_rows_for_code(conn: sqlite3.Connection, ts_code: str) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) FROM sw_index_daily WHERE ts_code = ?", (ts_code,)
    )
    return cur.fetchone()[0]


def _insert_rows_individually(
    conn: sqlite3.Connection, df: pd.DataFrame
) -> None:
    """逐行 INSERT OR IGNORE（当 to_sql 批量插入失败时的回退方案）。"""
    cols = [COL_TRADE_DATE, COL_TS_CODE, COL_NAME, COL_OPEN, COL_HIGH,
            COL_LOW, COL_CLOSE, COL_VOL, COL_AMOUNT]
    placeholders = ", ".join(["?" for _ in cols])
    sql = (
        f"INSERT OR IGNORE INTO sw_index_daily "
        f"({', '.join(cols)}) VALUES ({placeholders})"
    )
    for _, row in df.iterrows():
        values = [
            str(row.get(COL_TRADE_DATE, "")),
            str(row.get(COL_TS_CODE, "")),
            str(row.get(COL_NAME, "")),
            row.get(COL_OPEN),
            row.get(COL_HIGH),
            row.get(COL_LOW),
            row.get(COL_CLOSE),
            row.get(COL_VOL),
            row.get(COL_AMOUNT),
        ]
        conn.execute(sql, values)
    conn.commit()


def load_daily_data(
    conn: sqlite3.Connection,
    ts_codes: list[str],
    min_rows: int = 20,
) -> pd.DataFrame:
    """从数据库加载指定行业代码的日线数据。

    Args:
        conn: SQLite 连接
        ts_codes: 需要加载的行业代码列表（如 ["801010.SI", ...]）
        min_rows: 每个代码至少需要的数据行数

    Returns:
        DataFrame，按 ts_code, trade_date 排序
    """
    if not ts_codes:
        return pd.DataFrame()

    placeholders = ", ".join(["?" for _ in ts_codes])
    query = f"""
        SELECT trade_date, ts_code, name, open, high, low, close, vol, amount
        FROM sw_index_daily
        WHERE ts_code IN ({placeholders})
        ORDER BY ts_code, trade_date ASC
    """
    df = pd.read_sql_query(query, conn, params=ts_codes)

    # 过滤数据不足的行业
    counts = df.groupby(COL_TS_CODE).size()
    valid_codes = counts[counts >= min_rows].index.tolist()
    if len(valid_codes) < len(ts_codes):
        dropped = set(ts_codes) - set(valid_codes)
        logger.warning("以下行业数据不足 %d 条，已排除: %s", min_rows, dropped)

    return df[df[COL_TS_CODE].isin(valid_codes)].copy()


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """获取数据库连接（带 row_factory）。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
