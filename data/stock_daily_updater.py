"""
个股日线数据获取与存储
- 通过 Tushare pro.daily() 获取个股日线 OHLCV
- 增量更新：只拉取每个股票缺失的交易日
- 存储到 SQLite (sw_index_data.db → stock_daily)
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import tushare as ts

from config import (
    TUSHARE_TOKEN,
    DB_PATH,
    STOCK_INITIAL_FETCH_DAYS,
    STOCK_API_RATE_LIMIT,
    API_RETRY_COUNT,
    API_RETRY_DELAY,
    MAX_STOCKS_PER_INDUSTRY,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOL,
    COL_AMOUNT,
    COL_PCT_CHG,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=BEIJING_TZ_OFFSET))

# 建表 SQL
CREATE_STOCK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    trade_date  TEXT    NOT NULL,
    ts_code     TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    pre_close   REAL,
    pct_chg     REAL,
    vol         REAL,
    amount      REAL,
    PRIMARY KEY (trade_date, ts_code)
);
"""

CREATE_STOCK_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_sd_ts_code ON stock_daily(ts_code);",
    "CREATE INDEX IF NOT EXISTS idx_sd_trade_date ON stock_daily(trade_date);",
]


def init_stock_table(db_path: str = DB_PATH) -> None:
    """确保 stock_daily 表存在。"""
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_STOCK_TABLE_SQL)
    for idx_sql in CREATE_STOCK_INDEX_SQL:
        conn.execute(idx_sql)
    conn.commit()
    conn.close()


def _beijing_date_str() -> str:
    return datetime.now(_BEIJING_TZ).strftime("%Y%m%d")


def _get_existing_dates(conn: sqlite3.Connection, ts_code: str) -> set[str]:
    cur = conn.execute(
        "SELECT trade_date FROM stock_daily WHERE ts_code = ?", (ts_code,)
    )
    return {row[0] for row in cur.fetchall()}


def _count_stock_rows(conn: sqlite3.Connection, ts_code: str) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) FROM stock_daily WHERE ts_code = ?", (ts_code,)
    )
    return cur.fetchone()[0]


def fetch_stock_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """通过 Tushare pro.daily() 获取单只个股日线数据。

    Args:
        ts_code: 股票代码，如 "000001.SZ"
        start_date: 起始日期 YYYYMMDD
        end_date: 截止日期 YYYYMMDD

    Returns:
        DataFrame with columns: ts_code, trade_date, open, high, low,
        close, pre_close, pct_chg, vol, amount
    """
    pro = ts.pro_api(TUSHARE_TOKEN)

    for attempt in range(1, API_RETRY_COUNT + 1):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                return df
            else:
                logger.debug("%s: %s→%s 无数据", ts_code, start_date, end_date)
                return None
        except Exception as e:
            logger.warning(
                "%s 第 %d/%d 次获取失败: %s", ts_code, attempt, API_RETRY_COUNT, e
            )
            if attempt < API_RETRY_COUNT:
                time.sleep(API_RETRY_DELAY)

    logger.error("%s 获取彻底失败", ts_code)
    return None


def update_stocks_for_industries(
    db_path: str,
    industry_stocks: dict[str, list[str]],
    recent_only: bool = False,
) -> dict:
    """为指定行业的个股获取并存储日线数据。

    策略：
    - 首次运行：拉取 STOCK_INITIAL_FETCH_DAYS 个交易日（约1年）
    - 后续运行：只拉取数据库中不存在的日期
    - 每行业最多取 MAX_STOCKS_PER_INDUSTRY 只（按代码排序）

    Args:
        db_path: 数据库路径
        industry_stocks: {"801780.SI": ["000001.SZ", ...], ...}
        recent_only: True 则只拉最近 15 天缺失数据

    Returns:
        {"total_stocks": N, "updated": M, "new_rows": R}
    """
    init_stock_table(db_path)
    conn = sqlite3.connect(db_path)
    end_date = _beijing_date_str()

    total_stocks = 0
    updated_stocks = 0
    total_new_rows = 0

    for l1_code, stock_codes in industry_stocks.items():
        # 每行业限制数量
        selected = sorted(stock_codes)[:MAX_STOCKS_PER_INDUSTRY]
        logger.info(
            "行业 %s: %d 只成分股，选取 %d 只",
            l1_code, len(stock_codes), len(selected),
        )

        for idx, ts_code in enumerate(selected):
            total_stocks += 1
            existing = _get_existing_dates(conn, ts_code)

            if existing:
                latest = max(existing)
                start_dt = datetime.strptime(latest, "%Y%m%d") + timedelta(days=1)
                start_date = start_dt.strftime("%Y%m%d")
            else:
                # 首次获取，回溯交易日
                start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(
                    days=int(STOCK_INITIAL_FETCH_DAYS * 1.6)
                )
                start_date = start_dt.strftime("%Y%m%d")

            if start_date > end_date:
                continue

            df = fetch_stock_daily(ts_code, start_date, end_date)

            if df is not None and not df.empty:
                # 过滤已存在日期
                df["trade_date"] = df["trade_date"].astype(str)
                df_new = df[~df["trade_date"].isin(existing)]

                if not df_new.empty:
                    # 标准化列
                    cols = ["ts_code", "trade_date", "open", "high", "low",
                            "close", "pre_close", "pct_chg", "vol", "amount"]
                    df_write = df_new[cols].copy()
                    try:
                        df_write.to_sql(
                            "stock_daily", conn, if_exists="append", index=False
                        )
                    except Exception:
                        # 逐行插入
                        placeholders = ", ".join(["?" for _ in cols])
                        sql = (
                            f"INSERT OR IGNORE INTO stock_daily "
                            f"({', '.join(cols)}) VALUES ({placeholders})"
                        )
                        for _, row in df_write.iterrows():
                            conn.execute(sql, [row[c] for c in cols])

                    conn.commit()
                    new_rows = len(df_new)
                    total_new_rows += new_rows
                    updated_stocks += 1

                    if idx == 0 or new_rows > 10:
                        logger.info(
                            "  %s: +%d 条 (%d 条历史)",
                            ts_code, new_rows, len(existing) + new_rows,
                        )

            # 限速
            time.sleep(STOCK_API_RATE_LIMIT)

    conn.close()
    result = {
        "total_stocks": total_stocks,
        "updated": updated_stocks,
        "new_rows": total_new_rows,
    }
    logger.info(
        "个股数据: %d 只股票, %d 只有更新, +%d 条新记录",
        total_stocks, updated_stocks, total_new_rows,
    )
    return result


def fetch_all_stocks(
    db_path: str,
    stock_codes: list[str],
) -> dict:
    """全量拉取所有个股日线数据（首次慢，后续增量快）。

    Args:
        db_path: 数据库路径
        stock_codes: 全部需要拉取的股票代码（如 3000 只）

    Returns:
        {"total_stocks": N, "updated": M, "new_rows": R}
    """
    init_stock_table(db_path)
    conn = sqlite3.connect(db_path)
    end_date = _beijing_date_str()

    total_new_rows = 0
    updated_stocks = 0

    for idx, ts_code in enumerate(stock_codes):
        existing = _get_existing_dates(conn, ts_code)
        if existing:
            latest = max(existing)
            start_dt = datetime.strptime(latest, "%Y%m%d") + timedelta(days=1)
            start_date = start_dt.strftime("%Y%m%d")
        else:
            start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=int(STOCK_INITIAL_FETCH_DAYS * 1.6))
            start_date = start_dt.strftime("%Y%m%d")

        if start_date > end_date:
            continue

        df = fetch_stock_daily(ts_code, start_date, end_date)
        if df is not None and not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            df_new = df[~df["trade_date"].isin(existing)]
            if not df_new.empty:
                cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
                df_write = df_new[cols].copy()
                try:
                    df_write.to_sql("stock_daily", conn, if_exists="append", index=False)
                except Exception:
                    placeholders = ", ".join(["?" for _ in cols])
                    sql = f"INSERT OR IGNORE INTO stock_daily ({', '.join(cols)}) VALUES ({placeholders})"
                    for _, row in df_write.iterrows():
                        conn.execute(sql, [row[c] for c in cols])
                conn.commit()
                total_new_rows += len(df_new)
                updated_stocks += 1

        if (idx + 1) % 100 == 0:
            pct = (idx + 1) / len(stock_codes) * 100
            logger.info("  全量进度: %d/%d (%.0f%%), +%d 条", idx + 1, len(stock_codes), pct, total_new_rows)

        time.sleep(STOCK_API_RATE_LIMIT)

    conn.close()
    result = {"total_stocks": len(stock_codes), "updated": updated_stocks, "new_rows": total_new_rows}
    logger.info("全量个股: %d 只, %d 更新, +%d 条", len(stock_codes), updated_stocks, total_new_rows)
    return result


def load_stock_daily(
    db_path: str,
    stock_codes: list[str],
    min_rows: int = 10,
) -> pd.DataFrame:
    """从数据库加载个股日线数据。

    Args:
        db_path: 数据库路径
        stock_codes: 需要加载的股票代码列表
        min_rows: 每只股票至少需要的数据行数

    Returns:
        DataFrame，按 ts_code, trade_date 排序
    """
    if not stock_codes:
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)

    # 分批查询以避免 SQL 过长
    batch_size = 500
    frames = []
    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i + batch_size]
        placeholders = ", ".join(["?" for _ in batch])
        query = f"""
            SELECT trade_date, ts_code, open, high, low, close,
                   pre_close, pct_chg, vol, amount
            FROM stock_daily
            WHERE ts_code IN ({placeholders})
            ORDER BY ts_code, trade_date ASC
        """
        df = pd.read_sql_query(query, conn, params=batch)
        if not df.empty:
            frames.append(df)

    conn.close()

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # 过滤数据不足的股票
    counts = df_all.groupby(COL_TS_CODE).size()
    valid_codes = counts[counts >= min_rows].index.tolist()
    if len(valid_codes) < len(stock_codes):
        dropped = set(stock_codes) - set(valid_codes)
        logger.debug("以下个股数据不足 %d 条，已排除: %d 只", min_rows, len(dropped))

    return df_all[df_all[COL_TS_CODE].isin(valid_codes)].copy()
