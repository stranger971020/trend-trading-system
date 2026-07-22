from __future__ import annotations
"""
申万三级行业日线数据获取与存储
- 通过 Tushare index_classify 获取 346 个 L3 代码+父级关系
- 通过 akshare index_hist_sw() 获取日线 OHLCV
- 自动过滤无近期数据的陈旧代码
- 存储到 SQLite (sw_index_data.db → sw_l3_index_daily)
"""

import logging
import sqlite3
import time

import akshare as ak
import pandas as pd
import tushare as ts

from config import (
    TUSHARE_TOKEN,
    DB_PATH,
    AK_API_RATE_LIMIT,
    API_RETRY_COUNT,
    API_RETRY_DELAY,
    L3_TABLE,
    L3_MIN_RECENT_DATE,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOL,
    COL_AMOUNT,
    COL_NAME,
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

CREATE_L3_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {L3_TABLE} (
    trade_date  TEXT    NOT NULL,
    ts_code     TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    parent_l1   TEXT,
    parent_name TEXT,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    vol         REAL,
    amount      REAL,
    PRIMARY KEY (trade_date, ts_code)
);
"""

CREATE_L3_INDEX_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_l3_ts_code ON {L3_TABLE}(ts_code);",
    f"CREATE INDEX IF NOT EXISTS idx_l3_trade_date ON {L3_TABLE}(trade_date);",
    f"CREATE INDEX IF NOT EXISTS idx_l3_parent ON {L3_TABLE}(parent_l1);",
]


def _ts_code_to_symbol(ts_code: str) -> str:
    return ts_code.replace(".SI", "").replace(".SH", "").replace(".SZ", "")


def init_l3_table(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_L3_TABLE_SQL)
    for idx_sql in CREATE_L3_INDEX_SQL:
        conn.execute(idx_sql)
    conn.commit()
    conn.close()


def load_l3_mapping() -> dict[str, dict]:
    """从 Tushare 获取三级行业列表及父级关系。

    Returns:
        {"850111.SI": {"name": "种子", "parent_l1": "801010.SI", "parent_name": "农林牧渔"}, ...}
    """
    pro = ts.pro_api(TUSHARE_TOKEN)

    # 获取 L3 列表
    df_l3 = pro.index_classify(level="L3", src="SW2021")
    # 获取 L1 列表（用于解析 parent_name）
    df_l1 = pro.index_classify(level="L1", src="SW2021")
    l1_names = {}
    if df_l1 is not None:
        for _, row in df_l1.iterrows():
            l1_names[str(row["index_code"])] = str(row["industry_name"])

    mapping = {}
    if df_l3 is None or df_l3.empty:
        logger.error("Tushare index_classify(L3) 返回空")
        return mapping

    for _, row in df_l3.iterrows():
        code = str(row.get("index_code", ""))
        name = str(row.get("industry_name", ""))
        parent = str(row.get("parent_code", ""))
        if not code or not name:
            continue

        # 解析 parent_l1: parent_code 可能指向 L2，需要向上追溯到 L1
        # SW2021 中 L3 的 parent_code 是 L2 代码。我们通过 stock_industry_mapping 来获取 L1
        # 简化处理: 直接存 parent_code，后续从个股映射中补充 L1 关系
        parent_name = l1_names.get(parent, "")

        mapping[code] = {
            "name": name,
            "parent_code": parent,
            "parent_name": parent_name,
        }

    # 从 stock_industry_mapping 补充 L3→L1 关系
    _enrich_l3_parent_from_stock_mapping(mapping)

    logger.info("加载 %d 个 SW L3 行业映射", len(mapping))
    return mapping


def _enrich_l3_parent_from_stock_mapping(l3_mapping: dict) -> None:
    """从已有的 stock_industry_mapping.csv 提取 L3→L1 关系。"""
    import csv
    import os
    from config import DATA_DIR

    csv_path = os.path.join(DATA_DIR, "stock_industry_mapping.csv")
    if not os.path.exists(csv_path):
        return

    # 读取 stock→L3→L1 关系，构建 L3→L1 映射
    l3_to_l1: dict[str, tuple[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l3 = row.get("l3_code", "")
            l1_code = row.get("l1_code", "")
            l1_name = row.get("l1_name", "")
            if l3 and l1_code:
                l3_to_l1[l3] = (l1_code, l1_name)

    for l3_code in l3_mapping:
        if l3_code in l3_to_l1:
            l3_mapping[l3_code]["parent_l1"] = l3_to_l1[l3_code][0]
            l3_mapping[l3_code]["parent_name"] = l3_to_l1[l3_code][1]


def fetch_l3_daily(ts_code: str, name: str) -> pd.DataFrame | None:
    """通过 akshare 获取单个三级行业的全量日线数据。"""
    symbol = _ts_code_to_symbol(ts_code)

    for attempt in range(1, API_RETRY_COUNT + 1):
        try:
            df_raw = ak.index_hist_sw(symbol=symbol, period="day")
            if df_raw is None or df_raw.empty:
                return None

            df = df_raw.rename(columns=_AKSHARE_COL_MAP)
            df[COL_TS_CODE] = ts_code
            df[COL_NAME] = name
            std_cols = [COL_TRADE_DATE, COL_TS_CODE, COL_NAME,
                        COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOL, COL_AMOUNT]
            df = df[std_cols].copy()
            df[COL_TRADE_DATE] = df[COL_TRADE_DATE].astype(str).str.replace("-", "")
            return df
        except Exception as e:
            logger.warning("%s 第 %d/%d 次获取失败: %s", ts_code, attempt, API_RETRY_COUNT, e)
            if attempt < API_RETRY_COUNT:
                time.sleep(API_RETRY_DELAY)

    return None


def fetch_and_store_l3(
    db_path: str = DB_PATH,
    l3_mapping: dict[str, dict] | None = None,
) -> dict:
    """获取所有三级行业数据并存储。

    自动跳过数据陈旧（最新日期 < L3_MIN_RECENT_DATE）的代码。

    Returns:
        {"total": N, "active": M, "stale": S, "new_rows": R}
    """
    if l3_mapping is None:
        l3_mapping = load_l3_mapping()

    init_l3_table(db_path)
    conn = sqlite3.connect(db_path)

    active = 0
    stale = 0
    total_new_rows = 0
    codes = sorted(l3_mapping.keys())

    for idx, ts_code in enumerate(codes):
        info = l3_mapping[ts_code]
        name = info.get("name", ts_code)
        parent_l1 = info.get("parent_l1", "")
        parent_name = info.get("parent_name", "")

        # 检查是否已有数据
        cur = conn.execute(
            f"SELECT MAX(trade_date) FROM {L3_TABLE} WHERE ts_code = ?",
            (ts_code,),
        )
        latest = cur.fetchone()[0]

        df = fetch_l3_daily(ts_code, name)

        if df is None or df.empty:
            continue

        # 检查数据新鲜度
        df_max_date = df[COL_TRADE_DATE].max()
        if df_max_date < L3_MIN_RECENT_DATE:
            stale += 1
            if idx < 5:
                logger.debug("%s (%s): 数据过旧 (最新=%s), 跳过", ts_code, name, df_max_date)
            continue

        # 过滤已存在的日期
        if latest:
            df_new = df[df[COL_TRADE_DATE] > latest]
        else:
            df_new = df

        if df_new.empty:
            active += 1
            continue

        # 添加父级信息
        df_new["parent_l1"] = parent_l1
        df_new["parent_name"] = parent_name

        # 写入
        write_cols = [COL_TRADE_DATE, COL_TS_CODE, COL_NAME, "parent_l1", "parent_name",
                      COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOL, COL_AMOUNT]
        df_write = df_new[write_cols].copy()

        try:
            df_write.to_sql(L3_TABLE, conn, if_exists="append", index=False)
        except Exception:
            placeholders = ", ".join(["?" for _ in write_cols])
            sql = f"INSERT OR IGNORE INTO {L3_TABLE} ({', '.join(write_cols)}) VALUES ({placeholders})"
            for _, row in df_write.iterrows():
                conn.execute(sql, [row[c] for c in write_cols])

        conn.commit()
        new_rows = len(df_new)
        total_new_rows += new_rows
        active += 1

        if idx == 0 or new_rows > 10:
            logger.info(
                "[%d/%d] %s (%s): +%d 条, 父级=%s",
                idx + 1, len(codes), ts_code, name, new_rows, parent_name or parent_l1,
            )

        time.sleep(AK_API_RATE_LIMIT)

    conn.close()

    result = {
        "total": len(codes),
        "active": active,
        "stale": stale,
        "new_rows": total_new_rows,
    }
    logger.info(
        "L3 数据完成: %d 个代码, %d 活跃, %d 陈旧, +%d 条",
        len(codes), active, stale, total_new_rows,
    )
    return result


def load_l3_daily(
    db_path: str = DB_PATH,
    min_rows: int = 20,
) -> pd.DataFrame:
    """从数据库加载三级行业日线数据。"""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT trade_date, ts_code, name, parent_l1, parent_name,
               open, high, low, close, vol, amount
        FROM {L3_TABLE}
        ORDER BY ts_code, trade_date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    counts = df.groupby(COL_TS_CODE).size()
    valid_codes = counts[counts >= min_rows].index.tolist()
    return df[df[COL_TS_CODE].isin(valid_codes)].copy()
