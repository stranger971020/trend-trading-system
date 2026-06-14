"""
资金流过滤器
- 用 Tushare moneyflow 接口获取个股主力资金净流向
- 近5日累计净流入 → 加分；净流出 → 降权
- 数据缓存到 SQLite
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
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

CREATE_MF_TABLE = """
CREATE TABLE IF NOT EXISTS moneyflow_cache (
    ts_code     TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    net_mf_amount REAL,
    buy_elg_amount REAL,
    sell_elg_amount REAL,
    PRIMARY KEY (ts_code, trade_date)
);
"""


def _init_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_MF_TABLE)
    conn.commit()
    conn.close()


def fetch_and_cache_moneyflow(stock_codes: list[str], lookback: int = 5) -> dict[str, float]:
    """获取并缓存个股资金流向。

    Returns:
        {"000001.SZ": 12345.6, ...}  # net_mf_amount 近N日累计（万元）
    """
    _init_table()
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)
    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    end_date = beijing_now.strftime("%Y%m%d")
    start_date = (beijing_now - timedelta(days=lookback * 2)).strftime("%Y%m%d")

    result = {}
    for idx, code in enumerate(stock_codes):
        # 检查缓存
        cur = conn.execute(
            "SELECT COUNT(*) FROM moneyflow_cache WHERE ts_code=? AND trade_date>=?",
            (code, start_date)
        )
        cached_count = cur.fetchone()[0]

        if cached_count >= lookback:
            # 从缓存读取
            cur2 = conn.execute(
                "SELECT SUM(net_mf_amount) FROM moneyflow_cache WHERE ts_code=? AND trade_date>=?",
                (code, start_date)
            )
            row = cur2.fetchone()
            result[code] = row[0] or 0.0
            continue

        # 调取 API
        try:
            df = pro.moneyflow(ts_code=code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR IGNORE INTO moneyflow_cache VALUES (?,?,?,?,?)",
                        (code, str(row["trade_date"]), row.get("net_mf_amount", 0),
                         row.get("buy_elg_amount", 0), row.get("sell_elg_amount", 0))
                    )
                conn.commit()
                result[code] = float(df["net_mf_amount"].sum())
            else:
                result[code] = 0.0
        except Exception as e:
            logger.debug("moneyflow %s 失败: %s", code, e)
            result[code] = 0.0

        if idx < len(stock_codes) - 1:
            time.sleep(0.35)

    conn.close()
    positive = sum(1 for v in result.values() if v > 0)
    logger.info("资金流: %d/%d 只个股主力净流入", positive, len(result))
    return result


def apply_moneyflow_filter(stock_picks: list[dict], moneyflow: dict[str, float]) -> list[dict]:
    """对精选个股应用资金流评分调整。"""
    for pick in stock_picks:
        code = pick["ts_code"]
        net_flow = moneyflow.get(code, 0)
        if net_flow > 0:
            pick["score"] = round(pick["score"] + 0.5, 2)
            pick["moneyflow"] = "inflow"
        elif net_flow < 0:
            pick["score"] = round(max(0, pick["score"] - 0.3), 2)
            pick["moneyflow"] = "outflow"
        else:
            pick["moneyflow"] = "neutral"
        pick["net_mf_amount"] = round(net_flow, 2)
    return stock_picks
