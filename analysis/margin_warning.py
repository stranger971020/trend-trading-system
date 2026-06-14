"""
融资背离预警
- 用 Tushare margin_detail 获取融资余额
- 行业融资余额下降 + 价格仍涨 → 融资背离 = 见顶信号增强
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
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
)

logger = logging.getLogger(__name__)

CREATE_MARGIN_TABLE = """
CREATE TABLE IF NOT EXISTS margin_cache (
    trade_date TEXT NOT NULL,
    ts_code    TEXT NOT NULL,
    rzye       REAL,
    rzmre      REAL,
    PRIMARY KEY (trade_date, ts_code)
);
"""


def _init_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_MARGIN_TABLE)
    conn.commit()
    conn.close()


def fetch_today_margin() -> bool:
    """拉取今日（最新交易日）全市场融资数据。"""
    _init_table()
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)

    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    for attempt in range(3):
        trade_date = (beijing_now - timedelta(days=attempt)).strftime("%Y%m%d")

        # 检查是否已有
        cur = conn.execute("SELECT COUNT(*) FROM margin_cache WHERE trade_date=?", (trade_date,))
        if cur.fetchone()[0] > 100:
            conn.close()
            return True

        try:
            df = pro.margin_detail(trade_date=trade_date)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR IGNORE INTO margin_cache VALUES (?,?,?,?)",
                        (str(row["trade_date"]), str(row["ts_code"]),
                         row.get("rzye", 0), row.get("rzmre", 0))
                    )
                conn.commit()
                logger.info("融资数据: %s → %d 条", trade_date, len(df))
                conn.close()
                return True
        except Exception as e:
            logger.warning("margin_detail %s 失败: %s", trade_date, e)
            time.sleep(1)

    conn.close()
    return False


def detect_margin_divergence(
    daily_df: pd.DataFrame,
    stock_mapping: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """检测行业融资背离。

    Returns:
        [{"industry": "电子", "price_change": 3.2, "margin_change": -5.1}, ...]
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT DISTINCT trade_date FROM margin_cache ORDER BY trade_date DESC LIMIT 30")
    dates = [r[0] for r in cur.fetchall()]

    if len(dates) < 5:
        conn.close()
        return []

    recent_5 = dates[:5]
    prev_20 = dates[5:20] if len(dates) > 5 else dates[5:]

    # 构建 stock→industry 映射
    stock_to_l1 = {}
    if stock_mapping:
        for code, info in stock_mapping.items():
            stock_to_l1[code] = info.get("l1_code", "")

    # 按行业汇总融资余额
    cur2 = conn.execute(f"""
        SELECT mc.ts_code, mc.trade_date, mc.rzye
        FROM margin_cache mc
        WHERE mc.trade_date IN ({','.join(['?' for _ in dates])})
    """, dates)

    # 汇总
    industry_margin: dict[str, list[float]] = {}
    for ts_code, td, rzye in cur2.fetchall():
        l1 = stock_to_l1.get(ts_code, "")
        if not l1:
            continue
        if l1 not in industry_margin:
            industry_margin[l1] = {"recent": [], "prev": []}
        if td in recent_5:
            industry_margin[l1]["recent"].append(rzye or 0)
        elif td in prev_20:
            industry_margin[l1]["prev"].append(rzye or 0)

    conn.close()

    # 检测背离
    divergences = []
    for l1_code, data in industry_margin.items():
        if not data["recent"] or not data["prev"]:
            continue
        recent_avg = sum(data["recent"]) / len(data["recent"])
        prev_avg = sum(data["prev"]) / len(data["prev"])
        if prev_avg > 0:
            margin_change = (recent_avg / prev_avg - 1) * 100
        else:
            continue

        # 获取行业价格变化
        ind_df = daily_df[daily_df[COL_TS_CODE] == l1_code].sort_values(COL_TRADE_DATE)
        if len(ind_df) >= 6:
            price_change = (ind_df[COL_CLOSE].iloc[-1] / ind_df[COL_CLOSE].iloc[-6] - 1) * 100
        else:
            price_change = 0

        # 融资降 + 价格涨 = 背离
        if margin_change < -3 and price_change > 0:
            divergences.append({
                "industry_code": l1_code,
                "margin_change": round(margin_change, 1),
                "price_change": round(price_change, 1),
            })

    if divergences:
        logger.info("融资背离: %d 个行业", len(divergences))
    return divergences
