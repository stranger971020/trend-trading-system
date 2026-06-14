"""
基本面多因子模块
- PE/PB 分位数（行业内比较）
- ROE 质量因子
- 数据缓存到 SQLite
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import tushare as ts

from config import (
    TUSHARE_TOKEN,
    DB_PATH,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

# 建表
CREATE_FUNDA_TABLE = """
CREATE TABLE IF NOT EXISTS fundamental_cache (
    ts_code      TEXT NOT NULL,
    trade_date   TEXT NOT NULL,
    pe_ttm       REAL,
    pb           REAL,
    roe          REAL,
    roe_quarter  TEXT,
    PRIMARY KEY (ts_code, trade_date)
);
"""


def _init_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_FUNDA_TABLE)
    conn.commit()
    conn.close()


def fetch_daily_basic_batch(stock_codes: list[str]) -> pd.DataFrame:
    """批量获取个股每日估值数据并缓存。

    Returns:
        DataFrame with ts_code, trade_date, pe_ttm, pb
    """
    _init_table()
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)

    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    end_date = beijing_now.strftime("%Y%m%d")
    start_date = (beijing_now - timedelta(days=10)).strftime("%Y%m%d")

    all_data = []

    for idx, code in enumerate(stock_codes):
        # 检查缓存
        cur = conn.execute(
            "SELECT COUNT(*) FROM fundamental_cache WHERE ts_code=? AND trade_date>=?",
            (code, start_date),
        )
        if cur.fetchone()[0] >= 3:
            cur2 = conn.execute(
                "SELECT ts_code, trade_date, pe_ttm, pb, roe, roe_quarter FROM fundamental_cache WHERE ts_code=? AND trade_date>=? ORDER BY trade_date DESC",
                (code, start_date),
            )
            for row in cur2.fetchall():
                all_data.append({
                    "ts_code": row[0], "trade_date": row[1],
                    "pe_ttm": row[2], "pb": row[3],
                    "roe": row[4], "roe_quarter": row[5],
                })
            continue

        try:
            df = pro.daily_basic(ts_code=code, start_date=start_date, end_date=end_date,
                                 fields="ts_code,trade_date,pe_ttm,pb")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR IGNORE INTO fundamental_cache(ts_code,trade_date,pe_ttm,pb) VALUES (?,?,?,?)",
                        (code, str(row["trade_date"]), row.get("pe_ttm"), row.get("pb")),
                    )
                conn.commit()
                for _, row in df.iterrows():
                    all_data.append({
                        "ts_code": code, "trade_date": str(row["trade_date"]),
                        "pe_ttm": row.get("pe_ttm"), "pb": row.get("pb"),
                        "roe": None, "roe_quarter": None,
                    })
        except Exception as e:
            logger.debug("daily_basic %s 失败: %s", code, e)

        if idx < len(stock_codes) - 1:
            time.sleep(0.3)

    conn.close()
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def fetch_roe_batch(stock_codes: list[str]) -> dict[str, dict]:
    """批量获取最新 ROE 数据。

    Returns:
        {"000001.SZ": {"roe": 12.5, "quarter": "2026Q1", "roe_prev": 11.2}, ...}
    """
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)

    result = {}
    for idx, code in enumerate(stock_codes):
        try:
            df = pro.fina_indicator(ts_code=code, start_date="20250101", end_date="20260612",
                                     fields="ts_code,end_date,roe,roe_dt")
            if df is not None and not df.empty:
                df = df.sort_values("end_date", ascending=False)
                latest = df.iloc[0]
                roe = latest.get("roe")
                roe_dt = latest.get("roe_dt")

                # 更新缓存
                trade_date = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET))).strftime("%Y%m%d")
                conn.execute(
                    "UPDATE fundamental_cache SET roe=?, roe_quarter=? WHERE ts_code=? AND trade_date=?",
                    (roe, str(latest.get("end_date", "")), code, trade_date),
                )
                conn.commit()

                # ROE 趋势：对比上一季度
                roe_prev = float(df.iloc[1].get("roe", 0)) if len(df) > 1 else None
                trend = 0
                if roe is not None and roe_prev is not None and roe_prev != 0:
                    trend = (float(roe) - roe_prev) / abs(roe_prev)

                result[code] = {
                    "roe": round(float(roe), 2) if roe is not None else None,
                    "quarter": str(latest.get("end_date", "")),
                    "roe_trend": round(trend, 2),
                }
        except Exception as e:
            logger.debug("fina_indicator %s 失败: %s", code, e)

        if idx < len(stock_codes) - 1:
            time.sleep(0.3)

    conn.close()
    return result


def compute_fundamental_score(
    stock_picks: list[dict],
    stock_mapping: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """对精选个股计算基本面加分。

    策略:
    - 获取 PE/PB 数据
    - 在同 L3 行业内计算分位数
    - PE 分位越低 → 越便宜 → 加分 (max +1.0)
    - PB 分位越低 → 越便宜 → 加分 (max +0.5)
    - ROE 越高 → 质量越好 → 加分 (max +1.0)
    - ROE 趋势向上 → +0.3
    """
    if not stock_picks:
        return stock_picks

    codes = [p["ts_code"] for p in stock_picks]

    # 获取估值数据
    df_val = fetch_daily_basic_batch(codes)
    roe_data = fetch_roe_batch(codes)

    if df_val.empty and not roe_data:
        return stock_picks

    # 按 L3 行业分组计算 PE/PB 分位数
    pe_pct = {}
    pb_pct = {}
    if not df_val.empty and stock_mapping:
        latest = df_val.sort_values("trade_date").groupby("ts_code").last().reset_index()
        for code in latest["ts_code"]:
            l3 = stock_mapping.get(code, {}).get("l3_code", "")
            if not l3:
                continue
            l3_codes = [c for c, m in stock_mapping.items() if m.get("l3_code") == l3]
            l3_data = latest[latest["ts_code"].isin(l3_codes)]

            pe_vals = l3_data["pe_ttm"].dropna()
            pb_vals = l3_data["pb"].dropna()

            row = latest[latest["ts_code"] == code]
            if not row.empty and len(pe_vals) > 1:
                pe_val = row["pe_ttm"].values[0]
                if not pd.isna(pe_val) and pe_val > 0:
                    pe_pct[code] = (pe_vals < pe_val).sum() / len(pe_vals)

            if not row.empty and len(pb_vals) > 1:
                pb_val = row["pb"].values[0]
                if not pd.isna(pb_val) and pb_val > 0:
                    pb_pct[code] = (pb_vals < pb_val).sum() / len(pb_vals)

    # 应用加分
    for pick in stock_picks:
        code = pick["ts_code"]
        bonus = 0.0
        extra_info = {}

        # PE 分位加分
        if code in pe_pct:
            pct = pe_pct[code]
            pe_bonus = round((1 - pct) * 1.0, 2)  # 分位越低加分越多
            bonus += pe_bonus
            extra_info["pe_pct"] = round(pct * 100)

        # PB 分位加分
        if code in pb_pct:
            pct = pb_pct[code]
            pb_bonus = round((1 - pct) * 0.5, 2)
            bonus += pb_bonus
            extra_info["pb_pct"] = round(pct * 100)

        # ROE 加分
        roe_info = roe_data.get(code, {})
        roe = roe_info.get("roe")
        if roe is not None and roe > 0:
            # ROE 15%+ → +1.0, ROE 10% → +0.5, ROE 5% → +0.2
            roe_bonus = round(min(1.0, roe / 15 * 1.0), 2)
            bonus += roe_bonus
            extra_info["roe"] = roe

            # ROE 趋势
            trend = roe_info.get("roe_trend", 0)
            if trend > 0.05:
                bonus += 0.3
                extra_info["roe_trend"] = "↑"
            elif trend < -0.05:
                extra_info["roe_trend"] = "↓"
            else:
                extra_info["roe_trend"] = "→"

        # 总分最多 +3.0
        bonus = min(3.0, bonus)
        extra_info["funda_bonus"] = round(bonus, 2)

        old_score = pick.get("score", 0)
        pick["score"] = round(old_score + bonus, 2)
        pick["fundamental"] = extra_info

    logger.info("基本面因子: %d 只个股评分已调整", len(stock_picks))
    return stock_picks
