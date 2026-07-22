#!/usr/bin/env python3
"""初始化市场情绪历史缓存 - 拉取历史融资融券数据用于分位计算"""
import os, sys, json, logging
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TUSHARE_TOKEN, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("init_sentiment")
CACHE_PATH = os.path.join(DATA_DIR, "sentiment_history.json")


def build_cache():
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    cache = {'margin': [], 'turnover': []}

    logger.info("拉取融资融券历史数据...")
    df = pro.margin(start_date='20200101', end_date=datetime.now().strftime("%Y%m%d"))
    if df is not None and not df.empty:
        df['rzye'] = pd.to_numeric(df['rzye'], errors='coerce')
        agg = df.groupby('trade_date').agg(rzye=('rzye', 'sum')).reset_index()
        agg['rzye_chg_3d'] = agg['rzye'].pct_change(3) * 100
        agg = agg.dropna(subset=['rzye_chg_3d'])
        cache['margin'] = agg[['trade_date', 'rzye_chg_3d']].to_dict('records')
        logger.info("  融资数据: %d 条", len(cache['margin']))

    logger.info("拉取全市场换手率历史数据...")
    turnover_records = []
    for year in range(2020, datetime.now().year + 1):
        for month in range(1, 13):
            if year == datetime.now().year and month > datetime.now().month:
                break
            ym = f"{year}{month:02d}01"
            try:
                df = pro.daily_basic(trade_date=ym)
                if df is not None and not df.empty:
                    avg_to = pd.to_numeric(df['turnover_rate_f'], errors='coerce').mean()
                    if not pd.isna(avg_to):
                        turnover_records.append({'trade_date': ym, 'avg_turnover': round(avg_to, 2)})
            except:
                continue
    cache['turnover'] = turnover_records
    logger.info("  换手率数据: %d 条", len(turnover_records))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("✅ 情绪历史缓存已写入: %s", CACHE_PATH)


if __name__ == '__main__':
    build_cache()
