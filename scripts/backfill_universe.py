#!/usr/bin/env python3
"""一次性回填：新增股票历史行情（点时间点，从上市日起）。扩池专用。"""
import sys, time, logging
sys.path.insert(0, '/Users/jren/projects/trend-trading-system')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
from data.stock_industry_mapping import load_stock_universe
from data.stock_daily_updater import fetch_all_stocks
import config
start = time.time()
univ = load_stock_universe()
codes = [u['ts_code'] for u in univ]
list_dates = {u['ts_code']: u['list_date'] for u in univ}
print(f"全量回填启动: {len(codes)} 只, DB={config.DB_PATH}", flush=True)
summary = fetch_all_stocks(config.DB_PATH, codes, list_dates=list_dates)
print(f"回填完成: {summary}, 耗时 {(time.time()-start)/60:.1f} 分钟", flush=True)
