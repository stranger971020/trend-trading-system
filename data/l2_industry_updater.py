"""
申万二级行业日线数据获取与存储
- 从 Tushare index_classify 获取 L2 代码+名称
- 通过 akshare index_hist_sw() 获取日线
- 存储到 SQLite (sw_index_data.db → sw_l2_index_daily)
"""

import logging, sqlite3, time
import akshare as ak
import pandas as pd
import tushare as ts
from config import TUSHARE_TOKEN, DB_PATH, AK_API_RATE_LIMIT, API_RETRY_COUNT, API_RETRY_DELAY, L3_MIN_RECENT_DATE

logger = logging.getLogger(__name__)

L2_TABLE = "sw_l2_index_daily"
_AKSHARE_COL_MAP = {"日期":"trade_date","代码":"ak_code","收盘":"close","开盘":"open","最高":"high","最低":"low","成交量":"vol","成交额":"amount"}

CREATE_L2 = f"""
CREATE TABLE IF NOT EXISTS {L2_TABLE} (
    trade_date TEXT NOT NULL, ts_code TEXT NOT NULL, name TEXT NOT NULL,
    parent_l1 TEXT, parent_name TEXT,
    open REAL, high REAL, low REAL, close REAL, vol REAL, amount REAL,
    PRIMARY KEY (trade_date, ts_code)
);"""

def _ts_to_sym(code): return code.replace(".SI","").replace(".SH","").replace(".SZ","")

def load_l2_mapping():
    pro = ts.pro_api(TUSHARE_TOKEN)
    df_l2 = pro.index_classify(level="L2", src="SW2021")
    df_l1 = pro.index_classify(level="L1", src="SW2021")
    l1_names = {}
    if df_l1 is not None:
        for _, r in df_l1.iterrows():
            l1_names[str(r["index_code"])] = str(r["industry_name"])
    mapping = {}
    if df_l2 is not None:
        for _, r in df_l2.iterrows():
            code = str(r.get("index_code",""))
            name = str(r.get("industry_name",""))
            parent = str(r.get("parent_code",""))
            if code and name:
                mapping[code] = {"name": name, "parent_l1": parent, "parent_name": l1_names.get(parent,"")}
    logger.info("加载 %d 个 SW L2 行业映射", len(mapping))
    return mapping

def init_l2_table(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_L2)
    for idx in [f"CREATE INDEX IF NOT EXISTS idx_l2_ts ON {L2_TABLE}(ts_code);",
                f"CREATE INDEX IF NOT EXISTS idx_l2_dt ON {L2_TABLE}(trade_date);"]:
        conn.execute(idx)
    conn.commit(); conn.close()

def fetch_and_store_l2(db_path=DB_PATH, l2_mapping=None):
    """获取并存储 L2 行业日线数据。有新数据就用新的，没有就用现有的。"""
    return _fetch_l2_once(db_path, l2_mapping)


def _fetch_l2_once(db_path=DB_PATH, l2_mapping=None):
    if l2_mapping is None: l2_mapping = load_l2_mapping()
    init_l2_table(db_path)
    conn = sqlite3.connect(db_path)
    active = stale = total_new = 0
    for idx, (ts_code, info) in enumerate(sorted(l2_mapping.items())):
        name = info["name"]; parent_l1 = info.get("parent_l1",""); parent_name = info.get("parent_name","")
        cur = conn.execute(f"SELECT MAX(trade_date) FROM {L2_TABLE} WHERE ts_code=?", (ts_code,))
        latest = cur.fetchone()[0]
        symbol = _ts_to_sym(ts_code)
        try:
            df_raw = ak.index_hist_sw(symbol=symbol, period="day")
            if df_raw is None or df_raw.empty: continue
            df = df_raw.rename(columns=_AKSHARE_COL_MAP)
            df["ts_code"] = ts_code; df["name"] = name
            df = df[["trade_date","ts_code","name","open","high","low","close","vol","amount"]]
            df["trade_date"] = df["trade_date"].astype(str).str.replace("-","")
            if df["trade_date"].max() < L3_MIN_RECENT_DATE: stale += 1; continue
            if latest: df_new = df[df["trade_date"] > latest]
            else: df_new = df
            if df_new.empty: active += 1; continue
            df_new["parent_l1"] = parent_l1; df_new["parent_name"] = parent_name
            df_new.to_sql(L2_TABLE, conn, if_exists="append", index=False)
            conn.commit()
            total_new += len(df_new); active += 1
            if idx == 0 or len(df_new) > 10:
                logger.info("[%d/%d] %s (%s): +%d", idx+1, len(l2_mapping), ts_code, name, len(df_new))
        except Exception as e:
            logger.debug("%s fail: %s", ts_code, e)
        time.sleep(AK_API_RATE_LIMIT)
    conn.close()
    result = {"total": len(l2_mapping), "active": active, "stale": stale, "new_rows": total_new}
    logger.info("L2完成: %d代码, %d活跃, +%d条", len(l2_mapping), active, total_new)
    return result

def load_l2_daily(db_path=DB_PATH, min_rows=20):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {L2_TABLE} ORDER BY ts_code, trade_date ASC", conn)
    conn.close()
    if df.empty: return df
    counts = df.groupby("ts_code").size()
    return df[df["ts_code"].isin(counts[counts >= min_rows].index)].copy()
