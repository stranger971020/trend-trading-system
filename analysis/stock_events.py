"""
个股风险事件缓存系统
- 从 Tushare disclosure API 获取公告，分类提取减持/定增/质押等风险事件
- 存储在 SQLite 中，避免每次运行重复调 Tushare API
- 首次获取后只做增量更新（仅拉最新数据）
- 缓存 7 天有效，过期自动重新拉取
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN

logger = logging.getLogger(__name__)

TABLE = "stock_events"
FETCH_DAYS = 90      # 首次拉取的天数
INCR_DAYS = 14       # 增量拉取的天数
CACHE_DAYS = 7       # 缓存有效期
API_DELAY = 0.25     # API 调用间隔（限速）

# 风险关键词 → 显示标签
RISK_KEYWORDS = [
    ("减持", "⚠️ 减持"),
    ("减持计划", "⚠️ 减持计划"),
    ("定增", "🟡 定增"),
    ("非公开发行", "🟡 定增"),
    ("配股", "🟡 配股"),
    ("质押", "🟡 质押"),
    ("业绩预亏", "🔴 业绩预亏"),
    ("业绩预告亏损", "🔴 业绩预亏"),
    ("净利润为负", "🔴 亏损预警"),
    ("亏损", "🔴 亏损预警"),
    ("警示函", "🔴 监管警示"),
    ("问询函", "🟡 监管问询"),
    ("监管函", "🔴 监管警示"),
    ("立案", "🔴 立案调查"),
    ("调查", "🔴 被调查"),
    ("退市风险", "🔴 退市风险"),
    ("ST", "🔴 ST风险"),
    ("终止上市", "🔴 终止上市"),
    ("股份回购", "🟢 股份回购"),  # 正面信号
]

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_date TEXT NOT NULL,
    title TEXT NOT NULL,
    fetched_at TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(ts_code, event_date, title)
);
CREATE INDEX IF NOT EXISTS idx_se_ts_code ON {TABLE}(ts_code);
CREATE INDEX IF NOT EXISTS idx_se_date ON {TABLE}(event_date);
CREATE INDEX IF NOT EXISTS idx_se_fetched ON {TABLE}(fetched_at);
"""


def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_SQL)
    conn.commit()
    conn.close()


def _classify(title: str) -> str | None:
    """根据标题关键词返回事件类型标签，无匹配则返回 None"""
    for kw, label in RISK_KEYWORDS:
        if kw in title:
            return label
    return None


def _need_fetch(conn: sqlite3.Connection, code: str) -> bool:
    """检查该股票的事件缓存是否仍在有效期内"""
    row = conn.execute(
        f"SELECT MAX(fetched_at) FROM {TABLE} WHERE ts_code=?",
        (code,)
    ).fetchone()
    if not row or not row[0]:
        return True
    fetched = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
    return (datetime.now() - fetched).days >= CACHE_DAYS


def ensure_events(
    db_path: str,
    codes: list[str],
    force: bool = False,
) -> int:
    """确保指定个股的风险事件已缓存。

    只对缓存过期或从未缓存的个股从Tushare拉取。

    Args:
        db_path: 数据库路径
        codes: 个股 ts_code 列表
        force: 强制重新拉取（忽略缓存）

    Returns:
        新拉取的事件数量
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    pro = ts.pro_api(TUSHARE_TOKEN)
    today = datetime.now().strftime("%Y%m%d")
    new_count = 0

    for i, code in enumerate(codes):
        if not force and not _need_fetch(conn, code):
            continue  # 缓存有效，跳过

        # 首次拉取用 FETCH_DAYS，增量用 INCR_DAYS
        lookup_days = FETCH_DAYS if force else INCR_DAYS
        start = (datetime.now() - timedelta(days=lookup_days)).strftime("%Y%m%d")

        try:
            df = pro.disclosure(ts_code=code, start_date=start, end_date=today)
        except Exception as e:
            logger.warning("公告API失败 %s: %s", code, e)
            time.sleep(API_DELAY)
            continue

        if df is None or df.empty:
            time.sleep(API_DELAY)
            continue

        for _, row in df.iterrows():
            title = str(row.get("title", ""))
            etype = _classify(title)
            if etype is None:
                continue
            ann_date = str(row.get("ann_date", row.get("end_date", "")))[:10]
            try:
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO {TABLE} (ts_code, event_type, event_date, title) VALUES (?, ?, ?, ?)",
                    (code, etype, ann_date, title[:300]),
                )
                if cur.rowcount and cur.rowcount > 0:
                    new_count += 1
            except Exception:
                pass

        if (i + 1) % 20 == 0:
            conn.commit()
        time.sleep(API_DELAY)

    conn.commit()
    conn.close()
    logger.info("风险事件缓存: %d 只股票 → %d 条新事件", len(codes), new_count)
    return new_count


def get_events(
    db_path: str,
    code: str,
    max_days: int = 90,
) -> list[dict]:
    """查询某只个股的缓存风险事件。

    Returns:
        [{type, date, title, days_ago}, ...] 按日期降序
    """
    conn = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(days=max_days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        f"SELECT event_type, event_date, title FROM {TABLE} WHERE ts_code=? AND event_date>=? ORDER BY event_date DESC, id DESC",
        (code, cutoff),
    ).fetchall()
    conn.close()

    today = datetime.now().date()
    results = []
    for r in rows:
        ed = r[1]
        days_ago = (today - datetime.strptime(ed, "%Y-%m-%d").date()).days if ed and "-" in ed else 0
        results.append({
            "type": r[0],
            "date": ed,
            "title": r[2][:120],
            "days_ago": days_ago,
        })
    return results


def get_warnings_text(
    db_path: str,
    code: str,
    max_days: int = 90,
) -> str:
    """生成个股风险提示文字（单行），无风险返回空字符串"""
    events = get_events(db_path, code, max_days=max_days)
    if not events:
        return ""

    # 取最近 3 条
    recent = events[:3]
    tags = []
    for e in recent:
        label = e["type"]
        if e["days_ago"] <= 7:
            label += " 近期"
        tags.append(label)
    return " | ".join(tags)
