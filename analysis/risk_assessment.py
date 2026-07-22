#!/usr/bin/env python3
"""
市场脆弱度评估 (Market Fragility Assessment)
==========================================
基于三段式信号判断当前市场的风险状态。

信号体系:
  🚩 广度崩塌 — 近5天中N天超过X%行业下跌
  🔴 趋势破坏 — 指数收盘跌破MA20
  💀 死猫反弹 — 指数涨但大部分行业跌+缩量

输出:
  - alert_level: 'normal' | 'caution' | 'warning' | 'danger'
  - alert_label: 中文标签
  - signals: 各信号详情
"""
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from config import DB_PATH

# ── 参数（基于回测优化的默认值） ──
P1_DOWN_THRESHOLD = 70      # 行业下跌比例阈值 (%)
P1_DAYS_REQUIRED = 3        # 5天中需要多少天
P1_STRICT_DOWN = 75         # 严格模式的行业下跌阈值
P1_STRICT_DAYS = 4          # 严格模式的天数要求
P3_DOWN_THRESHOLD = 60      # 死猫反弹的行业下跌阈值
LOOKBACK = 5                # 回溯天数


def assess_fragility(
    l2_tech_result: dict = None,
    index_close: float = None,
    index_pct_chg: float = None,
    index_vol: float = None,
    l2_down_pct_history: List[float] = None,
    index_ma20: float = None,
    index_vol_ma20: float = None,
) -> Dict[str, Any]:
    """
    评估市场脆弱度

    参数可直接传入，也可通过 l2_tech_result 自动提取。
    这样既支持实时计算，也支持从已有数据提取。
    """
    # ── 从 l2_tech_result 提取数据（如果有） ──
    if l2_tech_result and not index_close:
        details = l2_tech_result.get("details", [])
        # 从 details 提取行业涨跌信息已不可行，需要外部传入历史数据

    # ── 计算各信号 ──
    signals = {}

    # Phase 1: 广度崩塌
    if l2_down_pct_history and len(l2_down_pct_history) >= LOOKBACK:
        recent = l2_down_pct_history[-LOOKBACK:]
        heavy_days = sum(1 for v in recent if v > P1_DOWN_THRESHOLD)
        heavy_days_strict = sum(1 for v in recent if v > P1_STRICT_DOWN)
        signals['p1_active'] = heavy_days >= P1_DAYS_REQUIRED
        signals['p1_strict'] = heavy_days_strict >= P1_STRICT_DAYS
        signals['p1_heavy_days'] = heavy_days
        signals['p1_recent_down_pct'] = recent
    else:
        signals['p1_active'] = False
        signals['p1_strict'] = False
        signals['p1_heavy_days'] = 0

    # Phase 2: 趋势破坏
    if index_close is not None and index_ma20 is not None and index_ma20 > 0:
        signals['p2_below_ma20'] = index_close < index_ma20
        signals['p2_deviation'] = (index_close - index_ma20) / index_ma20 * 100
    else:
        signals['p2_below_ma20'] = False
        signals['p2_deviation'] = 0

    # Phase 3: 死猫反弹
    if (index_pct_chg is not None and index_vol is not None
            and index_vol_ma20 is not None and index_vol_ma20 > 0):
        signals['p3_active'] = (
            index_pct_chg > 0
            and l2_down_pct_history is not None
            and len(l2_down_pct_history) > 0
            and l2_down_pct_history[-1] > P3_DOWN_THRESHOLD
            and index_vol < index_vol_ma20
        )
    else:
        signals['p3_active'] = False

    # ── 综合判定 ──
    if signals.get('p1_strict') or (signals.get('p1_active') and signals.get('p2_below_ma20')):
        alert_level = 'danger'
        alert_label = '⛔ 市场脆弱：高度谨慎'
        pos_cap = 15
        pos_desc = '广度崩塌+趋势破位，极端风险'
    elif signals.get('p1_active') or (signals.get('p2_below_ma20') and signals.get('p3_active')):
        alert_level = 'warning'
        alert_label = '⚠️ 市场偏弱：轻仓防御'
        pos_cap = 30
        pos_desc = '广度恶化或趋势走弱，轻仓防御'
    elif signals.get('p3_active'):
        alert_level = 'caution'
        alert_label = '📊 反弹乏力：注意风险'
        pos_cap = 40
        pos_desc = '指数上涨但个股分化，不宜追高'
    else:
        alert_level = 'normal'
        alert_label = None  # 沿用正常仓位计算
        pos_cap = 100
        pos_desc = None

    return {
        'alert_level': alert_level,
        'alert_label': alert_label,
        'pos_cap': pos_cap,
        'pos_desc_override': pos_desc,
        'signals': signals,
        'p1_strict': signals.get('p1_strict', False),
        'p1_active': signals.get('p1_active', False),
        'p2_active': signals.get('p2_below_ma20', False),
        'p3_active': signals.get('p3_active', False),
    }


def compute_from_db(db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    从数据库实时计算当前市场脆弱度
    """
    import tushare as ts
    from config import TUSHARE_TOKEN

    pro = ts.pro_api(TUSHARE_TOKEN)

    # 1. 获取上证指数数据
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    idx = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
    if idx is None or idx.empty:
        return {'alert_level': 'normal', 'error': '指数数据不可用'}

    idx = idx.sort_values('trade_date').reset_index(drop=True)
    latest = idx.iloc[-1]
    index_close = float(latest['close'])
    index_pct_chg = float(latest['pct_chg']) / 100
    index_vol = float(latest['vol'])

    idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
    idx['ma20'] = idx['close'].rolling(20).mean()
    index_ma20 = float(idx['ma20'].iloc[-1]) if not pd.isna(idx['ma20'].iloc[-1]) else 0

    idx['vol_ma20'] = pd.to_numeric(idx['vol'], errors='coerce').rolling(20).mean()
    index_vol_ma20 = float(idx['vol_ma20'].iloc[-1]) if not pd.isna(idx['vol_ma20'].iloc[-1]) else 0

    # 2. 获取 L2 行业数据（近10天）
    conn = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
    recent_l2 = pd.read_sql_query(
        f"SELECT trade_date, ts_code, close FROM sw_l2_index_daily "
        f"WHERE trade_date >= '{cutoff}' "
        f"ORDER BY trade_date", conn
    )
    conn.close()

    if recent_l2.empty:
        return {'alert_level': 'normal', 'error': 'L2数据不可用'}

    recent_l2['close'] = pd.to_numeric(recent_l2['close'], errors='coerce')
    recent_l2['prev_close'] = recent_l2.groupby('ts_code')['close'].shift(1)
    recent_l2['ret'] = (recent_l2['close'] - recent_l2['prev_close']) / recent_l2['prev_close']

    daily_down = recent_l2.groupby('trade_date').agg(
        down_pct=('ret', lambda x: (x < 0).sum() / len(x) * 100),
    ).reset_index().sort_values('trade_date')

    l2_down_pct_history = daily_down['down_pct'].tolist()

    return assess_fragility(
        index_close=index_close,
        index_pct_chg=index_pct_chg,
        index_vol=index_vol,
        l2_down_pct_history=l2_down_pct_history,
        index_ma20=index_ma20,
        index_vol_ma20=index_vol_ma20,
    )
