#!/usr/bin/env python3
"""
市场情绪仪表盘 (Market Sentiment Dashboard)
==========================================
基于交易行为数据量化市场情绪，提供辅助确认信号。

指标:
  1. 杠杆情绪: 融资买入额 / 全市场成交额
     当占比极高(>90%分位) = 散户狂热 → 见顶风险
     当占比极低(<10%分位) = 散户恐慌 → 见底信号

  2. 成交热度: 全市场换手率(TTM)在近N日分位
     当换手率极高(>90%分位) = 交易过热
     当换手率极低(<10%分位) = 交易冰点

  3. 涨跌停比 (扩展): 涨停家数/跌停家数
     极值辅助判断情绪顶点

数据源: Tushare Pro (margin, daily_basic)
"""
import os, sys, json, logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 分位参考期（交易日）
PERCENTILE_LOOKBACK = 600  # 约2.5年
HISTORY_CACHE_FILE = None  # 由 run_analysis.py 设置

# ── 情绪阈值 ──
EXTREME_HIGH_PCT = 90   # 极端情绪上分位
EXTREME_LOW_PCT = 10    # 极端情绪下分位
WARN_HIGH_PCT = 80      # 偏高
WARN_LOW_PCT = 20       # 偏低


def fetch_margin_leverage(pro, days: int = 60) -> pd.DataFrame:
    """获取融资融券数据，计算融资买入占比"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")

    df = pro.margin(start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()

    df['rzye'] = pd.to_numeric(df['rzye'], errors='coerce')
    df['rzmre'] = pd.to_numeric(df['rzmre'], errors='coerce')
    df['rzche'] = pd.to_numeric(df['rzche'], errors='coerce')

    # 按日期聚合（上海+深圳）
    agg = df.groupby('trade_date').agg(
        rzye=('rzye', 'sum'),
        rzmre=('rzmre', 'sum'),
        rzche=('rzche', 'sum'),
    ).reset_index()

    agg['rzye_chg_pct'] = agg['rzye'].pct_change(3) * 100  # 3日变化率
    agg['rz_net'] = agg['rzmre'] - agg['rzche']  # 融资净买入

    # 获取全市场成交额用于计算占比
    # 注意：rzmre是融资买入额(元)，需匹配全市场成交额
    return agg


def fetch_market_turnover(pro, days: int = 60) -> pd.DataFrame:
    """获取全市场换手率数据"""
    records = []
    end = datetime.now()
    # Tushare daily_basic 按日期取，需要遍历
    # 每天全市场平均换手率
    for i in range(days):
        date = (end - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = pro.daily_basic(trade_date=date)
            if df is None or df.empty:
                continue
            turnover = pd.to_numeric(df['turnover_rate_f'], errors='coerce')
            total_mv = pd.to_numeric(df['total_mv'], errors='coerce')
            amount = pd.to_numeric(df['amount'], errors='coerce')

            records.append({
                'trade_date': date,
                'avg_turnover': turnover.mean(),
                'med_turnover': turnover.median(),
                'total_mv': total_mv.sum() / 1e8,  # 亿元
                'total_amount': amount.sum() / 1e8 if not amount.isna().all() else 0,
                'pe_median': pd.to_numeric(df['pe_ttm'], errors='coerce').median(),
                'n_stocks': len(df),
            })
        except Exception as e:
            logger.debug("daily_basic %s: %s", date, e)
            continue

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values('trade_date').reset_index(drop=True)
    return result


def compute_sentiment(
    margin_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    history_margin: pd.DataFrame = None,
    history_turnover: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    计算当前市场情绪指标

    Args:
        margin_df: 近60天融资融券数据
        turnover_df: 近60天全市场换手率数据
        history_margin: 历史全量融资数据（用于分位计算）
        history_turnover: 历史全量换手率数据（用于分位计算）
    """
    result = {
        'indicators': {},
        'overall_sentiment': 'normal',
        'alert': None,
    }

    # ── 1. 杠杆情绪：融资买入占比 ──
    if not margin_df.empty and len(margin_df) > 5:
        latest = margin_df.iloc[-1]
        chg_3d = latest.get('rzye_chg_pct', 0)

        # 3日变化信号
        if chg_3d is not None and not pd.isna(chg_3d):
            if chg_3d < -1:
                leverage_signal = '去杠杆⚠️'
                leverage_status = '融资余额连续下降，杠杆资金离场'
            elif chg_3d > 1:
                leverage_signal = '加杠杆🔥'
                leverage_status = '融资余额上升，杠杆资金入场'
            else:
                leverage_signal = '平稳'
                leverage_status = '融资余额变化不大'
        else:
            leverage_signal = 'N/A'
            leverage_status = ''

        # 历史分位计算
        leverage_pct = 50
        if history_margin is not None and not history_margin.empty:
            hist_chg = history_margin['rzye_chg_3d'].dropna()
            if len(hist_chg) > 50 and chg_3d is not None and not pd.isna(chg_3d):
                leverage_pct = (hist_chg <= chg_3d).sum() / len(hist_chg) * 100

        result['indicators']['leverage'] = {
            'label': '杠杆情绪',
            'value': f"{chg_3d:.1f}%" if chg_3d is not None and not pd.isna(chg_3d) else 'N/A',
            'pct': round(leverage_pct),
            'signal': leverage_signal,
            'status': leverage_status,
        }

    # ── 2. 成交热度：全市场换手率 ──
    if not turnover_df.empty and len(turnover_df) > 5:
        latest = turnover_df.iloc[-1]
        avg_to = latest.get('avg_turnover', 0)
        amount = latest.get('total_amount', 0)

        # 历史分位
        turnover_pct = 50
        if history_turnover is not None and not history_turnover.empty:
            hist_to = history_turnover['avg_turnover'].dropna()
            if len(hist_to) > 50 and avg_to is not None and not pd.isna(avg_to):
                turnover_pct = (hist_to <= avg_to).sum() / len(hist_to) * 100

        turnover_label = '偏高🔥' if turnover_pct >= WARN_HIGH_PCT else (
            '偏低❄️' if turnover_pct <= WARN_LOW_PCT else '正常')

        result['indicators']['turnover'] = {
            'label': '成交热度',
            'value': f"{avg_to:.1f}%" if avg_to is not None and not pd.isna(avg_to) else 'N/A',
            'amount': f"{amount:.0f}亿" if amount else 'N/A',
            'pct': round(turnover_pct),
            'signal': turnover_label,
        }

    # ── 综合判定 ──
    n_extreme = 0
    alerts = []
    for name, ind in result['indicators'].items():
        pct = ind.get('pct', 50)
        if pct >= EXTREME_HIGH_PCT:
            n_extreme += 1
            alerts.append(f"{ind['label']}极端({pct}%分位)")
        elif pct >= WARN_HIGH_PCT:
            alerts.append(f"{ind['label']}偏高({pct}%分位)")

    if n_extreme >= 2:
        result['overall_sentiment'] = 'extreme'
        result['alert'] = '⚠️ 市场情绪极端（多指标确认）'
    elif n_extreme >= 1:
        result['overall_sentiment'] = 'elevated'
        result['alert'] = '📊 市场情绪偏高'
    elif all(ind.get('pct', 50) <= WARN_LOW_PCT for ind in result['indicators'].values()):
        result['overall_sentiment'] = 'cold'
        result['alert'] = '❄️ 市场情绪低迷'
    else:
        result['overall_sentiment'] = 'normal'

    return result


def load_history(filepath: str = None) -> Dict[str, pd.DataFrame]:
    """
    从本地缓存加载历史数据（用于分位计算）

    缓存格式: sentiment_history.json
    {
        "margin": [{"trade_date": "...", "rzye_chg_3d": 0.5}, ...],
        "turnover": [{"trade_date": "...", "avg_turnover": 1.2}, ...]
    }
    """
    if filepath is None or not os.path.exists(filepath):
        return {'margin': pd.DataFrame(), 'turnover': pd.DataFrame()}

    try:
        with open(filepath) as f:
            data = json.load(f)
        margin = pd.DataFrame(data.get('margin', []))
        turnover = pd.DataFrame(data.get('turnover', []))
        return {'margin': margin, 'turnover': turnover}
    except Exception as e:
        logger.warning("加载情绪历史缓存失败: %s", e)
        return {'margin': pd.DataFrame(), 'turnover': pd.DataFrame()}


def run(db_path: str = None, history_cache: str = None) -> Dict[str, Any]:
    """
    主入口：获取当日数据并计算情绪指标

    Args:
        db_path: 数据库路径（暂未使用，保留接口兼容）
        history_cache: 历史缓存文件路径

    Returns:
        sentiment dict
    """
    import tushare as ts
    from config import TUSHARE_TOKEN

    pro = ts.pro_api(TUSHARE_TOKEN)

    # 1. 加载历史数据
    history = load_history(history_cache)
    hist_margin = history['margin']
    hist_turnover = history['turnover']

    # 2. 获取近期数据
    margin_df = fetch_margin_leverage(pro, days=60)
    turnover_df = fetch_market_turnover(pro, days=60)

    # 3. 计算情绪
    sentiment = compute_sentiment(
        margin_df=margin_df,
        turnover_df=turnover_df,
        history_margin=hist_margin,
        history_turnover=hist_turnover,
    )

    return sentiment
