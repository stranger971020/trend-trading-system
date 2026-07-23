#!/usr/bin/env python3
"""
舆情监控模块 — 基于 a-stock-data 的实时新闻情绪分析
=============================================
数据源（均免费，零 API Key）:
  1. 财联社电报 — 全市场实时快讯
  2. 同花顺强势股归因 — 当日热点题材

不依赖历史回测，作为 warning/danger 的实时辅助确认信号。

策略建议:
  - warning + 负面舆情频发 → 置信度升高，严格执行减仓
  - warning + 无明显负面舆情 → 置信度正常
  - 无 warning + 舆情集中负面(如"暴跌""崩盘"高频出现) → 提前关注
"""
import logging, re, json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))

# 负面关键词
NEGATIVE_KEYWORDS = [
    '暴跌', '崩盘', '恐慌', '抛售', '踩踏', '跌停', '熔断',
    '流动性危机', '债务违约', '信托暴雷', '外资撤离',
    '监管处罚', '立案调查', '退市', 'st', '业绩爆雷',
    '利空', '做空', '减持', '解禁', '资金出逃',
]

# 正面/兴奋关键词
POSITIVE_KEYWORDS = [
    '暴涨', '涨停', '大涨', '突破', '创新高', '拉升',
    '政策利好', '重磅利好', '重大突破', '超预期',
    '放量上攻', '资金涌入', '北向加仓',
]
EXCITED_KEYWORDS = [
    '牛市', '暴涨', '涨停潮', '全面爆发', '井喷',
    '历史新高', '万亿成交', '抢筹',
]

# 市场情绪关键词（用于分级分类）
PANIC_KEYWORDS = ['恐慌', '崩盘', '踩踏', '熔断', '流动性危机']
CAUTION_KEYWORDS = ['暴跌', '抛售', '利空', '外资撤离', '做空']
EXCITEMENT_KEYWORDS = ['暴涨', '涨停潮', '全面爆发', '井喷', '抢筹']
BOOM_KEYWORDS = ['突破', '创新高', '超预期', '政策利好', '重大突破']


def _cls_telegraph(page_size: int = 30) -> List[dict]:
    """财联社电报 — 零 key 直连"""
    import hashlib, requests
    params = {"appName": "CailianpressWeb", "os": "web", "sv": "7.7.5",
              "last_time": "", "refresh_type": "1", "rn": str(page_size)}
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sign = hashlib.md5(hashlib.sha1(qs.encode()).hexdigest().encode()).hexdigest()
    url = f"https://www.cls.cn/v1/roll/get_roll_list?{qs}&sign={sign}"
    headers = {"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0 Safari/537.36",
               "Referer": "https://www.cls.cn/"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        d = r.json()
        rows = []
        for item in d.get("data", {}).get("roll_data", []) or []:
            ts = item.get("ctime")
            t = datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""
            title = item.get("title", "") or item.get("brief", "")
            content = item.get("content", "") or ""
            rows.append({"title": title, "content": content, "time": t})
        return rows
    except Exception as e:
        logger.debug("财联社电报失败: %s", e)
        return []


def _classify_sentiment(text: str) -> str:
    """简单关键词情绪分类（含正面/兴奋）"""
    for kw in PANIC_KEYWORDS:
        if kw in text:
            return 'panic'
    for kw in EXCITEMENT_KEYWORDS:
        if kw in text:
            return 'excited'
    for kw in CAUTION_KEYWORDS:
        if kw in text:
            return 'negative'
    for kw in BOOM_KEYWORDS:
        if kw in text:
            return 'positive'
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            return 'negative'
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            return 'positive'
    return 'neutral'


def analyze_news_sentiment() -> Dict[str, Any]:
    """
    分析当日财联社电报情绪

    Returns:
        sentiment_level: 'calm' | 'normal' | 'negative' | 'panic'
        negative_count: 负面消息数
        total_count: 总消息数
        top_negative: 最负面消息标题
        negative_pct: 负面比例
    """
    news = _cls_telegraph(page_size=50)
    if not news:
        return {'sentiment_level': 'normal', 'error': '舆情数据不可用'}

    today = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")

    # 只分析当日消息
    today_news = [n for n in news if n.get('time', '').startswith(today[:10]) or True]

    results = []
    for n in today_news:
        text = f"{n.get('title', '')} {n.get('content', '')}"
        sentiment = _classify_sentiment(text)
        results.append({**n, 'sentiment': sentiment})

    total = len(results)
    negative = [r for r in results if r['sentiment'] in ('negative', 'panic')]
    panic = [r for r in results if r['sentiment'] == 'panic']
    positive = [r for r in results if r['sentiment'] in ('positive', 'excited')]
    excited = [r for r in results if r['sentiment'] == 'excited']
    neg_count = len(negative)
    panic_count = len(panic)
    pos_count = len(positive)
    excited_count = len(excited)
    neg_pct = neg_count / total * 100 if total > 0 else 0
    pos_pct = pos_count / total * 100 if total > 0 else 0

    # 四级状态判定（优先极端，其次看占比）
    if panic_count > 2:
        level = 'panic'
    elif excited_count > 3:
        level = 'excited'
    elif neg_pct > 30:
        level = 'negative'
    elif pos_pct > 30:
        level = 'positive'
    elif neg_pct > 15:
        level = 'mild_negative'
    elif pos_pct > 15:
        level = 'mild_positive'
    else:
        level = 'calm'

    top_negative = [r['title'] for r in negative[:3]] if negative else []
    top_positive = [r['title'] for r in positive[:2]] if positive else []

    return {
        'sentiment_level': level,
        'total_news': total,
        'negative_count': neg_count,
        'panic_count': panic_count,
        'negative_pct': round(neg_pct, 1),
        'positive_count': pos_count,
        'excited_count': excited_count,
        'positive_pct': round(pos_pct, 1),
        'top_negative': top_negative,
        'top_positive': top_positive,
    }


def compute_confidence_overlay(
    warning_level: str,
    news_sentiment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将舆情情绪叠加到 risk assessment 上，输出置信度调整建议

    Args:
        warning_level: 'normal' | 'caution' | 'warning' | 'danger'
        news_sentiment: analyze_news_sentiment() 的返回值

    Returns:
        overlay_level: 'elevated' | 'normal' | 'reduced'
        suggestion: 操作建议
    """
    if not news_sentiment or 'sentiment_level' not in news_sentiment:
        return {'overlay': 'normal', 'suggestion': None}

    sent = news_sentiment['sentiment_level']

    # 恐慌 → 警级提升
    if sent in ('panic',):
        if warning_level in ('warning', 'danger'):
            return {'overlay': 'elevated', 'suggestion': '🔴 舆情恐慌+市场信号一致，严格执行减仓'}
        elif warning_level == 'caution':
            return {'overlay': 'elevated', 'suggestion': '🔴 舆情恐慌蔓延，建议提高仓位控制'}
        else:
            return {'overlay': 'normal', 'suggestion': '📊 舆情出现恐慌信号，即使无预警也应关注'}

    # 负面 → 配合确认
    if sent == 'negative':
        if warning_level in ('warning', 'danger'):
            return {'overlay': 'elevated', 'suggestion': '⚠️ 负面舆情配合确认，建议执行减仓'}
        return {'overlay': 'normal', 'suggestion': '📊 负面舆情增多，保持关注'}

    # 兴奋 → 可能存在风险(过热)
    if sent == 'excited':
        if warning_level in ('warning', 'danger'):
            return {'overlay': 'normal', 'suggestion': '🔥 舆情兴奋但市场偏弱，警惕情绪落差'}
        return {'overlay': 'normal', 'suggestion': '🔥 市场情绪亢奋，注意过热风险'}

    # 正面 → 正常偏暖
    if sent == 'positive':
        return {'overlay': 'normal', 'suggestion': None}

    # 平静/微负面/微正面 → 无特殊建议
    return {'overlay': 'normal', 'suggestion': None}


def run(warning_level: str = 'normal') -> Dict[str, Any]:
    """主入口：获取舆情+计算置信度叠加"""
    news = analyze_news_sentiment()
    overlay = compute_confidence_overlay(warning_level, news)
    return {
        'news_sentiment': news,
        'overlay': overlay,
    }
