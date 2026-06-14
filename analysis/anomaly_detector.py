"""
异常归因检测器
- 比较今日结果 vs 近20日滚动均值
- 检测情绪突变、排名剧变、选股突变
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.json")
MAX_HISTORY = 60  # 保留最多60天历史


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)


def detect_anomalies(
    sentiment_result: dict,
    persistence_df: pd.DataFrame | None,
    stock_result: dict,
) -> dict:
    """检测异常并归因。

    Returns:
        {"alerts": [...], "summary": "正常" | "有X项异常"}
    """
    alerts = []
    history = load_history()
    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    today_str = beijing_now.strftime("%Y%m%d")

    # 记录今日快照
    snapshot = {
        "date": today_str,
        "sentiment": sentiment_result.get("sentiment"),
        "avg_momentum": sentiment_result.get("avg_momentum", 0),
        "bullish_count": sentiment_result.get("bullish_count", 0),
        "bearish_count": sentiment_result.get("bearish_count", 0),
        "stock_count": len(stock_result.get("stocks", [])),
    }

    # === 情绪突变检测 ===
    if len(history) >= 5:
        momentums = [h.get("avg_momentum", 0) for h in history[-20:]]
        if len(momentums) >= 5:
            mu = np.mean(momentums)
            sigma = np.std(momentums) + 1e-10
            today_mom = snapshot["avg_momentum"]
            z_score = (today_mom - mu) / sigma

            if abs(z_score) > 2:
                direction = "恶化" if today_mom < mu else "改善"
                alerts.append({
                    "type": "情绪突变",
                    "detail": f"今日动量 {today_mom:+.1f}% 偏离均值 {mu:+.1f}% ({z_score:+.1f}σ)，"
                              f"主要因为 {(snapshot['bearish_count'] if today_mom < mu else snapshot['bullish_count'])} 个行业{direction}",
                    "severity": "high" if abs(z_score) > 3 else "medium",
                })

    # === 排名剧变检测 ===
    if persistence_df is not None and not persistence_df.empty and len(history) >= 1:
        prev = history[-1]
        prev_stock_count = prev.get("stock_count", 0)
        today_stock_count = snapshot["stock_count"]

        if prev_stock_count > 0:
            change_rate = abs(today_stock_count - prev_stock_count) / prev_stock_count
            if change_rate > 0.5:
                alerts.append({
                    "type": "选股突变",
                    "detail": f"精选个股数从 {prev_stock_count} 变为 {today_stock_count}（变化 {change_rate:.0%}），"
                              f"可能因行业持续性排名大幅调整",
                    "severity": "medium",
                })

    # === 保存历史 ===
    history.append(snapshot)
    save_history(history)

    result = {
        "alerts": alerts,
        "summary": f"有{len(alerts)}项异常" if alerts else "正常",
        "snapshot": snapshot,
    }

    if alerts:
        logger.warning("异常检测: %s", result["summary"])
    return result
