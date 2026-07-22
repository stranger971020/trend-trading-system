from __future__ import annotations
"""
虚拟持仓跟踪
- 维护模拟持仓表，每日记录理论持仓变化
- 计算净值曲线
- 输出持仓明细
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import (
    DATA_DIR,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
HISTORY_FILE = os.path.join(DATA_DIR, "portfolio_history.csv")


def load_portfolio() -> dict:
    """加载当前持仓。"""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {"holdings": {}, "nav": 1.0, "last_update": None}


def save_portfolio(portfolio: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def update_portfolio(
    stock_picks: list[dict],
    stock_daily_df: pd.DataFrame | None = None,
) -> dict:
    """根据今日选股更新虚拟持仓。

    Args:
        stock_picks: 今日精选个股列表
        stock_daily_df: 个股日线数据（用于计算收益）

    Returns:
        {"action": "rebalanced", "new_positions": N, "closed": M, "nav": 1.05, ...}
    """
    portfolio = load_portfolio()
    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    today_str = beijing_now.strftime("%Y%m%d")

    # 今日选中的股票集合
    today_codes = {p["ts_code"] for p in stock_picks}

    # 旧持仓
    old_holdings = portfolio.get("holdings", {})
    old_codes = set(old_holdings.keys())

    # 计算旧持仓今日收益
    if stock_daily_df is not None and old_holdings:
        old_nav = portfolio.get("nav", 1.0)
        for code in list(old_holdings.keys()):
            sdf = stock_daily_df[stock_daily_df["ts_code"] == code].sort_values("trade_date")
            if len(sdf) >= 2:
                ret = (sdf["close"].iloc[-1] / sdf["close"].iloc[-2] - 1)
                weight = old_holdings[code]["weight"]
                old_nav *= (1 + weight * ret)
        portfolio["nav"] = old_nav

    # 调仓
    new_holdings = {}
    closed = old_codes - today_codes
    new_positions = today_codes - old_codes
    held = today_codes & old_codes

    if today_codes:
        weight = 1.0 / len(today_codes)
    else:
        weight = 0

    for code in today_codes:
        entry_price = portfolio["nav"]
        if code in old_holdings:
            entry_price = old_holdings[code].get("entry_price", entry_price)
        new_holdings[code] = {
            "weight": weight,
            "entry_date": old_holdings.get(code, {}).get("entry_date", today_str),
            "entry_price": entry_price,
        }

    portfolio["holdings"] = new_holdings
    portfolio["last_update"] = today_str

    # 记录历史
    _append_history(today_str, portfolio, len(new_positions), len(closed))

    save_portfolio(portfolio)

    result = {
        "action": "rebalanced" if new_positions or closed else "held",
        "total_positions": len(new_holdings),
        "new_positions": len(new_positions),
        "closed": len(closed),
        "held": len(held),
        "nav": round(portfolio["nav"], 4),
    }
    logger.info(
        "虚拟持仓: %d 只 (新开%d, 平仓%d, 持有%d), NAV=%.4f",
        result["total_positions"], result["new_positions"],
        result["closed"], result["held"], result["nav"],
    )
    return result


def _append_history(date_str: str, portfolio: dict, new_pos: int, closed: int) -> None:
    """追加历史记录。"""
    row = {
        "date": date_str,
        "nav": portfolio["nav"],
        "positions": len(portfolio["holdings"]),
        "new": new_pos,
        "closed": closed,
    }
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)


def get_portfolio_summary() -> dict:
    """获取持仓摘要。"""
    portfolio = load_portfolio()
    history = pd.DataFrame()
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)

    return {
        "nav": portfolio.get("nav", 1.0),
        "positions": len(portfolio.get("holdings", {})),
        "last_update": portfolio.get("last_update"),
        "total_return": round((portfolio.get("nav", 1.0) - 1) * 100, 2),
        "history_days": len(history),
    }
