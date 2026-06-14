"""
交易日判断与调度
- 通过 Tushare trade_cal API 判断某日是否为 A 股交易日
- 北京时间（UTC+8）日期计算
"""

import logging
from datetime import datetime, timedelta, timezone

import tushare as ts

from config import (
    TUSHARE_TOKEN,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=BEIJING_TZ_OFFSET))


def _beijing_date_str(dt: datetime | None = None) -> str:
    """返回北京时间今日的 YYYYMMDD 字符串。"""
    if dt is None:
        dt = datetime.now(_BEIJING_TZ)
    return dt.strftime("%Y%m%d")


def is_trading_day(
    pro=None,
    target_date: str | None = None,
) -> bool:
    """判断目标日期是否为 A 股交易日。

    Args:
        pro: Tushare pro_api 实例（可选，不传则自动创建）
        target_date: 目标日期 YYYYMMDD，默认今天（北京时间）

    Returns:
        True 如果是交易日
    """
    if target_date is None:
        target_date = _beijing_date_str()

    # 先做快速周末判断
    dt = datetime.strptime(target_date, "%Y%m%d")
    if dt.weekday() >= 5:  # 周六=5, 周日=6
        logger.info("今日 (%s) 是周末，非交易日", target_date)
        return False

    if pro is None:
        pro = ts.pro_api(TUSHARE_TOKEN)

    try:
        df = pro.trade_cal(
            exchange="SSE",
            start_date=target_date,
            end_date=target_date,
        )
        if df is not None and not df.empty:
            is_open = df.iloc[0].get("is_open", 0)
            result = str(is_open) == "1"
            if result:
                logger.info("✓ %s 是交易日", target_date)
            else:
                logger.info("✗ %s 非交易日（节假日）", target_date)
            return result
        else:
            logger.warning("trade_cal 返回空，假设非交易日")
            return False
    except Exception as e:
        logger.error("交易日查询失败: %s，假设非交易日", e)
        return False


def get_latest_trading_day(
    pro=None,
    from_date: str | None = None,
    max_lookback: int = 10,
) -> str | None:
    """获取最近的交易日。

    Args:
        pro: Tushare pro_api
        from_date: 起始日期，默认今天
        max_lookback: 最多回看天数

    Returns:
        最近的交易日 YYYYMMDD，找不到返回 None
    """
    if from_date is None:
        from_date = _beijing_date_str()
    if pro is None:
        pro = ts.pro_api(TUSHARE_TOKEN)

    end_dt = datetime.strptime(from_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=max_lookback)

    try:
        df = pro.trade_cal(
            exchange="SSE",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
        if df is not None and not df.empty:
            open_days = df[df["is_open"] == 1].sort_values(
                "cal_date", ascending=False
            )
            if not open_days.empty:
                return str(open_days.iloc[0]["cal_date"])
    except Exception as e:
        logger.error("查询最近交易日失败: %s", e)

    return None
