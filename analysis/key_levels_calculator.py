#!/usr/bin/env python3
"""
关键点位计算器 (Key Levels Calculator)
======================================
为每日交易报告「大盘趋势」模块提供支撑位、阻力位、突破位的自动化计算。

功能:
  1. 获取指数日线数据 (akshare, 免费无 API Key 需求)
  2. 计算 MA5/10/20/60/120 均线
  3. 计算 Bollinger 通道 (20日, 2σ)
  4. 计算 VWAP (日内成交量加权均价近似值)
  5. 识别强支撑位 / 极强支撑位 / 中档阻力 / 突破确认位
  6. 判断当前状态 (震荡中继 / 突破信号 / 跌破风险)
  7. 输出结构化 JSON (可直接注入 daily_report.md 模板)
  8. 生成 Markdown 关键点位简报

复用来源:
  - zh_a_stock_akshare.py        (指数数据获取模式)
  - zh_a_601138_chart_analysis.py (技术指标计算)

用法:
  python3 key_levels_calculator.py [--index 上证指数] [--output-dir ./output]

依赖:
  pip install akshare pandas numpy
"""

import os, sys, json, argparse, logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("key_levels")

# ── 可使用的指数代码映射 ──
INDEX_MAP = {
    "上证指数": "000001.SH",
    "深证成指": "399001.SZ",
    "创业板指": "399006.SZ",
    "科创50": "000688.SH",
    "沪深300": "000300.SH",
    "上证50": "000016.SH",
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
}

# ── 默认配置 ──
DEFAULT_LOOKBACK_DAYS = 250  # 约1年数据

DEFAULT_CONFIG = {
    "boll_period": 20,
    "boll_std": 2.0,
    "volume_surge_ratio": 1.2,   # 量增确认阈值
    "breakdown_buffer": 0.005,    # 跌破缓冲 0.5%
    "consolidation_days": 20,     # 震荡区间计算周期
    "gap_lookback_days": 30,      # 跳空缺口回溯期
    "resistance_lookback_high": 120,  # 阻力位需要回溯的最高价周期
}


# ====================================================================
#  数据获取
# ====================================================================

def fetch_index_data_tushare(
    index_name: str = "上证指数",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Optional[pd.DataFrame]:
    """
    使用 Tushare Pro 获取指数日线行情数据（优先使用）

    Tushare 已在此项目中配置且网络连通，比 akshare 更可靠。
    """
    try:
        import tushare as ts
        # 从 project config 读取 token
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import TUSHARE_TOKEN

        pro = ts.pro_api(TUSHARE_TOKEN)

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")

        ts_code = INDEX_MAP.get(index_name, "")
        if not ts_code:
            logger.warning(f"[Tushare] 未知指数: {index_name}")
            return None

        logger.info(f"[Tushare] {index_name}({ts_code}), {start_date} ~ {end_date}")

        df = pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or df.empty:
            logger.warning(f"[Tushare] {index_name} 数据为空")
            return None

        # 统一列名: trade_date, open, high, low, close, vol, amount
        df = df.rename(columns={
            "trade_date": "date", "vol": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"[Tushare] 成功获取 {len(df)} 条 ({df['date'].min().date()} ~ {df['date'].max().date()}), 最新收盘 {df.iloc[-1]['close']}")
        return df

    except ImportError:
        logger.warning("[Tushare] tushare 未安装")
        return None
    except Exception as e:
        logger.warning(f"[Tushare] 获取失败: {e}")
        return None


def fetch_index_data_akshare(
    index_name: str = "上证指数",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Optional[pd.DataFrame]:
    """
    备选：使用 akshare 获取指数日线行情数据
    """
    try:
        import akshare as ak

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")

        logger.info(f"[akshare] 指数: {index_name}, {start_date} ~ {end_date}")

        # 当前 akshare 版本使用 index_zh_a_hist（需要指数代码）
        ts_code = INDEX_MAP.get(index_name, "")
        if ts_code:
            symbol = ts_code.split(".")[0]  # "000001.SH" → "000001"
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "amount",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                logger.info(f"[akshare] 成功获取 {len(df)} 条")
                return df

        # 备选：旧版接口
        df = ak.stock_zh_index_daily_em(symbol=ts_code)
        if df is not None and not df.empty:
            df = df[df["date"] >= start_date]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"[akshare] 备选接口成功 {len(df)} 条")
            return df

        logger.warning(f"[akshare] 无法获取 {index_name} 数据")
        return None

    except ImportError:
        logger.warning("[akshare] akshare 未安装")
        return None
    except Exception as e:
        logger.warning(f"[akshare] 获取失败: {e}")
        return None


def fetch_index_data(
    index_name: str = "上证指数",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Optional[pd.DataFrame]:
    """
    获取指数日线行情数据

    数据源优先级:
      1. Tushare Pro（项目已有配置，网络可靠）
      2. akshare（备选）

    所有数据源均失败时返回 None（报告中将跳过关键点位模块），
    绝不用模拟数据误导判断。

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        或 None（全部数据源不可用时）
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
    logger.info(f"[数据获取] 指数: {index_name}, {start_date} ~ {end_date}")

    # 1. Tushare Pro（优先）
    df = fetch_index_data_tushare(index_name, lookback_days)
    if df is not None and not df.empty:
        return df

    # 2. akshare（备选）
    df = fetch_index_data_akshare(index_name, lookback_days)
    if df is not None and not df.empty:
        return df

    # 所有数据源均不可用 → 返回 None，调用方自行跳过
    logger.warning("[数据获取] 所有数据源均失败，跳过关键点位")
    return None


# ====================================================================
#  技术指标计算
# ====================================================================

def calc_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60, 120]) -> pd.DataFrame:
    """计算移动均线 (复用自: zh_a_601138_chart_analysis.py)"""
    df = df.copy()
    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(window=w).mean()
    return df


def calc_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
) -> pd.DataFrame:
    """
    计算 Bollinger 通道

    复用自: zh_a_601138_chart_analysis.py 中的类似计算逻辑

    Returns:
        df with columns: boll_mid (MA20), boll_upper, boll_lower
    """
    df = df.copy()
    df["boll_mid"] = df["close"].rolling(window=period).mean()
    df["boll_std"] = df["close"].rolling(window=period).std()
    df["boll_upper"] = df["boll_mid"] + std_mult * df["boll_std"]
    df["boll_lower"] = df["boll_mid"] - std_mult * df["boll_std"]
    return df


def calc_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算成交量均线"""
    df = df.copy()
    df["volume_ma"] = df["volume"].rolling(window=period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    return df


def calc_vwap(df: pd.DataFrame) -> float:
    """
    计算 VWAP (成交量加权均价)

    使用当日数据计算:
      VWAP = ∑(成交价×成交量) / ∑成交量
    近似值: (high + low + close) / 3 或 (open + high + low + close) / 4

    若要精确计算需分钟线，此处用日线典型值近似:
      typical_price = (high + low + close) / 3
      VWAP ≈ (typical_price * volume).sum() / volume.sum()   (近N日)

    实际日内 VWAP 由交易所实时计算，此处提供日线级近似参考。
    """
    if df.empty:
        return 0.0

    latest = df.iloc[-1]
    # 使用当日典型价近似
    typical = (latest["high"] + latest["low"] + latest["close"]) / 3
    return round(typical, 2)


def find_recent_gaps(df: pd.DataFrame, lookback: int = 30) -> List[Dict[str, Any]]:
    """
    识别近期跳空缺口

    跳空 = 当日最低 > 前日最高 (向上缺口)
        或 当日最高 < 前日最低 (向下缺口)

    Returns:
        缺口列表，每个缺口包含: date, type(up/down), top, bottom, gap_size
    """
    if len(df) < 2:
        return []

    recent = df.tail(lookback).reset_index(drop=True)
    gaps = []

    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]

        # 向上跳空 (高开缺口)
        if curr["low"] > prev["high"]:
            gaps.append({
                "date": str(curr["date"].date()),
                "type": "up",
                "bottom": round(curr["low"], 2),
                "top": round(curr["high"], 2),
                "gap_size": round(curr["low"] - prev["high"], 2),
                "gap_pct": round((curr["low"] - prev["high"]) / prev["high"] * 100, 2),
            })
        # 向下跳空
        elif curr["high"] < prev["low"]:
            gaps.append({
                "date": str(curr["date"].date()),
                "type": "down",
                "bottom": round(curr["low"], 2),
                "top": round(curr["high"], 2),
                "gap_size": round(prev["low"] - curr["high"], 2),
                "gap_pct": round((prev["low"] - curr["high"]) / prev["high"] * 100, 2),
            })

    return gaps


def find_chip_concentration_zone(df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
    """
    计算筹码密集区 (成交量最大价格区间)

    将价格区间分桶，找出成交量最大的价格区间。
    用于确定筹码密集区上沿/下沿。

    Returns:
        {"upper": 上沿, "lower": 下沿, "center": 中心价, "volume_pct": 集中度}
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback).copy()

    # 价格区间分箱 (20 bins)
    price_min = recent["low"].min()
    price_max = recent["high"].max()

    if price_max == price_min:
        return {"upper": price_max, "lower": price_min, "center": price_max, "volume_pct": 100.0}

    bins = np.linspace(price_min, price_max, 21)  # 20个区间
    labels = (bins[:-1] + bins[1:]) / 2

    # 每个K线的成交量按价格区间分布 (使用典型价)
    recent["typical"] = (recent["high"] + recent["low"] + recent["close"]) / 3
    recent["price_bin"] = pd.cut(recent["typical"], bins=bins, labels=labels)

    vol_by_bin = recent.groupby("price_bin")["volume"].sum()

    if vol_by_bin.empty:
        return {"upper": price_max, "lower": price_min, "center": (price_max + price_min) / 2,
                "volume_pct": 100.0}

    # 密集区 = 成交量最大的价格区间
    max_vol_bin = vol_by_bin.idxmax()
    max_vol = vol_by_bin.max()
    total_vol = vol_by_bin.sum() if vol_by_bin.sum() > 0 else 1

    # 密集区上下沿 (该bin的范围)
    bin_idx = list(vol_by_bin.index).index(max_vol_bin)
    upper = round(bins[bin_idx + 1], 2)
    lower = round(bins[bin_idx], 2)
    center = round(float(max_vol_bin), 2)
    vol_pct = round(max_vol / total_vol * 100, 1)

    return {"upper": upper, "lower": lower, "center": center, "volume_pct": vol_pct}


# ====================================================================
#  关键点位计算
# ====================================================================

def calculate_key_levels(df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    计算所有关键点位

    点位逻辑 (R5-R7 规范):
    - 强支撑位 = min(MA20, MA60, 30日最低价)
    - 极强支撑 = min(MA120, 60日最低价)
    - 中档阻力 = max(MA60, MA120)
    - 突破确认 = min(BOLL上轨, 震荡区间上轨, 近期跳空缺口顶端) 取最接近的值
    - VWAP = 日内成交量加权均价 (近似)

    Args:
        df: 日线行情 DataFrame
        config: 配置参数覆盖

    Returns:
        levels dict with all computed key levels
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # 计算技术指标
    df = calc_moving_averages(df)
    df = calc_bollinger_bands(df, period=cfg["boll_period"], std_mult=cfg["boll_std"])
    df = calc_volume_ma(df)

    latest = df.iloc[-1]
    recent_30 = df.tail(30)
    recent_60 = df.tail(60)

    close_price = round(latest["close"], 2)

    # ── 1. 强支撑位 ──
    ma20 = latest.get("ma20", 0)
    ma60 = latest.get("ma60", 0)
    low_30 = recent_30["low"].min()

    strong_support_values = [v for v in [ma20, ma60, low_30] if v > 0]
    strong_support = round(min(strong_support_values), 2) if strong_support_values else 0.0

    # ── 2. 极强支撑位 ──
    ma120 = latest.get("ma120", 0) if len(df) >= 120 else df["close"].min()
    low_60 = recent_60["low"].min()

    ultra_support_values = [v for v in [ma120, low_60] if v > 0]
    ultra_support = round(min(ultra_support_values), 2) if ultra_support_values else 0.0

    # ── 3. 中档阻力位 ──
    high_60 = recent_60["high"].max()
    high_120 = df.tail(120)["high"].max() if len(df) >= 120 else df["high"].max()

    resistance_mid = round(max(ma60, ma120), 2) if ma60 > 0 and ma120 > 0 else round(max(high_60, high_120), 2)

    # ── 4. 强阻力位 (筹码密集区上沿 + 历史最高) ──
    chip_zone = find_chip_concentration_zone(df, lookback=cfg["consolidation_days"])
    resistance_high = round(max(high_60, high_120, chip_zone["upper"]), 2)

    # ── 5. 突破确认位 ──
    boll_upper = latest.get("boll_upper", 0)

    # 震荡区间上轨 (近20日最高)
    consolidation_upper = recent_30["high"].max()

    # 近期跳空缺口顶端
    gaps = find_recent_gaps(df, lookback=cfg["gap_lookback_days"])
    recent_up_gaps = [g for g in gaps if g["type"] == "up"]
    nearest_gap_top = recent_up_gaps[-1]["top"] if recent_up_gaps else 0

    # 取最接近当前价位的突破确认位
    candidates = []
    for name, val in [("boll_upper", boll_upper),
                       ("consolidation_upper", consolidation_upper),
                       ("gap_top", nearest_gap_top)]:
        if val > 0 and val > close_price:
            candidates.append((name, val))

    if candidates:
        # 选择最接近当前价位的 (最易突破的位置)
        breakout_confirm = min(candidates, key=lambda x: abs(x[1] - close_price))[1]
        breakout_confirm = round(breakout_confirm, 2)
    else:
        # 若无有效突破位，使用boll上轨
        breakout_confirm = round(boll_upper, 2) if boll_upper > 0 else round(consolidation_upper, 2)

    # ── 6. VWAP ──
    vwap = calc_vwap(df)

    # ── 7. 日内滚动参考 ──
    day_high = round(latest["high"], 2)
    day_low = round(latest["low"], 2)

    return {
        # 核心点位
        "close_price": close_price,
        "strong_support": strong_support,
        "ultra_support": ultra_support,
        "resistance_mid": resistance_mid,
        "resistance_high": resistance_high,
        "breakout_confirm": breakout_confirm,

        # 滚动参考
        "vwap": vwap,
        "day_high": day_high,
        "day_low": day_low,

        # 均线
        "ma_5": round(latest.get("ma5", 0), 2) if "ma5" in latest else 0.0,
        "ma_10": round(latest.get("ma10", 0), 2) if "ma10" in latest else 0.0,
        "ma_20": round(ma20, 2),
        "ma_60": round(ma60, 2),
        "ma_120": round(ma120, 2) if ma120 > 0 else 0.0,

        # Bollinger
        "boll_upper": round(boll_upper, 2) if boll_upper > 0 else 0.0,
        "boll_mid": round(latest.get("boll_mid", 0), 2) if "boll_mid" in latest else 0.0,
        "boll_lower": round(latest.get("boll_lower", 0), 2) if "boll_lower" in latest else 0.0,

        # 成交量
        "volume": round(latest["volume"], 2),
        "volume_ma_20": round(latest["volume_ma"], 2) if "volume_ma" in latest else 0.0,
        "volume_ratio": round(latest["volume_ratio"], 2) if "volume_ratio" in latest else 1.0,

        # 筹码密集区
        "chip_zone_upper": chip_zone["upper"],
        "chip_zone_lower": chip_zone["lower"],
        "chip_zone_center": chip_zone["center"],
        "chip_zone_volume_pct": chip_zone["volume_pct"],

        # 缺口
        "gaps": gaps,
    }


# ====================================================================
#  状态判定
# ====================================================================

def determine_market_status(
    close_price: float,
    strong_support: float,
    resistance_mid: float,
    breakout_confirm: float,
    volume_ratio: float,
    volume_surge_ratio: float = 1.2,
    breakdown_buffer: float = 0.005,
) -> Dict[str, Any]:
    """
    判定当前市场状态

    Returns:
        {
            "status": "震荡中继" | "突破信号" | "跌破风险",
            "signal": "hold" | "add" | "reduce",
            "details": str
        }
    """
    support_breakdown = close_price < strong_support * (1 - breakdown_buffer)
    breakout_up = close_price > breakout_confirm
    volume_surge = volume_ratio >= volume_surge_ratio

    # 震荡区间
    is_consolidation = strong_support <= close_price <= resistance_mid

    if breakout_up and volume_surge:
        return {
            "status": "突破信号",
            "action": "add",
            "action_cn": "加仓确认",
            "details": f"收盘 {close_price} > 突破位 {breakout_confirm}，且成交量放大 {volume_ratio}x，突破有效。",
            "confidence": "high" if volume_ratio >= 1.5 else "medium",
        }
    elif breakout_up:
        return {
            "status": "突破信号（待量确认）",
            "action": "watch",
            "action_cn": "观察",
            "details": f"收盘 {close_price} > 突破位 {breakout_confirm}，但成交量仅 {volume_ratio}x 均量，需放量确认。",
            "confidence": "low",
        }
    elif support_breakdown:
        return {
            "status": "跌破风险",
            "action": "reduce",
            "action_cn": "减仓信号",
            "details": f"收盘 {close_price} < 强支撑 {strong_support} 的 {breakdown_buffer*100}% 以下，注意下行风险。",
            "confidence": "high" if volume_ratio >= 1.0 else "medium",
        }
    elif is_consolidation:
        return {
            "status": "震荡中继",
            "action": "hold",
            "action_cn": "持仓观望",
            "details": f"当前在 {strong_support} ~ {resistance_mid} 区间震荡，无明确方向。",
            "confidence": "medium",
        }
    else:
        return {
            "status": "方向不明",
            "action": "hold",
            "action_cn": "观望",
            "details": "当前价格处于关键区域之外，需进一步确认方向。",
            "confidence": "low",
        }


# ====================================================================
#  输出格式化
# ====================================================================

def generate_markdown_report(
    index_name: str,
    levels: Dict[str, Any],
    status: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> str:
    """生成 Markdown 格式的关键点位简报"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    change_pct = ""
    if df is not None and len(df) >= 2:
        prev_close = df.iloc[-2]["close"]
        change_pct = f"({round((levels['close_price'] - prev_close) / prev_close * 100, 2)}%)"

    md = f"""### 【关键点位】
生成时间: {now} | 数据源: akshare

当前点位：{levels['close_price']} {change_pct}

| 类型 | 价位 | 说明 |
|------|------|------|
| **强支撑** | {levels['strong_support']} | min(MA20/MA60, 30日最低) — 跌破+量增→减仓 |
| **极强支撑** | {levels['ultra_support']} | min(MA120, 60日最低) — 极端下行保护 |
| **中档阻力** | {levels['resistance_mid']} | max(MA60/MA120) — 初步压力区 |
| **强阻力/筹码密集区** | {levels['resistance_high']} | max(60日/120日高点, 筹码密集区上沿) |
| **突破确认位** | {levels['breakout_confirm']} | BOLL上轨/震荡区间上轨/近期缺口 — 站稳+量增1.2x→加仓 |

### 今日滚动参考
| 指标 | 数值 |
|------|------|
| VWAP | {levels['vwap']} |
| 日内高 | {levels['day_high']} |
| 日内低 | {levels['day_low']} |
| MA5 | {levels['ma_5']} |
| MA10 | {levels['ma_10']} |
| MA20 | {levels['ma_20']} |
| MA60 | {levels['ma_60']} |
| BOLL上轨 | {levels['boll_upper']} |
| BOLL中轨 | {levels['boll_mid']} |
| BOLL下轨 | {levels['boll_lower']} |
| 成交量 | {levels['volume']:.0f} |
| 量比(20日均量) | {levels['volume_ratio']}x |

### 【当前状态判定】
- **判定**: {status['status']}
- **操作建议**: {status['action_cn']}
- **置信度**: {status['confidence']}
- **依据**: {status['details']}

### 筹码分布
- 密集区: {levels['chip_zone_lower']} ~ {levels['chip_zone_upper']} (集中度 {levels['chip_zone_volume_pct']}%)
- 密集区中心: {levels['chip_zone_center']}

"""

    # 近期缺口
    if levels["gaps"]:
        md += "### 近期跳空缺口\n"
        for i, g in enumerate(levels["gaps"], 1):
            direction = "↑ 向上" if g["type"] == "up" else "↓ 向下"
            md += f"- {direction}缺口 ({g['date']}): {g['bottom']} ~ {g['top']} ({g['gap_pct']}%)\n"
    else:
        md += "### 近期跳空缺口\n- 近30日无跳空缺口\n"

    md += "\n---\n"
    md += f"*关键点位由 key_levels_calculator.py 自动计算 | 指数: {index_name} | {now}*\n"
    md += "*数据仅供参考，不构成投资建议。*\n"

    return md


def generate_json_output(levels: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
    """生成 JSON 输出 (可直接注入 daily_report.md 模板的字段映射)"""

    # 模板字段映射
    template_fields = {
        "strong_support": levels["strong_support"],
        "ultra_support": levels["ultra_support"],
        "resistance_mid": levels["resistance_mid"],
        "resistance_high": levels["resistance_high"],
        "breakout_confirm": levels["breakout_confirm"],
        "breakout_level": levels["breakout_confirm"],
        "support": levels["strong_support"],
        "resistance": levels["resistance_mid"],
        "close_price": levels["close_price"],
        "vwap": levels["vwap"],
        "day_high": levels["day_high"],
        "day_low": levels["day_low"],
        "ma_5": levels["ma_5"],
        "ma_10": levels["ma_10"],
        "ma_20": levels["ma_20"],
        "ma_60": levels["ma_60"],
        "volume": levels["volume"],
        "vol_ma_20": levels["volume_ma_20"],
        "volume_ma_20": levels["volume_ma_20"],
        "volume_ratio": levels["volume_ratio"],

        # 状态
        "market_status": status["status"],
        "market_action": status["action_cn"],
        "market_confidence": status["confidence"],
        "market_details": status["details"],
    }

    return {
        "index_name": "",
        "trade_date": datetime.now().strftime("%Y-%m-%d"),
        "levels": levels,
        "status": status,
        "template_fields": template_fields,
    }


# ====================================================================
#  主入口
# ====================================================================

def run(
    index_name: str = "上证指数",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    主运行函数

    Args:
        index_name: 指数名称
        output_dir: 输出目录 (None = 仅返回结果不写文件)

    Returns:
        {"levels": dict, "status": dict, "markdown": str, "json": dict}
    """
    logger.info(f"=== 关键点位计算开始 === 指数: {index_name}")

    # Step 1: 获取数据
    df = fetch_index_data(index_name)
    if df is None or len(df) < 30:
        logger.error("数据不足，无法计算关键点位")
        return {"error": "数据不足"}

    # Step 2: 计算关键点位
    levels = calculate_key_levels(df)

    # Step 3: 状态判定
    status = determine_market_status(
        close_price=levels["close_price"],
        strong_support=levels["strong_support"],
        resistance_mid=levels["resistance_mid"],
        breakout_confirm=levels["breakout_confirm"],
        volume_ratio=levels["volume_ratio"],
    )

    # Step 4: 生成输出
    markdown = generate_markdown_report(index_name, levels, status, df)
    json_output = generate_json_output(levels, status)

    result = {
        "index_name": index_name,
        "success": True,
        "levels": levels,
        "status": status,
        "markdown": markdown,
        "json": json_output,
    }

    logger.info(f"=== 关键点位计算完成 === "
                f"当前点位: {levels['close_price']} | "
                f"状态: {status['status']} | "
                f"支撑: {levels['strong_support']} | "
                f"阻力: {levels['resistance_mid']}")

    # Step 5: 写文件 (可选)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Markdown 简报
        md_path = os.path.join(output_dir, f"key_levels_{index_name}_{ts}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info(f"Markdown 简报已写入: {md_path}")

        # JSON 数据
        json_path = os.path.join(output_dir, f"key_levels_{index_name}_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 数据已写入: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="关键点位计算器 — 为每日交易报告提供支撑位/阻力位/突破位",
    )
    parser.add_argument(
        "--index", "-i",
        default="上证指数",
        choices=list(INDEX_MAP.keys()),
        help="指数名称 (默认: 上证指数)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="输出目录 (不指定则仅打印到终端)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="仅输出 JSON (stdout)",
    )

    args = parser.parse_args()

    result = run(
        index_name=args.index,
        output_dir=args.output_dir,
    )

    if "error" in result:
        print(json.dumps({"error": result["error"]}, ensure_ascii=False))
        sys.exit(1)

    if args.json:
        print(json.dumps(result["json"], ensure_ascii=False, indent=2))
    else:
        print(result["markdown"])


if __name__ == "__main__":
    main()
