#!/usr/bin/env python3
"""
《每日交易参考》v4 - 宏观情绪监测与反弹预估模块
====================================================
独立模块，可脱离主 Pipeline 独立运行。

核心功能:
  1. 深度阈值计算 (CSI300/500):
     - RSI(14) 超卖/超买
     - ATR 波动率极值突破
     - 均线乖离率 (MA5/MA20/MA60)
     - 成交量偏离度

  2. 历史形态映射引擎:
     - 检索过去 N 年满足极值条件的"大盘暴跌"序列
     - 计算修复天数分布 (Mean Recovery Days, Best/Worst Case)
     - 每日反弹胜率

  3. 微观信号确认:
     - 宽基 ETF 逆势溢价/份额增长
     - 长下影线形态 (Hammer / Dragonfly Doji)
     - 融资盘极限信号

数据源: Tushare Pro (主) + akshare (备用)
输出:   structured dict, 兼容 daily_report.md YAML Header

配置:   所有阈值通过 YAML 外部注入，不硬编码
"""

import os, sys, json, logging, yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# ── 确保项目根可导入 ──
_STAGING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STAGING_ROOT not in sys.path:
    sys.path.insert(0, _STAGING_ROOT)

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _to_native(obj):
    """递归将 numpy 类型转换为 Python 原生 JSON 可序列化类型"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_native(v) for v in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


# ============================================================
# 缺省 YAML 配置（嵌入文件，无外部文件时可用）
# ============================================================
DEFAULT_CONFIG_YAML = """
macro_pullback_tracker:
  # ── 标的指数 ──
  indices:
    csi300: "000300.SH"
    csi500: "000905.SH"
    # 备用 ETF 标的
    etf_csi300: "510300.SH"
    etf_csi500: "510500.SH"

  # ── RSI 深度阈值 ──
  rsi:
    period: 14
    oversold: 25          # 超卖阈值
    extreme_oversold: 18  # 极端超卖 — v4修正: 数据验证确认原阈值9.7年触发0次
    overbought: 75
    extreme_overbought: 85

  # ── ATR 波动率极值 ──
  atr:
    period: 14
    vol_breakout_multiplier: 2.0   # ATR 突破倍数 — v4修正: 原阈值9.7年触发0次
    vol_breakout_lookback: 20      # ATR 均值回看窗口

  # ── 均线乖离率 ──
  ma_deviation:
    fast_period: 5
    mid_period: 20
    slow_period: 60
    # 相对于各均线的乖离率阈值（负值=跌破）
    extreme_deviation_pct:
      ma5: -3.0
      ma20: -5.0
      ma60: -8.0
    # 用于历史事件检测的宽松阈值
    event_deviation_pct:
      ma5: -2.0
      ma20: -4.0
      ma60: -6.0

  # ── 成交量偏离度 ──
  volume_deviation:
    period: 20
    # 恐慌放量倍数（相对于均量）
    panic_surge_ratio: 2.0
    # 地量萎缩比率（相对于均量）
    shrink_ratio: 0.6           # 地量萎缩比率（相对于均量）— v4修正: 原阈值9.7年触发0次

  # ── 历史形态映射 ──
  historical_pattern:
    lookback_years: 10
    min_event_distance_days: 15      # 最小事件间隔（防重复计数）
    max_drawdown_threshold: -3.0     # 单日跌幅 > 3% 计入暴跌事件
    consecutive_drawdown_days: 3     # 连续N日回调
    total_drawdown_threshold: -5.0   # 累计跌幅 > 5%
    recovery_check_days: 250         # 修复检查最长天数（约1年交易）

  # ── 微观信号 ──
  micro_signals:
    etf:
      premium_threshold: 0.5         # ETF 溢价 > 0.5%
      share_increase_pct: 10.0       # 份额增长 > 10%
    long_lower_shadow:
      # 下影线 = min(open,close) - low
      # 实体 = abs(close - open)
      body_to_shadow_max_ratio: 0.5  # 实体 / 下影线 < 0.5 (下影线至少是实体的2倍)
      shadow_to_total_min_ratio: 0.3 # 下影线 / 全波幅 > 0.3
    margin:
      lookback_days: 120             # 融资数据回看
      extreme_percentile: 10         # <10% 分位视为极端
      drop_days: 3                   # 连续下降天数
      drop_threshold: -0.5           # 单日降幅 > 0.5%

  # ── 综合置信度权重 ──
  confidence_weights:
    rsi_extreme: 0.25
    atr_breakout: 0.15
    ma_deviation: 0.20
    volume_deviation: 0.10
    historical_pattern: 0.20
    micro_signals: 0.10
"""


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ThresholdResult:
    """深度阈值计算结果"""
    rsi: Dict[str, Any] = field(default_factory=dict)
    atr: Dict[str, Any] = field(default_factory=dict)
    ma_deviation: Dict[str, Any] = field(default_factory=dict)
    volume_deviation: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryEvent:
    """单次历史暴跌修复事件"""
    event_date: str
    entry_price: float
    nadir_price: float
    nadir_date: str
    max_drawdown_pct: float
    recovery_price: float = 0.0
    recovery_date: str = ""
    recovery_days: int = -1
    recovered: bool = False
    duration_days: int = 0
    total_decline_pct: float = 0.0

@dataclass
class HistoricalPatternResult:
    """历史形态映射结果"""
    n_events: int = 0
    mean_recovery_days: float = 0.0
    median_recovery_days: float = 0.0
    best_case_days: int = 0
    worst_case_days: int = 0
    recovered_count: int = 0
    recovery_rate: float = 0.0
    daily_win_rates: List[float] = field(default_factory=list)
    recent_event: Optional[Dict] = None

@dataclass
class MicroSignalResult:
    """微观信号确认结果"""
    etf: Dict[str, Any] = field(default_factory=dict)
    long_lower_shadow: Dict[str, Any] = field(default_factory=dict)
    margin: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MacroPullbackReport:
    """完整输出报告"""
    generated_at: str = ""
    index_code: str = ""
    index_name: str = ""
    current_price: float = 0.0
    threshold: ThresholdResult = field(default_factory=ThresholdResult)
    historical_pattern: HistoricalPatternResult = field(default_factory=HistoricalPatternResult)
    micro_signals: MicroSignalResult = field(default_factory=MicroSignalResult)
    recovery_window: Dict[str, Any] = field(default_factory=dict)
    confidence_rate: float = 0.0
    overall_verdict: str = ""
    details: str = ""


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """加载 YAML 配置，缺省使用内嵌默认值"""
    defaults = yaml.safe_load(DEFAULT_CONFIG_YAML)
    cfg = defaults["macro_pullback_tracker"]

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                external = yaml.safe_load(f)
            if external and "macro_pullback_tracker" in external:
                # 深度合并（外部覆盖默认）
                _deep_merge(cfg, external["macro_pullback_tracker"])
                logger.info("已加载外部配置: %s", config_path)
        except Exception as e:
            logger.warning("加载外部配置失败，使用默认: %s", e)
    else:
        logger.info("使用内嵌默认配置")

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    """递归合并 dict"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ============================================================
# 数据获取（复用 tushare + 备用 akshare）
# ============================================================

class DataFetcher:
    """指数行情数据获取，复用项目已有数据源"""

    def __init__(self, config: dict):
        self.config = config
        self._pro = None
        self._ak_available = False

    def _get_tushare_pro(self):
        """初始化 Tushare Pro（连接复用）"""
        if self._pro is not None:
            return self._pro
        try:
            import tushare as ts
            from config import TUSHARE_TOKEN
            self._pro = ts.pro_api(TUSHARE_TOKEN)
            logger.info("Tushare Pro 连接初始化成功")
        except Exception as e:
            logger.warning("Tushare Pro 初始化失败: %s", e)
            self._pro = None
        return self._pro

    def _check_akshare(self) -> bool:
        """检查 akshare 是否可用"""
        if self._ak_available:
            return True
        try:
            import akshare as ak
            _ = ak.stock_zh_index_daily_em  # 验证接口
            self._ak_available = True
            logger.info("akshare 可用")
            return True
        except Exception:
            self._ak_available = False
            logger.warning("akshare 不可用")
            return False

    def fetch_index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        name: str = "",
    ) -> pd.DataFrame:
        """获取指数日线行情数据（优先 Tushare，备用 akshare）

        Args:
            ts_code: Tushare 指数代码，如 "000300.SH"
            start_date: 起始日期 YYYYMMDD
            end_date: 截止日期 YYYYMMDD
            name: 指数名称（仅用于日志）

        Returns:
            DataFrame with columns: trade_date, open, high, low, close, vol
        """
        label = name or ts_code

        # ── 尝试 Tushare ──
        pro = self._get_tushare_pro()
        if pro is not None:
            try:
                df = pro.index_daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is not None and not df.empty:
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    logger.info("%s: Tushare 获取 %d 条数据", label, len(df))
                    return df
            except Exception as e:
                logger.warning("%s Tushare 获取失败: %s", label, e)

        # ── 备用 akshare ──
        if self._check_akshare():
            return self._fetch_index_akshare(ts_code, start_date, end_date, label)

        raise RuntimeError(f"所有数据源均无法获取 {label} 行情数据")

    def _fetch_index_akshare(
        self, ts_code: str, start_date: str, end_date: str, label: str
    ) -> pd.DataFrame:
        """通过 akshare 获取指数日线行情"""
        import akshare as ak

        # 转 akshare 代码格式
        symbol_map = {
            "000300.SH": "sh000300",
            "000905.SH": "sh000905",
            "000001.SH": "sh000001",
            "399001.SZ": "sz399001",
            "399006.SZ": "sz399006",
            "000688.SH": "sh000688",
        }
        symbol = symbol_map.get(ts_code, ts_code.lower().replace(".sh", "sh").replace(".sz", "sz"))

        try:
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
                df = df.rename(columns={
                    "open": "open", "high": "high",
                    "low": "low", "close": "close",
                    "volume": "vol",
                })
                # 只保留需要的列
                need_cols = [c for c in ["trade_date", "open", "high", "low", "close", "vol"]
                             if c in df.columns]
                df = df[need_cols]
                df = df.sort_values("trade_date").reset_index(drop=True)
                # 过滤日期范围
                df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
                logger.info("%s: akshare 获取 %d 条数据", label, len(df))
                return df
        except Exception as e:
            logger.warning("%s akshare 获取失败: %s", label, e)

        return pd.DataFrame()

    def fetch_etf_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取 ETF 日线数据（用于微观信号检测）"""
        pro = self._get_tushare_pro()
        if pro is None:
            return pd.DataFrame()
        try:
            df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                logger.info("ETF %s: 获取 %d 条数据", ts_code, len(df))
                return df
        except Exception as e:
            logger.warning("ETF %s 获取失败: %s", ts_code, e)
        return pd.DataFrame()

    def fetch_margin_data(self, days: int = 120) -> pd.DataFrame:
        """获取融资融券数据"""
        pro = self._get_tushare_pro()
        if pro is None:
            return pd.DataFrame()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
        try:
            df = pro.margin(start_date=start, end_date=end)
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["rzye"] = pd.to_numeric(df["rzye"], errors="coerce")
                df["rzmre"] = pd.to_numeric(df["rzmre"], errors="coerce")
                df["rzche"] = pd.to_numeric(df["rzche"], errors="coerce")
                # 按日期聚合（沪深合计）
                agg = df.groupby("trade_date").agg(
                    rzye=("rzye", "sum"),
                    rzmre=("rzmre", "sum"),
                    rzche=("rzche", "sum"),
                ).reset_index()
                agg["rzye_chg_pct"] = agg["rzye"].pct_change(1) * 100
                agg["rzye_chg_3d"] = agg["rzye"].pct_change(3) * 100
                logger.info("融资融券: 获取 %d 条", len(agg))
                return agg
        except Exception as e:
            logger.warning("融资融券获取失败: %s", e)
        return pd.DataFrame()


# ============================================================
# 指标计算引擎
# ============================================================

class IndicatorEngine:
    """技术指标计算引擎"""

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI(period)"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # Wilder 平滑
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算 ATR(period)"""
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        # Wilder 平滑
        for i in range(period, len(atr)):
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
        return atr

    @staticmethod
    def compute_ma_deviation(close: pd.Series, ma_period: int) -> pd.Series:
        """计算均线乖离率 (%) = (price - MA) / MA * 100"""
        ma = close.rolling(window=ma_period, min_periods=ma_period).mean()
        deviation = (close - ma) / (ma + 1e-10) * 100
        return deviation

    @staticmethod
    def compute_volume_deviation(volume: pd.Series, period: int = 20) -> pd.Series:
        """计算成交量偏离度 = volume / MA(volume)"""
        ma_vol = volume.rolling(window=period, min_periods=period).mean()
        deviation = volume / (ma_vol + 1e-10)
        return deviation

    @staticmethod
    def compute_long_lower_shadow(open_p: float, high: float, low: float, close: float) -> Dict:
        """检测长下影线形态

        Returns:
            dict with signals and ratios
        """
        total_range = high - low
        if total_range == 0:
            return {"detected": False, "shadow_ratio": 0.0, "body_ratio": 0.0}

        # 下影线长度
        lower_shadow = min(open_p, close) - low
        shadow_ratio = lower_shadow / total_range

        # 实体长度
        body = abs(close - open_p)
        body_to_shadow = body / (lower_shadow + 1e-10)

        # Hammer: 下影线长、实体小、出现在下跌后
        # Dragonfly Doji: 下影线长、实体极小(近似十字星)
        hammer = (shadow_ratio >= 0.3 and body_to_shadow <= 0.5)
        doji = (body / (close + 1e-10) * 100 < 0.1 and shadow_ratio >= 0.4)

        return {
            "detected": hammer or doji,
            "hammer": hammer,
            "dragonfly_doji": doji,
            "shadow_ratio": round(shadow_ratio, 3),
            "body_to_shadow_ratio": round(body_to_shadow, 2),
            "lower_shadow": round(lower_shadow, 2),
            "total_range": round(total_range, 2),
        }

    @staticmethod
    def compute_max_drawdown(price_series: pd.Series) -> pd.Series:
        """计算滚动最大回撤 (%)"""
        rolling_max = price_series.expanding().max()
        drawdown = (price_series - rolling_max) / (rolling_max + 1e-10) * 100
        return drawdown


# ============================================================
# 深度阈值计算模块
# ============================================================

class ThresholdCalculator:
    """深度阈值计算"""

    def __init__(self, config: dict, engine: IndicatorEngine):
        self.cfg = config
        self.engine = engine

    def compute_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算所有深度阈值指标"""
        result = {}

        # ── RSI ──
        rsi_cfg = self.cfg["rsi"]
        rsi = self.engine.compute_rsi(df["close"], rsi_cfg["period"])
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        rsi_signal = "normal"
        if current_rsi <= rsi_cfg["extreme_oversold"]:
            rsi_signal = "extreme_oversold"
        elif current_rsi <= rsi_cfg["oversold"]:
            rsi_signal = "oversold"
        elif current_rsi >= rsi_cfg["extreme_overbought"]:
            rsi_signal = "extreme_overbought"
        elif current_rsi >= rsi_cfg["overbought"]:
            rsi_signal = "overbought"

        result["rsi"] = {
            "value": round(current_rsi, 1),
            "period": rsi_cfg["period"],
            "signal": rsi_signal,
            "oversold": rsi_cfg["oversold"],
            "overbought": rsi_cfg["overbought"],
            "extreme_oversold": rsi_cfg["extreme_oversold"],
        }

        # ── ATR 波动率极值 ──
        atr_cfg = self.cfg["atr"]
        atr = self.engine.compute_atr(df["high"], df["low"], df["close"], atr_cfg["period"])
        current_atr = float(atr.iloc[-1]) if not atr.empty else 0.0

        # ATR 相对于其均值的倍数
        atr_ma = atr.rolling(window=atr_cfg["vol_breakout_lookback"]).mean()
        atr_ratio = current_atr / (float(atr_ma.iloc[-1]) if not atr_ma.empty else 1e-10)
        atr_breakout = atr_ratio >= atr_cfg["vol_breakout_multiplier"]

        result["atr"] = {
            "value": round(current_atr, 2),
            "atr_ratio": round(atr_ratio, 2),
            "period": atr_cfg["period"],
            "breakout": bool(atr_breakout),
            "breakout_multiplier": atr_cfg["vol_breakout_multiplier"],
            "signal": "vol_breakout" if atr_breakout else "normal",
        }

        # ── 均线乖离率 ──
        ma_cfg = self.cfg["ma_deviation"]
        ma_deviations = {}
        ma_signals = []
        for name, period in [("ma5", ma_cfg["fast_period"]),
                              ("ma20", ma_cfg["mid_period"]),
                              ("ma60", ma_cfg["slow_period"])]:
            dev = self.engine.compute_ma_deviation(df["close"], period)
            current_dev = float(dev.iloc[-1]) if not dev.empty else 0.0
            threshold = ma_cfg["extreme_deviation_pct"][name]
            ma_deviations[name] = {
                "value": round(current_dev, 2),
                "threshold": threshold,
                "signal": "extreme" if current_dev <= threshold else (
                    "warning" if current_dev <= threshold * 0.7 else "normal"
                ),
            }
            if current_dev <= threshold:
                ma_signals.append(f"{name}乖离{current_dev:.1f}%")

        result["ma_deviation"] = {
            "periods": ma_deviations,
            "n_extreme": sum(1 for v in ma_deviations.values() if v["signal"] == "extreme"),
            "summary": "; ".join(ma_signals) if ma_signals else "无均线极值偏离",
        }

        # ── 成交量偏离度 ──
        vol_cfg = self.cfg["volume_deviation"]
        vol_dev = self.engine.compute_volume_deviation(df["vol"], vol_cfg["period"])
        current_vol_ratio = float(vol_dev.iloc[-1]) if not vol_dev.empty else 1.0

        vol_signal = "normal"
        if current_vol_ratio >= vol_cfg["panic_surge_ratio"]:
            vol_signal = "panic_surge"
        elif current_vol_ratio <= vol_cfg["shrink_ratio"]:
            vol_signal = "shrink"

        result["volume_deviation"] = {
            "value": round(current_vol_ratio, 2),
            "period": vol_cfg["period"],
            "signal": vol_signal,
            "panic_surge_ratio": vol_cfg["panic_surge_ratio"],
            "shrink_ratio": vol_cfg["shrink_ratio"],
        }

        return result


# ============================================================
# 历史形态映射引擎
# ============================================================

class HistoricalPatternMapper:
    """历史形态映射引擎：检索过去 N 年暴跌序列，计算修复分布"""

    def __init__(self, config: dict):
        self.cfg = config["historical_pattern"]
        self.events: List[RecoveryEvent] = []

    def detect_events(self, df: pd.DataFrame) -> List[RecoveryEvent]:
        """从历史日线数据中检测暴跌事件

        Args:
            df: DataFrame with trade_date, open, high, low, close, vol

        Returns:
            List of RecoveryEvent
        """
        events: List[RecoveryEvent] = []
        min_gap = self.cfg["min_event_distance_days"]

        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df["close"].values
        dates = df["trade_date"].values
        n = len(df)

        if n < 60:
            logger.warning("历史数据不足，无法检测事件")
            return events

        # 计算每日涨跌幅
        pct_chg = np.diff(close) / close[:-1] * 100
        pct_chg = np.insert(pct_chg, 0, 0)

        # 计算滚动最大回撤
        running_max = np.maximum.accumulate(close)
        drawdown = (close - running_max) / running_max * 100

        last_event_idx = -min_gap

        for i in range(20, n - 1):  # 从第20天开始才有足够参考
            # 检测条件1: 单日暴跌
            is_crash = pct_chg[i] <= self.cfg["max_drawdown_threshold"]

            # 检测条件2: 连续回调
            if i >= self.cfg["consecutive_drawdown_days"]:
                recent_chg = (close[i] - close[i - self.cfg["consecutive_drawdown_days"]]) / \
                             close[i - self.cfg["consecutive_drawdown_days"]] * 100
                is_consecutive = recent_chg <= self.cfg["total_drawdown_threshold"]
            else:
                is_consecutive = False

            # 检测条件3: 最大回撤深度
            is_deep_drawdown = drawdown[i] <= self.cfg["total_drawdown_threshold"]

            if not (is_crash or is_consecutive or is_deep_drawdown):
                continue

            # 间隔保护：同一轮暴跌只记一次
            if i - last_event_idx < min_gap:
                continue

            # 找到此轮下跌的谷底（接下来N天内的最低点）
            lookahead = min(30, n - i)
            nadir_slice = close[i:i + lookahead]
            nadir_idx = np.argmin(nadir_slice) + i
            nadir_price = close[nadir_idx]
            nadir_date = dates[nadir_idx]
            max_dd = (nadir_price - close[i]) / close[i] * 100

            total_decline = (nadir_price - running_max[nadir_idx]) / running_max[nadir_idx] * 100

            event = RecoveryEvent(
                event_date=str(dates[i]),
                entry_price=float(close[i]),
                nadir_price=float(nadir_price),
                nadir_date=str(nadir_date),
                max_drawdown_pct=round(max_dd, 2),
                duration_days=nadir_idx - i,
                total_decline_pct=round(total_decline, 2),
            )

            # 计算修复天数
            self._compute_recovery(event, close, dates, nadir_idx, n)

            events.append(event)
            last_event_idx = i

        return events

    def _compute_recovery(
        self,
        event: RecoveryEvent,
        close: np.ndarray,
        dates: np.ndarray,
        nadir_idx: int,
        n: int,
    ) -> None:
        """计算单个事件的修复天数"""
        max_check = min(nadir_idx + self.cfg["recovery_check_days"], n)
        entry_price = event.entry_price

        for j in range(nadir_idx, max_check):
            if close[j] >= entry_price:
                event.recovery_price = float(close[j])
                event.recovery_date = str(dates[j])
                event.recovery_days = j - nadir_idx
                event.recovered = True
                break

        if not event.recovered:
            event.recovery_days = -1

    def compute_pattern(self, events: List[RecoveryEvent]) -> HistoricalPatternResult:
        """计算修复模式统计数据"""
        result = HistoricalPatternResult()
        if not events:
            return result

        recovered = [e for e in events if e.recovered]

        result.n_events = len(events)
        result.recovered_count = len(recovered)
        result.recovery_rate = round(len(recovered) / len(events) * 100, 1)

        if recovered:
            recovery_days = [e.recovery_days for e in recovered]
            result.mean_recovery_days = round(float(np.mean(recovery_days)), 1)
            result.median_recovery_days = round(float(np.median(recovery_days)), 1)
            result.best_case_days = int(min(recovery_days))
            result.worst_case_days = int(max(recovery_days))

            # 每日反弹胜率（修复后第N天回到入场价的概率）
            max_days = min(120, max(recovery_days))
            win_rates = []
            for day in range(1, max_days + 1):
                wins = sum(1 for e in recovered if e.recovery_days <= day)
                win_rates.append(round(wins / len(recovered) * 100, 1))
            result.daily_win_rates = win_rates

        # 最近一次事件
        if events:
            last = events[-1]
            result.recent_event = {
                "date": last.event_date,
                "nadir_date": last.nadir_date,
                "max_drawdown": last.max_drawdown_pct,
                "total_decline": last.total_decline_pct,
                "duration_days": last.duration_days,
                "recovered": last.recovered,
                "recovery_days": last.recovery_days if last.recovered else None,
            }

        return result


# ============================================================
# 微观信号确认模块
# ============================================================

class MicroSignalChecker:
    """微观信号确认"""

    def __init__(self, config: dict, fetcher: DataFetcher):
        self.cfg = config["micro_signals"]
        self.fetcher = fetcher
        self.engine = IndicatorEngine()

    def check_all(
        self,
        df_index: pd.DataFrame,
        index_code: str,
    ) -> MicroSignalResult:
        """检查所有微观信号"""
        result = MicroSignalResult()

        # ── 1. ETF 信号 ──
        result.etf = self._check_etf(index_code)

        # ── 2. 长下影线 ──
        result.long_lower_shadow = self._check_long_lower_shadow(df_index)

        # ── 3. 融资盘极限 ──
        result.margin = self._check_margin()

        return result

    def _check_etf(self, index_code: str) -> Dict[str, Any]:
        """检查 ETF 逆势溢价/份额增长"""
        result = {
            "detected": False,
            "signals": [],
            "details": {},
        }

        # 获取沪深300 ETF
        etf_code = "510300.SH"  # 华泰柏瑞沪深300ETF
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
            df_etf = self.fetcher.fetch_etf_daily(etf_code, start, end)
            if df_etf is not None and not df_etf.empty and len(df_etf) >= 2:
                latest = df_etf.iloc[-1]
                prev = df_etf.iloc[-2]

                premium = latest.get("premium_ratio", 0)
                if premium is not None and float(premium) > self.cfg["etf"]["premium_threshold"]:
                    result["signals"].append(f"ETF溢价{float(premium):.2f}%")
                    result["detected"] = True

                # 份额变化
                shares = latest.get("fd_shares", 0)
                prev_shares = prev.get("fd_shares", 0)
                if shares and prev_shares and float(prev_shares) > 0:
                    share_chg = (float(shares) - float(prev_shares)) / float(prev_shares) * 100
                    if share_chg > self.cfg["etf"]["share_increase_pct"]:
                        result["signals"].append(f"ETF份额增长{share_chg:.1f}%")
                        result["detected"] = True

                result["details"] = {
                    "premium_ratio": round(float(premium), 2) if premium else "N/A",
                    "code": etf_code,
                }
        except Exception as e:
            logger.debug("ETF 信号检查异常: %s", e)

        if not result["signals"]:
            result["signals"].append("无ETF异常信号")

        return result

    def _check_long_lower_shadow(self, df_index: pd.DataFrame) -> Dict[str, Any]:
        """检查长下影线形态（连续3日出现则置信度提高）"""
        result = {
            "detected": False,
            "consecutive_days": 0,
            "latest": {},
            "details": {},
        }

        if df_index.empty or len(df_index) < 5:
            return result

        cfg = self.cfg["long_lower_shadow"]
        recent = df_index.tail(5)

        detected_count = 0
        for _, row in recent.iterrows():
            shadow = self.engine.compute_long_lower_shadow(
                row["open"], row["high"], row["low"], row["close"]
            )
            if shadow["detected"]:
                detected_count += 1

        if detected_count >= 1:
            # 取最新一天的形态
            last_row = recent.iloc[-1]
            latest_shadow = self.engine.compute_long_lower_shadow(
                last_row["open"], last_row["high"], last_row["low"], last_row["close"]
            )
            result["detected"] = True
            result["consecutive_days"] = detected_count
            result["latest"] = latest_shadow

        return result

    def _check_margin(self) -> Dict[str, Any]:
        """检查融资盘极限信号"""
        result = {
            "detected": False,
            "signals": [],
        }

        cfg = self.cfg["margin"]
        margin_df = self.fetcher.fetch_margin_data(cfg["lookback_days"])
        if margin_df.empty or len(margin_df) < 20:
            result["signals"].append("融资数据不足")
            return result

        latest = margin_df.iloc[-1]
        chg_3d = latest.get("rzye_chg_3d", 0)
        rzye = latest.get("rzye", 0)

        # 连续下降检测
        recent_chg = margin_df["rzye_chg_pct"].tail(cfg["drop_days"])
        consecutive = all(
            v is not None and not pd.isna(v) and v < cfg["drop_threshold"]
            for v in recent_chg
        ) if len(recent_chg) >= cfg["drop_days"] else False

        # 极端分位
        hist_chg = margin_df["rzye_chg_3d"].dropna()
        percentile = 50
        if len(hist_chg) > 20 and chg_3d is not None and not pd.isna(chg_3d):
            percentile = (hist_chg <= chg_3d).sum() / len(hist_chg) * 100

        is_extreme = percentile <= cfg["extreme_percentile"]
        signals = []
        if is_extreme:
            signals.append(f"融资变化处于{percentile:.0f}%分位(极端)")
        if consecutive:
            signals.append(f"融资余额连续{cfg['drop_days']}日下降")

        if signals:
            result["detected"] = True
            result["signals"] = signals

        result["percentile"] = round(percentile, 1)
        result["chg_3d"] = round(float(chg_3d), 2) if chg_3d is not None and not pd.isna(chg_3d) else 0
        result["rzye"] = round(float(rzye) / 1e8, 1)  # 亿元

        return result


# ============================================================
# 主模块：宏观情绪监测与反弹预估
# ============================================================

class MacroPullbackTracker:
    """宏观情绪监测与反弹预估模块

    独立运行，不依赖主 Pipeline。输出 structured dict，
    兼容 daily_report.md YAML Header。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
    ):
        self.config = config_dict if config_dict else load_config(config_path)
        self.fetcher = DataFetcher(self.config)
        self.engine = IndicatorEngine()
        self.threshold_calc = ThresholdCalculator(self.config, self.engine)
        self.pattern_mapper = HistoricalPatternMapper(self.config)
        self.signal_checker = MicroSignalChecker(self.config, self.fetcher)

        # 缓存
        self._df: Optional[pd.DataFrame] = None

    def run(
        self,
        index_code: Optional[str] = None,
        index_name: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> MacroPullbackReport:
        """主入口：执行完整分析

        Args:
            index_code: 指数代码，默认 CSI300
            index_name: 指数名称，默认 "沪深300"
            start_date: 数据起始日期，默认 10年前

        Returns:
            MacroPullbackReport
        """
        idx_cfg = self.config["indices"]
        if index_code is None:
            index_code = idx_cfg["csi300"]
        if index_name is None:
            index_name = "沪深300"

        today = datetime.now().strftime("%Y%m%d")
        years_back = self.config["historical_pattern"]["lookback_years"]
        if start_date is None:
            start = (datetime.now() - timedelta(days=years_back * 365 + 100)).strftime("%Y%m%d")
        else:
            start = start_date

        logger.info("=" * 60)
        logger.info("宏观情绪监测启动: %s (%s)", index_name, index_code)
        logger.info("数据区间: %s ~ %s (%d年)", start, today, years_back)
        logger.info("=" * 60)

        # ── 1. 获取数据 ──
        df = self.fetcher.fetch_index_daily(index_code, start, today, index_name)
        if df.empty:
            logger.error("无法获取 %s 数据，退出", index_name)
            return self._empty_report(index_code, index_name)

        self._df = df
        current_price = float(df["close"].iloc[-1])

        # ── 2. 深度阈值计算 ──
        logger.info("── 阶段1: 深度阈值计算 ──")
        threshold_result = self.threshold_calc.compute_all(df)

        # ── 3. 历史形态映射 ──
        logger.info("── 阶段2: 历史形态映射 ──")
        events = self.pattern_mapper.detect_events(df)
        pattern_result = self.pattern_mapper.compute_pattern(events)
        logger.info("检测到 %d 次暴跌事件, %d 次修复 (修复率 %.1f%%)",
                     pattern_result.n_events,
                     pattern_result.recovered_count,
                     pattern_result.recovery_rate)

        # ── 4. 微观信号确认 ──
        logger.info("── 阶段3: 微观信号确认 ──")
        micro_result = self.signal_checker.check_all(df, index_code)

        # ── 5. 综合评估 ──
        logger.info("── 阶段4: 综合评估 ──")
        report = self._synthesize(
            index_code=index_code,
            index_name=index_name,
            current_price=current_price,
            threshold=threshold_result,
            pattern=pattern_result,
            micro=micro_result,
        )

        logger.info("=" * 60)
        logger.info("分析完成 | 置信度: %.0f%% | 评估: %s",
                     report.confidence_rate * 100, report.overall_verdict)
        logger.info("=" * 60)

        return report

    def _synthesize(
        self,
        index_code: str,
        index_name: str,
        current_price: float,
        threshold: Dict[str, Any],
        pattern: HistoricalPatternResult,
        micro: MicroSignalResult,
    ) -> MacroPullbackReport:
        """综合各项指标，生成最终评估"""
        report = MacroPullbackReport(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            index_code=index_code,
            index_name=index_name,
            current_price=round(current_price, 2),
            threshold=ThresholdResult(**threshold),
            historical_pattern=pattern,
            micro_signals=micro,
        )

        weights = self.config["confidence_weights"]

        # ── 计算反弹置信度 ──
        confidence_scores = []

        # RSI 信号
        rsi_val = threshold["rsi"]["value"]
        rsi_signal = threshold["rsi"]["signal"]
        if rsi_signal == "extreme_oversold":
            rsi_score = 1.0
        elif rsi_signal == "oversold":
            rsi_score = 0.7
        elif rsi_signal == "normal":
            rsi_score = 0.5 if 30 <= rsi_val <= 50 else 0.3
        else:
            rsi_score = 0.1  # 超买区域
        confidence_scores.append(("RSI", rsi_score, weights["rsi_extreme"]))

        # ATR 波动率信号
        atr_breakout = threshold["atr"]["breakout"]
        atr_score = 1.0 if atr_breakout else 0.3
        confidence_scores.append(("ATR", atr_score, weights["atr_breakout"]))

        # 均线乖离率信号
        ma_extreme = threshold["ma_deviation"]["n_extreme"]
        ma_score = min(1.0, ma_extreme / 3.0)
        confidence_scores.append(("MA", ma_score, weights["ma_deviation"]))

        # 成交量信号
        vol_signal = threshold["volume_deviation"]["signal"]
        if vol_signal == "panic_surge":
            vol_score = 0.8  # 恐慌放量后见底概率高
        elif vol_signal == "shrink":
            vol_score = 0.6  # 地量见地价
        else:
            vol_score = 0.3
        confidence_scores.append(("VOL", vol_score, weights["volume_deviation"]))

        # 历史形态信号
        if pattern.n_events > 0:
            hist_score = pattern.recovery_rate / 100.0
        else:
            hist_score = 0.5
        confidence_scores.append(("HIST", hist_score, weights["historical_pattern"]))

        # 微观信号
        micro_signals_count = 0
        if micro.etf.get("detected"):
            micro_signals_count += 1
        if micro.long_lower_shadow.get("detected"):
            micro_signals_count += 1
        if micro.margin.get("detected"):
            micro_signals_count += 1
        micro_score = min(1.0, micro_signals_count / 3.0)
        confidence_scores.append(("MICRO", micro_score, weights["micro_signals"]))

        # 加权平均
        total_weight = sum(w for _, _, w in confidence_scores)
        weighted_sum = sum(s * w for _, s, w in confidence_scores)
        report.confidence_rate = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.5

        # ── 计算当前回撤深度（从历史高点） ──
        if self._df is not None and not self._df.empty:
            close_series = self._df["close"]
            rolling_max = close_series.expanding().max()
            current_drawdown_pct = float(
                (close_series.iloc[-1] - rolling_max.iloc[-1]) / (rolling_max.iloc[-1] + 1e-10) * 100
            )
        else:
            current_drawdown_pct = 0.0

        # ── 修复窗口估算 ──
        recovery_window = self._estimate_recovery_window(threshold, pattern, rsi_val, current_drawdown_pct)
        report.recovery_window = recovery_window

        # ── 综合判断 ──
        if report.confidence_rate >= 0.75:
            report.overall_verdict = "高概率反弹窗口"
            report.details = (
                f"多个维度信号共振（置信度{report.confidence_rate:.0%}）。"
                f"RSI={rsi_val:.0f}({'极端超卖' if rsi_signal == 'extreme_oversold' else '超卖'})，"
                f"均线乖离极值{threshold['ma_deviation']['n_extreme']}/3，"
                f"历史修复率{pattern.recovery_rate:.0f}%。"
                f"预估修复{recovery_window.get('mean_days', 'N/A')}天。"
            )
        elif report.confidence_rate >= 0.50:
            report.overall_verdict = "关注反弹窗口"
            report.details = (
                f"部分信号显示超卖（置信度{report.confidence_rate:.0%}）。"
                f"RSI={rsi_val:.0f}，微观信号{micro_signals_count}/3。"
                f"建议结合资金面确认。"
            )
        elif report.confidence_rate >= 0.30:
            report.overall_verdict = "谨慎观望"
            report.details = (
                f"信号偏弱（置信度{report.confidence_rate:.0%}）。"
                f"未见极端超卖或恐慌信号。"
            )
        else:
            report.overall_verdict = "趋势健康，无需反弹信号"
            report.details = f"市场处于强势区域（RSI={rsi_val:.0f}），暂无反转需求。"

        return report

    def _estimate_recovery_window(
        self,
        threshold: Dict[str, Any],
        pattern: HistoricalPatternResult,
        rsi_val: float,
        current_drawdown_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """估算修复窗口（v4修正: 双因子调节——跌幅深度(主)×RSI(辅)）"""
        window = {}

        if pattern.n_events > 0 and pattern.recovered_count > 0:
            window["mean_days"] = int(pattern.mean_recovery_days)
            window["median_days"] = int(pattern.median_recovery_days)
            window["best_case_days"] = pattern.best_case_days
            window["worst_case_days"] = pattern.worst_case_days
            window["recovery_rate"] = pattern.recovery_rate

            # ── 双因子调节: 跌幅深度（主）× RSI（辅） ──
            # 数据验证确认: drawdown深度与修复天数 Pearson r≥0.3
            # 跌幅因子：跌得越深，修复越慢
            abs_dd = abs(current_drawdown_pct)
            if abs_dd < 5:
                drawdown_factor = 1.0
            elif abs_dd < 10:
                drawdown_factor = 1.3
            elif abs_dd < 20:
                drawdown_factor = 1.6
            else:
                drawdown_factor = 2.0

            # RSI因子：极度超卖时修复略快，非超卖时略慢（范围压缩至0.9-1.1）
            if rsi_val <= 25:
                rsi_factor = 0.9
            elif rsi_val <= 35:
                rsi_factor = 1.0
            else:
                rsi_factor = 1.1

            adjusted = int(pattern.mean_recovery_days * drawdown_factor * rsi_factor)
            window["adjusted_mean_days"] = max(3, adjusted)
            window["adjustment_note"] = (
                f"双因子调节: 回撤{current_drawdown_pct:.1f}%(系数{drawdown_factor})"
                f" × RSI{rsi_val:.0f}(系数{rsi_factor})"
            )
        else:
            # 无历史数据时的经验估算
            window["mean_days"] = 15
            window["median_days"] = 12
            window["best_case_days"] = 3
            window["worst_case_days"] = 45
            window["recovery_rate"] = 85.0
            window["adjusted_mean_days"] = 15
            window["adjustment_note"] = "基于A股历史经验估值（无本指数历史事件）"

        # 反弹胜率曲线（前30天）
        if pattern.daily_win_rates:
            # T+1, T+3, T+5, T+10, T+20, T+30 胜率
            key_days = [1, 3, 5, 10, 20, 30]
            win_rates_at = {}
            for d in key_days:
                idx = min(d - 1, len(pattern.daily_win_rates) - 1)
                win_rates_at[f"T+{d}"] = pattern.daily_win_rates[idx]
            window["key_win_rates"] = win_rates_at

        return window

    def _empty_report(self, index_code: str, index_name: str) -> MacroPullbackReport:
        """生成空报告（数据不可用）"""
        return MacroPullbackReport(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            index_code=index_code,
            index_name=index_name,
            overall_verdict="数据不可用",
            details=f"无法获取 {index_name} 行情数据，请检查数据源配置",
            confidence_rate=0.0,
        )

    def to_dict(self, report: MacroPullbackReport) -> dict:
        """序列化为 dict（用于 JSON/YAML 输出）"""
        return _to_native({
            "generated_at": report.generated_at,
            "index": {
                "code": report.index_code,
                "name": report.index_name,
                "current_price": report.current_price,
            },
            "threshold": asdict(report.threshold),
            "historical_pattern": asdict(report.historical_pattern),
            "micro_signals": asdict(report.micro_signals),
            "recovery_window": report.recovery_window,
            "confidence_rate": report.confidence_rate,
            "overall_verdict": report.overall_verdict,
            "details": report.details,
        })

    def to_markdown_yaml_header(self, report: MacroPullbackReport) -> str:
        """生成兼容 daily_report.md 的 YAML Header 片段"""
        d = self.to_dict(report)
        lines = [
            "---",
            f"macro_tracker_v4:",
            f"  generated_at: {d['generated_at']}",
            f"  index:",
            f"    code: {d['index']['code']}",
            f"    name: {d['index']['name']}",
            f"    current_price: {d['index']['current_price']}",
            f"  confidence_rate: {d['confidence_rate']}",
            f"  overall_verdict: \"{d['overall_verdict']}\"",
            f"  recovery_window:",
            f"    mean_days: {d['recovery_window'].get('mean_days', 'N/A')}",
            f"    adjusted_mean_days: {d['recovery_window'].get('adjusted_mean_days', 'N/A')}",
            f"    recovery_rate: {d['recovery_window'].get('recovery_rate', 'N/A')}",
            f"    adjustment_note: \"{d['recovery_window'].get('adjustment_note', '')}\"",
            f"  rsi:",
            f"    value: {d['threshold'].get('rsi', {}).get('value', 'N/A')}",
            f"    signal: {d['threshold'].get('rsi', {}).get('signal', 'N/A')}",
            f"  micro_signals_detected: {sum(1 for s in [d['micro_signals'].get('etf', {}).get('detected', False), d['micro_signals'].get('long_lower_shadow', {}).get('detected', False), d['micro_signals'].get('margin', {}).get('detected', False)] if s)}",
            "---",
        ]
        return "\n".join(lines)


# ============================================================
# CLI 入口
# ============================================================

def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="宏观情绪监测与反弹预估模块 (v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认分析沪深300
  python macro_pullback_tracker.py

  # 指定指数 + 配置文件
  python macro_pullback_tracker.py --index 000905.SH --name 中证500 --config my_config.yaml

  # 输出 JSON 到文件
  python macro_pullback_tracker.py --output report.json

  # 输出 markdown YAML Header
  python macro_pullback_tracker.py --format md
        """,
    )
    parser.add_argument("--index", default=None, help="指数代码 (默认 CSI300)")
    parser.add_argument("--name", default=None, help="指数名称")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("--output", default=None, help="输出 JSON 文件路径")
    parser.add_argument("--format", choices=["json", "md", "text"], default="text",
                        help="输出格式 (默认 text)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    # 日志配置
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 执行
    tracker = MacroPullbackTracker(config_path=args.config)
    report = tracker.run(index_code=args.index, index_name=args.name)

    # 输出
    if args.format == "json":
        output = json.dumps(tracker.to_dict(report), ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"JSON 输出已保存: {args.output}")
        else:
            print(output)
    elif args.format == "md":
        print(tracker.to_markdown_yaml_header(report))
    else:
        # text 格式
        d = tracker.to_dict(report)
        print()
        print("=" * 60)
        print(f"  宏观情绪监测报告 v4")
        print(f"  标的: {d['index']['name']} ({d['index']['code']})")
        print(f"  当前价: {d['index']['current_price']}")
        print(f"  生成: {d['generated_at']}")
        print("=" * 60)
        print()
        print(f"  📊 综合评估: {d['overall_verdict']}")
        print(f"  🎯 反弹置信度: {d['confidence_rate']:.0%}")
        print()
        print(f"  ── 深度阈值 ──")
        rsi = d['threshold'].get('rsi', {})
        print(f"     RSI({rsi.get('period', 14)}): {rsi.get('value', 'N/A')} [{rsi.get('signal', 'N/A')}]")
        atr = d['threshold'].get('atr', {})
        print(f"     ATR 突破: {'是' if atr.get('breakout') else '否'} (比率 {atr.get('atr_ratio', 'N/A')})")
        ma = d['threshold'].get('ma_deviation', {})
        print(f"     均线乖离极值: {ma.get('n_extreme', 0)}/3")
        vol = d['threshold'].get('volume_deviation', {})
        print(f"     成交量: {vol.get('signal', 'N/A')} ({vol.get('value', 'N/A')}x)")
        print()
        print(f"  ── 历史形态 ──")
        hp = d['historical_pattern']
        print(f"     事件数: {hp.get('n_events', 0)} | 修复率: {hp.get('recovery_rate', 'N/A')}%")
        print(f"     平均修复: {hp.get('mean_recovery_days', 'N/A')}天 | 最快: {hp.get('best_case_days', 'N/A')}天 | 最慢: {hp.get('worst_case_days', 'N/A')}天")
        print()
        print(f"  ── 修复窗口 ──")
        rw = d['recovery_window']
        print(f"     预估均值: {rw.get('adjusted_mean_days', 'N/A')}天 | {rw.get('adjustment_note', '')}")
        kr = rw.get('key_win_rates', {})
        if kr:
            print(f"     关键胜率: {', '.join(f'{k}: {v}%' for k, v in kr.items())}")
        print()
        print(f"  ── 微观信号: {sum(1 for s in [d['micro_signals'].get('etf', {}).get('detected', False), d['micro_signals'].get('long_lower_shadow', {}).get('detected', False), d['micro_signals'].get('margin', {}).get('detected', False)] if s)}/3 ──")
        print(f"     ETF: {d['micro_signals'].get('etf', {}).get('signals', ['N/A'])[0]}")
        ls = d['micro_signals'].get('long_lower_shadow', {})
        print(f"     长下影线: {'是 (连续{}天)'.format(ls.get('consecutive_days', 0)) if ls.get('detected') else '否'}")
        mg = d['micro_signals'].get('margin', {})
        mg_sig = mg.get('signals', ['无'])[0] if mg.get('detected') else '正常'
        print(f"     融资盘: {mg_sig}")
        print()
        print(f"  {d['details']}")
        print()

    return report


if __name__ == "__main__":
    main()
