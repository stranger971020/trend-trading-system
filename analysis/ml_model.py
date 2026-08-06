from __future__ import annotations
"""
ML 评分模型 V4 — 三分组 LightGBM + 全量特征体系

# ── Changelog ──
# 2026-07-29 Claude: 修复静态缓存导致每日 Top5 雷同
#               根因: _rerank_via_cache 用 cache.groupby().last() 取
#                     训练时产生的静态 parquet 文件，每日评分不变
#               修复: 添加日期新鲜度检查，缓存日期<当前日期时
#                     回退到 _rerank_v3_online 用当前数据在线构建特征
#               同步: _rerank_v3_online 修复 preds 索引 bug（多股票时
#                     len(dict)计数式索引导致预测值错位），改为每股票
#                     只取最新日期行的 idxmax 方式；评分归一化改为与
#                     V4 路径一致的截面百分位方式（替代固定 ms*5+5）
# ─────────────

=== 数据源（全部来自现有 DB，零新增 API）===
stock_daily         → 全量 A 股约 4999 只 × ~295 天 OHLCV + amount + pct_chg
sw_index_daily(L1)  → 31 个一级行业指数
sw_l2_index_daily   → 123 个二级行业指数
sw_l3_index_daily   → 258 个三级行业指数
margin_cache        → 4457 只个股融资融券（25 个交易日）
moneyflow_cache     → 473 只主力资金流
fundamental_cache   → 479 只 PE/PB/ROE/总市值
financial_quality   → 326 只季度财务（2025年报）

=== 特征分类 ===
动量(11) | 均线(8) | 量/额(10) | 波动率/风险(10) | 价格形态(13)
K线形态(9) | 筹码位置(5) | 行业层级(7) | 截面排名(11) | Beta/市场相对(6)
时间序列统计(7) | 流动性微观(5) | 融资融券(3) | 资金流向(4) | 基本面(13)
财务质量比率(6) | 交叉特征(7)  | 板块属性(1)
合计 ~135+ 特征
"""

import logging
import os
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from config import (
    DB_PATH,
    DATA_DIR,
    BEIJING_TZ_OFFSET,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOL,
    COL_AMOUNT,
    RSI_PERIOD,
)

logger = logging.getLogger(__name__)

# ── 路径 ──
MODEL_DIR = os.path.join(DATA_DIR, "lgb_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 全量池扩池 (2026-08-05): 未分类股票行业特征用沪深300市场指数代理
MARKET_INDEX_CODE = "000300.SH"
UNCLASSIFIED_L1_CODE = "UNCLASSIFIED"
MODEL_PATHS = {
    "micro": os.path.join(MODEL_DIR, "lgb_micro.pkl"),
    "mid":   os.path.join(MODEL_DIR, "lgb_mid.pkl"),
}

# ═══════════════════════════════════════════════════════════════
# 全量特征清单
# ═══════════════════════════════════════════════════════════════

# ── 动量类 (11) ──
MOMENTUM_FEATURES = [
    "mom1", "mom5", "mom10", "mom20", "mom60",
    "mom_ratio_5_20",         # mom5 / mom20
    "vol_adjusted_mom20",     # mom20 / atr_pct
    "exp_wgt_return_20d",     # 换手率加权指数衰减
    "mom_accel",              # mom5 - mom20
    "upside_downside_ratio",  # 涨日平均收益 / 跌日平均收益
    "momentum_consistency",   # 最近10日方向连续天数
]

# ── 均线类 (8) ──
MA_FEATURES = [
    "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev", "ma120_dev", "ma250_dev",
    "ma_cross",           # MA5 > MA20 = 1 (金叉)
    "ma_arrangement",     # 多头排列评分 (0-4)
]

# ── 成交量/额类 (10) ──
VOLUME_FEATURES = [
    "vol_ratio",               # vol / MA20(vol)
    "vol_ma5_ratio",           # vol / MA5(vol)
    "vol_shock",               # vol[i] / vol[i-1]
    "volume_oscillator",       # (MA5vol - MA20vol) / MA20vol
    "amount_ratio",            # amount / MA20(amount)
    "amount_trend",            # MA5(amount) / MA20(amount)
    "turnover_ratio_5d_20d",   # 短期换手率比
    "turnover_std_20d",        # 换手率标准差(CV)
    "vol_amplitude_ratio",     # vol_ratio / atr_pct
    "vol_conviction",          # vol_ratio * ma20_dev (交叉)
]

# ── 波动率/风险类 (9) ──
VOLATILITY_FEATURES = [
    "atr_pct",              # ATR(14) / close %
    "close_position",       # (close - low) / (high - low)
    "daily_range",          # (high - low) / close
    "yang_zhang_vol",       # Yang-Zhang 波动率估计
    "sharpe_20d",           # 20日夏普比
    "max_drawdown_20d",     # 20日最大回撤
    "skewness_20d",         # 20日收益偏度
    "kurtosis_20d",         # 20日收益峰度
    "downside_vol",         # 下行波动率
]

# ── 价格形态类 (10) ──
PRICE_PATTERN_FEATURES = [
    "bb_pct_b",             # 布林带 %B
    "bb_width",             # 布林带宽度 (2*std/MA)
    "gap_pct",              # 跳空 % (open/prev_close - 1)
    "streak",               # 连续涨跌天数
    "price_position_20d",   # 20天高低百分位
    "price_position_60d",   # 60天高低百分位
    "higher_high_count",    # 过去10日新高天数
    "vwap_deviation",       # 收盘 vs VWAP 偏离
    "rsi_14",               # RSI(14)
    "ma_gap",               # (MA5 - MA20) / close 均线发散
]

# ── K线形态类 (9) ──
CANDLESTICK_FEATURES = [
    "candle_hammer",         # 锤子线
    "candle_shooting_star",  # 射击之星
    "candle_doji",           # 十字星
    "candle_engulfing_bull", # 看涨吞没
    "candle_engulfing_bear", # 看跌吞没
    "candle_evening_star",   # 黄昏星
    "candle_three_soldiers", # 三白兵
    "candle_three_crows",    # 三黑鸦
    "upper_shadow_ratio",    # 上影线比率
]

# ── 行业层级特征 (7) ──
INDUSTRY_FEATURES = [
    "excess_return_l1",      # 个股 vs L1 行业
    "excess_return_l2",      # 个股 vs L2 行业
    "excess_return_l3",      # 个股 vs L3 行业
    "sector_persistence",    # 行业持续性评分
    "L3_L2_divergence",      # L3 动量 - L2 动量
    "L2_L1_divergence",      # L2 动量 - L1 动量
    "industry_cascade",      # 个股+L3+L2+L1 方向一致数(0-4)
]

# ── 截面排名特征 (11) ──
CROSS_SECTIONAL_FEATURES = [
    "rank_mom20",
    "rank_vol_ratio",
    "rank_atr_pct",
    "rank_excess_return_l1",
    "rank_amount",
    "rank_price_position_20d",
    "rank_turnover_ratio",
    "rank_exp_wgt_return",
    "zscore_mom20",
    "zscore_excess_return",
    "zscore_vol_ratio",
]

# ── Beta / 市场相对 (6) ──
MARKET_RELATIVE_FEATURES = [
    "beta_20d",
    "beta_60d",
    "corr_market_20d",
    "idio_return_20d",
    "r_squared_market_20d",
    "relative_strength",
]

# ── 时间序列统计 (7) ──
STATISTICAL_FEATURES = [
    "autocorr_1d",
    "autocorr_2d",
    "autocorr_5d",
    "variance_ratio_5_1",
    "hurst_exponent",
    "runs_ratio",
    "gain_loss_consistency",
]

# ── 流动性微观 (5) ──
LIQUIDITY_FEATURES = [
    "amihud_illiq_20d",
    "price_impact",
    "corwin_schultz_spread",
    "zero_return_days_20d",
    "high_low_ratio_20d",
]

# ── 外部数据 (13+6+3+4 = 26) ── 在 join 阶段添加
EXTERNAL_FEATURES = [
    # 融资融券 (3)
    "margin_balance_change",
    "margin_buy_intensity",
    "margin_net_buy",
    # 资金流向 (4)
    "net_mf_ratio",
    "mf_buy_sell_ratio",
    "mf_consecutive",
    "mf_intensity_rank",
    # 基本面 (13)
    "pe_ttm", "pb", "roe", "earnings_yield",
    "pe_industry_percentile", "pb_industry_percentile",
    "gross_margin", "total_mv",
    "net_profit_margin", "cfo_margin",
    "fcf_margin", "capex_intensity",
    "cash_conversion",
]

# ── 复合因子 (5) ──
COMPOSITE_FEATURES = [
    "quality_score",
    "value_score",
    "momentum_quality",
    "composite_crowding",
    "board_type",
]

# ── V5 扩展特征 (182) — 全部可用现有数据计算 ──
V5_EXT_FEATURES = [
    # A2: 多尺度动量 (12)
    "mom2", "mom3", "mom4", "mom7", "mom12", "mom15",
    "mom30", "mom45", "mom75", "mom90", "mom120",
    "mom_harmonic_mean", "mom_frequency_ratio",
    # A3: 均线扩展 (11)
    "ma3_dev", "ma8_dev", "ma15_dev", "ma30_dev", "ma40_dev",
    "ma80_dev", "ma90_dev",
    "ma_short_arrangement", "ma_long_arrangement",
    "ma5_ma10_gap", "ma20_ma60_gap",
    # A4: MACD (6)
    "macd_dif", "macd_signal", "macd_hist",
    "macd_cross_long", "macd_zero_cross", "macd_histogram_accel",
    # A5/A6/A7: 成交量/波动率/BB扩展 (12)
    "amount_ma10_ratio", "vol_shock_count_5d",
    "volume_oscillator_accel", "volume_dry_up_flag",
    "realized_vol_5d", "realized_vol_10d",
    "vol_surprise", "vol_change_acceleration",
    "vol_regime_adaptive_ma",
    "bb_cross_upper", "bb_cross_lower", "bb_squeeze_flag",
    "inside_bb_days",
    # A9: RSI 扩展 (8)
    "rsi_3", "rsi_5", "rsi_7", "rsi_21", "rsi_60",
    "rsi_divergence_bull", "rsi_divergence_bear", "rsi_momentum",
    # A10: VWAP/日内 (8)
    "intraday_vwap_proxy", "close_vs_intraday_ratio",
    "open_vs_intraday_ratio", "high_vs_close_spread",
    "low_vs_close_spread", "open_position_ratio",
    "body_to_range_ratio", "open_position_accel",
    # A12: 流动性扩展 (2)
    "amihud_5d_mean", "volume_shock_indicator",
    # B: 行业涨停生态 (12)
    "lt_count_l1", "lt_rate_l1", "dt_count_l1", "lt_to_dt_ratio_l1",
    "lt_count_l2", "lt_rate_l2", "dt_count_l2", "lt_to_dt_ratio_l2",
    "up_pct_l1",
    "stock_vs_L1_alpha_daily",
    "rank_l1_in_all", "rank_l2_in_all",
    # C1: 融资融券扩展 (3)
    "margin_daily_change_pct", "leverage_ratio", "margin_accel",
    # C2: 资金流扩展 (4)
    "mf_bull_bear_divergence", "mf_direction_confidence",
    "net_mf_5d_sum", "net_mf_20d_sum",
    # D2: 财务质量 (8)
    "dso", "op_ratio_roe",
    "fcf_to_net_income", "revenue_cfo_gap",
    "fcf_yield", "asset_turnover_proxy",
    "capex_revenue_ratio", "working_capital_margin",
    # Layer 2: 北向/宏观 (5)
    "north_total", "north_sgt", "north_hgt",
    "shibor_1w", "shibor_1m",
    # V4 交叉特征（已有但未加入 ALL_FEATURES） (6)
    "momentum_volume", "price_volume_diverg",
    "bb_streak", "gap_momentum",
    "position_conviction", "vol_drawdown",
    # ── B9/B10: 市场广度 + 交易日历 (25) ──
    "advance_decline_ratio", "new_high_count_20d", "new_low_count_20d",
    "total_up_days", "market_regime_score", "market_extreme_signal",
    "breadth_momentum", "bull_bear_power", "market_fear_greed_proxy",
    "breadth_thrust_flag", "adtl_ratio_5d_ma", "breadth_ratio_trend",
    "market_mcginley_dynamic", "bulk_market_strength_index",
    "breadth_confirmation_score",
    "day_of_week_effect", "monthly_mean_reversion_signal",
    "annual_pattern_week_nr", "holiday_proximity_days",
    "quarter_earnings_window", "month_end_effect",
    "lunar_new_year_proximity", "golden_week_effect",
    "year_end_window_dressing", "tax_loss_selling_window",
    # ── B8: 行业扩展 (9) ──
    "industry_rank_price_move", "industry_vol_rank",
    "industry_turnover_rank", "industry_moneyflow_net_5d",
    "industry_moneyflow_share", "industry_price_divergence",
    "L3_strength_index", "industry_mom_acceleration",
    "industry_concentration_HHI",
    # ── A14: 非线性变换 (9) ──
    "rank_mom20_winsor", "rank_vol_ratio_winsor", "rank_atr_pct_winsor",
    "zscore_mom20_robust", "zscore_excess_return_log",
    "log1p_amount", "log1p_total_mv",
    "huber_delta_mom20", "huber_delta_vol_ratio",
    # ── A15: 交互特征 (30) ──
    "mom20_x_low_vol", "mom20_x_industry_rank",
    "vol_ratio_x_mom1", "mf_x_mom_convergence",
    "excess_return_x_industry_rank",
    "quality_x_mom20", "fcf_yield_x_mom120",
    "mom20_x_bb_width", "vol_ratio_x_atr_pct",
    "turnover_x_price_pos", "streak_x_vol_ratio",
    "rsi_x_mom20", "gap_x_mom5", "amount_x_pct_chg",
    "float_mv_x_turnover", "bb_pct_b_x_vol_ratio",
    "macd_hist_x_mom5", "leverage_x_mom20",
    "quality_x_value_residual",
    "industry_alpha_x_vol_surprise",
    "rsi_x_bb_width_direction", "macd_hist_x_vwap_dev",
    "atr_x_vol_ratio_diverg", "streak_x_mf_buy_sell",
    "turnover_x_mom20_diverg", "price_pos_x_amount_surge",
    "gap_x_rsi_divergence", "quality_x_momentum_consistency",
    "beta_x_idio_vol_ratio", "sector_momentum_x_lt_rate",
    # ── A16: 自适应阈值 (14) ──
    "adaptive_ma_short", "adaptive_ma_long",
    "vol_regime_flag", "trend_strength_score",
    "adaptive_atr_stop", "adaptive_bb_window",
    "adaptive_mom_window", "vol_regime_streak",
    "adaptive_vol_target_exposure", "dynamic_stop_loss_pct",
    "regime_adaptive_momentum_threshold",
    "adaptive_position_sizing_signal",
    # ── A13: 残差特征 (5) ──
    "market_residual_momentum", "idiosyncratic_vol_20d",
    "market_impact_coeff", "residual_skewness_20d",
    "alpha_persistence",
    # ── D4: 复合扩展 (8) ──
    "growth_score", "efficiency_score",
    "quality_value_residual", "composite_regime_factor",
    "quality_growth_accel", "value_momentum_divergence",
    "low_vol_anomaly_score",
    "momentum_crowding_divergence",
    # ── A6: 波动率扩展补齐 (4) ──
    "atr_ratio_14_7", "atr_regime_short_term",
    "atr_regression_slope_10d", "realized_vol_half_month",
    # ── A7: BB 扩展补齐 (4) ──
    "bb_convergence", "bb_expansion_rate",
    "bb_position_accel", "bb_shoot_through",
    # ── A10: VWAP 扩展补齐 (2) ──
    "intraday_momentum_30m", "vwap_ma20_divergence",
    # ── B: 行业涨停补全 (15) ──
    "lt_to_dt_ratio", "up_down_ratio_l1", "up_pct_in_l1",
    "lt_rate_trend_5d", "lt_max_chain_d", "lt_count_mom_accel",
    "lt_rate_mul_L1_rank", "L2_rank_in_L1",
    "board_max_chain_position", "is_20cm_eligible",
    "sector_momentum_3d_diff", "stock_rank_alpha_vs_L1L2",
    "sector_breadth_x_mom20", "pivot_high_5d", "pivot_low_5d",
    # ── C1: 融资融券扩展补齐 (5) ──
    "margin_5d_trend", "margin_buy_intensity_ratio",
    "margin_extreme_flag", "margin_price_divergence",
    "net_margin_position",
    # ── C2: 资金流扩展补齐 (4) ──
    "mf_buy_acceleration", "mf_sell_acceleration",
    "mf_turnover_effectiveness", "mf_buy_sell_rate_change",
    "mf_sustained_divergence_5d",
    # ── A5: 成交量扩展补齐 (6) ──
    "vol_ratio_rank_in_sector", "turnover_std_rank",
    "volume_distribution_skew_10d",
    "volume_price_trend_divergence", "volume_ratio_zscore_20d",
    "amount_growth_1d_rank",
    # ── D1/D2: 基本面扩展补齐 (6) ──
    "roa", "asset_liability_ratio", "current_ratio",
    "inventory_turnover", "gross_margin_trend_2q",
    "receivables_growth_rate",
    # ── A4: MACD 补全 (1) ──
    # ── 额外 A15/A16 (4) ──
    "mom3_mom20_ratio", "regime_adaptive_weight",
    "vol_regime_zscore_position", "illiquidity_ratio_rank_in_sector",
]

# ── 全量合并 (V4 + V5) ──
ALL_FEATURES_V4 = (
    MOMENTUM_FEATURES + MA_FEATURES + VOLUME_FEATURES +
    VOLATILITY_FEATURES + PRICE_PATTERN_FEATURES +
    CANDLESTICK_FEATURES + INDUSTRY_FEATURES +
    CROSS_SECTIONAL_FEATURES + MARKET_RELATIVE_FEATURES +
    STATISTICAL_FEATURES + LIQUIDITY_FEATURES +
    EXTERNAL_FEATURES + COMPOSITE_FEATURES
)
ALL_FEATURES = list(dict.fromkeys(list(ALL_FEATURES_V4) + V5_EXT_FEATURES))

# 分组配置
GROUP_CONFIG = {
    "micro": {
        "features": ALL_FEATURES,
        "params": dict(n_estimators=200, num_leaves=15, min_child_samples=20,
                       learning_rate=0.05, verbosity=-1, force_col_wise=True),
    },
    "mid": {
        "features": ALL_FEATURES,
        "params": dict(n_estimators=250, num_leaves=31, min_child_samples=30,
                       learning_rate=0.05, verbosity=-1, force_col_wise=True),
    },
}

_BEIJING_TZ = timezone(timedelta(hours=BEIJING_TZ_OFFSET))


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _safe_div(a: float, b: float, default=0.0) -> float:
    return a / b if abs(b) > 1e-10 else default


def compute_exp_wgt_return(closes: np.ndarray, vols: np.ndarray, lookback: int = 20) -> float:
    """换手率加权指数衰减动量。"""
    n = len(closes)
    if n < lookback + 1:
        return 0.0
    total_w = 0.0
    total_rw = 0.0
    for j in range(n - lookback, n):
        if closes[j - 1] <= 0:
            continue
        daily_ret = (closes[j] / closes[j - 1] - 1) * 100
        x = n - 1 - j
        decay = np.exp(-x / 4)
        w = decay * (vols[j] if vols[j] > 0 else 1)
        total_rw += w * daily_ret
        total_w += w
    return round(_safe_div(total_rw, total_w), 4)


def compute_turnover_std(vols: np.ndarray, lookback: int = 20) -> float:
    recent = vols[-lookback:]
    if recent.mean() <= 0:
        return 0.0
    return round(_safe_div(recent.std(), recent.mean()), 4)


def compute_turnover_ratio(vols: np.ndarray, short: int = 5, long_: int = 20) -> float:
    if len(vols) < long_ or vols[-long_:].mean() <= 0:
        return 1.0
    return round(vols[-short:].mean() / vols[-long_:].mean(), 4)


def compute_sector_persistence(stock_code: str, stock_mapping: dict,
                                l1_persist: dict, l2_persist: dict) -> float:
    ind_l2 = stock_mapping.get(stock_code, {}).get("l2_code", "") if stock_mapping else ""
    ind_l1 = stock_mapping.get(stock_code, {}).get("l1_code", "") if stock_mapping else ""
    return l2_persist.get(ind_l2, l1_persist.get(ind_l1, 5.0))


def compute_excess_return(stock_ret: float, industry_daily_df: pd.DataFrame | None,
                           stock_code: str, trade_date: str, stock_mapping: dict | None,
                           level: str = "l1", lookback: int = 20) -> float:
    """个股相对指定层级 (l1/l2/l3) 行业的超额收益。"""
    if industry_daily_df is None or stock_mapping is None:
        return 0.0
    level_key = f"{level}_code"
    ind_code = stock_mapping.get(stock_code, {}).get(level_key, "")
    if not ind_code:
        return 0.0
    grp = industry_daily_df[industry_daily_df[COL_TS_CODE] == ind_code].sort_values(COL_TRADE_DATE)
    grp = grp[grp[COL_TRADE_DATE] <= trade_date]
    if len(grp) < lookback + 1:
        return 0.0
    ind_ret = (grp[COL_CLOSE].iloc[-1] / grp[COL_CLOSE].iloc[-(lookback + 1)] - 1) * 100
    return round(stock_ret - ind_ret, 2)


def compute_industry_momentum(industry_daily_df: pd.DataFrame | None,
                               ind_code: str, trade_date: str, lookback: int = 20) -> float:
    """行业指数自身动量。"""
    if industry_daily_df is None or not ind_code:
        return 0.0
    grp = industry_daily_df[industry_daily_df[COL_TS_CODE] == ind_code].sort_values(COL_TRADE_DATE)
    grp = grp[grp[COL_TRADE_DATE] <= trade_date]
    if len(grp) < lookback + 1:
        return 0.0
    return (grp[COL_CLOSE].iloc[-1] / grp[COL_CLOSE].iloc[-(lookback + 1)] - 1) * 100


def compute_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float:
    """从 numpy array 计算 RSI(14)。"""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _compute_yang_zhang_vol(opens: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray, closes: np.ndarray) -> float:
    """Yang-Zhang 波动率估计（最有效 OHLC 波动率）。"""
    n = len(opens)
    if n < 2:
        return 0.0
    # Overnight volatility
    overnight = np.log(opens[1:] / closes[:-1])
    vo = np.var(overnight, ddof=1)
    # Open-to-close volatility
    open_close = np.log(closes / opens)
    vc = np.var(open_close, ddof=1)
    # Rogers-Satchell
    rs = np.log(highs / closes) * np.log(highs / opens) + \
         np.log(lows / closes) * np.log(lows / opens)
    vrs = np.mean(rs)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = np.sqrt(vo + k * vc + (1 - k) * vrs) * 100
    return float(yz)


def _compute_hurst(returns: np.ndarray, max_lag: int = 20) -> float:
    """Hurst 指数估计（R/S 分析法）。"""
    if len(returns) < max_lag * 2:
        return 0.5
    lags = range(2, min(max_lag, len(returns) // 2))
    tau = []
    for lag in lags:
        rs = 0.0
        count = 0
        for i in range(0, len(returns) - lag, lag):
            segment = returns[i:i + lag]
            mean_adj = segment - np.mean(segment)
            cum = np.cumsum(mean_adj)
            r = np.max(cum) - np.min(cum)
            s = np.std(segment, ddof=1)
            if s > 1e-10:
                rs += r / s
                count += 1
        if count > 0:
            tau.append(rs / count)
    if len(tau) < 3:
        return 0.5
    try:
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return float(np.clip(hurst, 0.01, 0.99))
    except Exception:
        return 0.5


def _compute_vwap_approx(closes: np.ndarray, amounts: np.ndarray) -> float:
    """VWAP 近似 = sum(close * amount) / sum(amount)。"""
    if np.sum(amounts) <= 0:
        return closes[-1]
    return float(np.sum(closes * amounts) / np.sum(amounts))


# ═══════════════════════════════════════════════════════════════
# 核心：特征矩阵构建（全量 OHLCV 衍生特征）
# ═══════════════════════════════════════════════════════════════

def build_feature_matrix(
    stock_daily_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame | None = None,
    stock_mapping: dict | None = None,
    persistence_scores: dict | None = None,
    multi_day: bool = False,
) -> pd.DataFrame:
    """从原始数据构建全量特征矩阵。

    覆盖：动量、均线、量/额、波动率、价格形态、K线、统计特征、流动性。
    截面排名/Beta/行业层级在后续阶段添加。

    Args:
        stock_daily_df: 个股日线（需含 open/high/low/close/vol/amount）
        industry_daily_df: L1 行业指数日线
        stock_mapping: 行业映射 {ts_code: {l1_code, l2_code, l3_code}}
        persistence_scores: 行业持续性评分
        multi_day: 是否为多日推理模式
    """
    if stock_daily_df is None or stock_daily_df.empty:
        return pd.DataFrame()

    l1_persist = (persistence_scores or {}).get("l1", {})
    l2_persist = (persistence_scores or {}).get("l2", {})

    df = stock_daily_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    codes = df[COL_TS_CODE].unique()

    features = []
    for code in codes:
        sdf = df[df[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
        if len(sdf) < 41:
            continue

        opens_ = sdf[COL_OPEN].values if COL_OPEN in sdf.columns else sdf[COL_CLOSE].values
        closes = sdf[COL_CLOSE].values
        highs = sdf[COL_HIGH].values if COL_HIGH in sdf.columns else closes
        lows = sdf[COL_LOW].values if COL_LOW in sdf.columns else closes
        vols = sdf[COL_VOL].values if COL_VOL in sdf.columns else np.ones(len(closes))
        amounts = sdf[COL_AMOUNT].values if COL_AMOUNT in sdf.columns else vols
        dates_arr = sdf[COL_TRADE_DATE].values
        n = len(closes)

        # 预计算日收益（全序列）
        daily_rets = np.zeros(n)
        for t in range(1, n):
            daily_rets[t] = _safe_div(closes[t], closes[t-1]) * 100 - 100 if closes[t-1] > 0 else 0

        # ── 全序列统计量（高效计算一次）──
        autocorr_1d = 0.0
        autocorr_2d = 0.0
        autocorr_5d = 0.0
        vratio_5_1 = 1.0
        hurst_exp = 0.5
        runs_r = 1.0
        gl_consistency = 1.0
        if len(daily_rets) > 30:
            r = daily_rets[1:] - np.mean(daily_rets[1:])
            # 自相关
            if np.var(r) > 1e-10:
                autocorr_1d = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else 0
                autocorr_2d = float(np.corrcoef(r[:-2], r[2:])[0, 1]) if len(r) > 3 else 0
                autocorr_5d = float(np.corrcoef(r[:-5], r[5:])[0, 1]) if len(r) > 6 else 0
            # Variance Ratio
            if len(daily_rets) > 10:
                var_1 = np.var(daily_rets[1:], ddof=1)
                ret_5d = np.array([np.sum(daily_rets[max(0, t-4):t+1]) for t in range(5, len(daily_rets))])
                var_5 = np.var(ret_5d, ddof=1) if len(ret_5d) > 1 else 0
                vratio_5_1 = _safe_div(var_5, 5 * var_1) if var_1 > 0 else 1.0
            # Hurst
            hurst_exp = _compute_hurst(daily_rets[1:])
            # Runs ratio
            if len(daily_rets) > 20:
                med = np.median(daily_rets[1:])
                above = daily_rets[1:] > med
                runs = 1 + np.sum(above[1:] != above[:-1])
                expected = 1 + 2 * np.sum(above) * np.sum(~above) / len(above)
                runs_r = _safe_div(runs, expected)
            # Gain/loss consistency
            gains = daily_rets[1:][daily_rets[1:] > 0]
            losses = daily_rets[1:][daily_rets[1:] < 0]
            gl_consistency = _safe_div(np.std(gains) if len(gains) > 0 else 0,
                                        np.std(losses) if len(losses) > 0 else 1)

        for i in range(20, n):
            date_str = dates_arr[i] if isinstance(dates_arr[i], str) else str(dates_arr[i])

            # ── 基础引用 ──
            close_i = closes[i]
            open_i = opens_[i]
            high_i = highs[i]
            low_i = lows[i]
            vol_i = vols[i]
            amt_i = amounts[i]
            prev_close = closes[i-1] if i > 0 else close_i

            # ═════════════════════════════════════════════════════
            # 1. 动量特征
            # ═════════════════════════════════════════════════════

            # mom1
            mom1 = _safe_div(close_i, prev_close) * 100 - 100 if prev_close > 0 else 0

            # mom5 / mom10 / mom20
            mom5  = _safe_div(close_i, closes[i-5]) * 100 - 100 if closes[i-5] > 0 else 0
            mom10 = _safe_div(close_i, closes[i-10]) * 100 - 100 if closes[i-10] > 0 else 0
            mom20 = _safe_div(close_i, closes[i-20]) * 100 - 100 if closes[i-20] > 0 else 0

            # mom60
            mom60 = 0.0
            if i >= 60 and closes[i-60] > 0:
                mom60 = _safe_div(close_i, closes[i-60]) * 100 - 100

            # mom_ratio
            mom_ratio = _safe_div(mom5, mom20) if abs(mom20) > 0.01 else 0

            # 换手率加权指数衰减动量
            exp_wgt_ret = compute_exp_wgt_return(closes[max(0, i-20):i+1],
                                                  vols[max(0, i-20):i+1])

            # mom_accel
            mom_accel = mom5 - mom20

            # upside_downside_ratio
            ret_window = daily_rets[i-19:i+1]
            up_mean = np.mean(ret_window[ret_window > 0]) if np.sum(ret_window > 0) > 0 else 0
            down_mean = abs(np.mean(ret_window[ret_window < 0])) if np.sum(ret_window < 0) > 0 else 1
            ud_ratio = _safe_div(up_mean, down_mean)

            # momentum_consistency: 最近10日方向一致天数
            consistency = 0
            ref_sign = mom5 >= 0
            for k in range(1, 11):
                if i >= k:
                    ret_k = _safe_div(closes[i-k+1], closes[i-k]) * 100 - 100 if closes[i-k] > 0 else 0
                    if (ret_k >= 0) == ref_sign:
                        consistency += 1
                    else:
                        break

            # vol_adjusted_mom20
            atr_for_adjust = 0.0  # will be computed below
            vol_adjusted_mom = 0.0

            # ═════════════════════════════════════════════════════
            # 2. 均线特征
            # ═════════════════════════════════════════════════════

            # MA5 / MA10 / MA20 / MA60 / MA120 / MA250
            ma5  = np.mean(closes[i-4:i+1]) if i >= 4 else close_i
            ma10 = np.mean(closes[i-9:i+1]) if i >= 9 else close_i
            ma20 = np.mean(closes[i-19:i+1])
            ma60 = np.mean(closes[i-59:i+1]) if i >= 59 else close_i
            ma120 = np.mean(closes[i-119:i+1]) if i >= 119 else close_i
            ma250 = np.mean(closes[i-249:i+1]) if i >= 249 else close_i

            ma5_dev  = _safe_div(close_i, ma5) * 100 - 100
            ma10_dev = _safe_div(close_i, ma10) * 100 - 100
            ma20_dev = _safe_div(close_i, ma20) * 100 - 100
            ma60_dev = _safe_div(close_i, ma60) * 100 - 100 if i >= 59 else 0
            ma120_dev = _safe_div(close_i, ma120) * 100 - 100 if i >= 119 else 0
            ma250_dev = _safe_div(close_i, ma250) * 100 - 100 if i >= 249 else 0

            # 均线排列评分: MA5>MA10>MA20>MA60? 多头排列
            arr_score = 0
            arr_score += 1 if ma5 > ma10 else -1 if ma5 < ma10 else 0
            arr_score += 1 if ma10 > ma20 else -1 if ma10 < ma20 else 0
            arr_score += 1 if ma20 > ma60 else -1 if ma20 < ma60 else 0
            # Normalize to 0-4 (shift by +3, 但多头排列时 arr_score >= 2, 全空头 <= -2)
            ma_arr = max(0, min(4, arr_score + 2))

            # 金叉信号
            ma_cross_flag = 1.0 if ma5 > ma20 else 0.0

            # 均线发散度 (MA5 - MA20) / close
            ma_gap_val = _safe_div(ma5 - ma20, close_i) * 100

            # ═════════════════════════════════════════════════════
            # 3. 成交量/额特征
            # ═════════════════════════════════════════════════════

            ma20_vol = np.mean(vols[i-19:i+1]) if i >= 19 else 1
            ma5_vol = np.mean(vols[i-4:i+1]) if i >= 4 else 1

            vol_ratio_val = _safe_div(vol_i, ma20_vol)
            vol_ma5_ratio_val = _safe_div(vol_i, ma5_vol)
            vol_shock = _safe_div(vol_i, vols[i-1]) if i > 0 else 1
            vol_osc = _safe_div(ma5_vol - ma20_vol, ma20_vol)

            # 成交额特征
            ma20_amt = np.mean(amounts[i-19:i+1]) if i >= 19 else 1
            ma5_amt = np.mean(amounts[i-4:i+1]) if i >= 4 else 1
            amount_ratio_val = _safe_div(amt_i, ma20_amt)
            amount_trend_val = _safe_div(ma5_amt, ma20_amt)

            # 换手率特征
            turn_std = compute_turnover_std(vols)
            turn_ratio = compute_turnover_ratio(vols)

            # ═════════════════════════════════════════════════════
            # 4. 波动率特征
            # ═════════════════════════════════════════════════════

            # ATR(14)
            tr_vals = [max(highs[j] - lows[j],
                           abs(highs[j] - closes[j-1]),
                           abs(lows[j] - closes[j-1]))
                       for j in range(max(0, i-13), i+1)]
            atr = np.mean(tr_vals)
            atr_pct_val = _safe_div(atr, close_i) * 100

            # 填充 vol_adjusted_mom（需要 atr_pct）
            vol_adjusted_mom = _safe_div(mom20, atr_pct_val) if atr_pct_val > 0.01 else 0

            # 日内位置
            candle_range = high_i - low_i
            close_pos = _safe_div(close_i - low_i, candle_range) if candle_range > 0 else 0.5

            # 日内振幅
            daily_range_val = _safe_div(candle_range, close_i) * 100

            # Yang-Zhang Vol
            yz_start = max(0, i - 20)
            yz_vol = _compute_yang_zhang_vol(
                opens_[yz_start:i+1], highs[yz_start:i+1],
                lows[yz_start:i+1], closes[yz_start:i+1]
            )

            # 20日夏普比
            ret_win = daily_rets[i-19:i+1]
            sharpe_20d = _safe_div(np.mean(ret_win), np.std(ret_win, ddof=1)) * np.sqrt(252) \
                if np.std(ret_win, ddof=1) > 0 else 0

            # 20日最大回撤
            peak = np.maximum.accumulate(closes[i-19:i+1])
            dd = (closes[i-19:i+1] / peak - 1)
            max_dd = float(np.min(dd)) * 100 if len(dd) > 0 else 0

            # 偏度/峰度
            skew_20d = float(pd.Series(ret_win).skew()) if len(ret_win) > 3 else 0
            kurt_20d = float(pd.Series(ret_win).kurt()) if len(ret_win) > 3 else 0

            # 下行波动率
            neg_rets = ret_win[ret_win < 0]
            downside_vol_val = float(np.std(neg_rets, ddof=1)) if len(neg_rets) > 1 else 0

            # ═════════════════════════════════════════════════════
            # 5. 价格形态
            # ═════════════════════════════════════════════════════

            # 跳空
            gap_pct_val = _safe_div(open_i, prev_close) * 100 - 100

            # 连续涨跌
            streak_val = 0
            for j in range(i, max(0, i-10), -1):
                if closes[j] > closes[j-1]:
                    streak_val = streak_val + 1 if streak_val >= 0 else 1
                elif closes[j] < closes[j-1]:
                    streak_val = streak_val - 1 if streak_val <= 0 else -1
                else:
                    break

            # 布林带
            std20 = np.std(closes[i-19:i+1])
            bb_pct_b_val = _safe_div(close_i - (ma20 - 2 * std20), 4 * std20) * 100 if std20 > 0 else 50
            bb_width_val = _safe_div(2 * std20, ma20) * 100 if ma20 > 0 else 0

            # 价格位置
            lo_20 = np.min(closes[i-19:i+1])
            hi_20 = np.max(closes[i-19:i+1])
            price_pos_20 = _safe_div(close_i - lo_20, hi_20 - lo_20) * 100 if hi_20 > lo_20 else 50

            lo_60 = np.min(closes[max(0, i-59):i+1])
            hi_60 = np.max(closes[max(0, i-59):i+1])
            price_pos_60 = _safe_div(close_i - lo_60, hi_60 - lo_60) * 100 if hi_60 > lo_60 else 50

            # 新高计数
            hh_count = 0
            for k in range(1, 11):
                if i >= k and closes[i-k+1] == np.max(closes[i-k:i+1]):
                    hh_count += 1

            # VWAP 偏离
            vwap = _compute_vwap_approx(closes[max(0, i-20):i+1], amounts[max(0, i-20):i+1])
            vwap_dev = _safe_div(close_i - vwap, vwap) * 100 if vwap > 0 else 0

            # RSI
            rsi_val = compute_rsi(closes[max(0, i-14):i+1])

            # ═════════════════════════════════════════════════════
            # 6. K线形态
            # ═════════════════════════════════════════════════════

            body = abs(close_i - open_i)
            total_range = high_i - low_i

            # 锤子线: 下影线≥实体2倍, 上影线短, 实体在下端
            upper_shadow = high_i - max(close_i, open_i)
            lower_shadow = min(close_i, open_i) - low_i
            hammer = 1.0 if (total_range > 0 and body > 0 and
                             lower_shadow >= 2 * body and upper_shadow <= body * 0.3) else 0.0

            # 射击之星: 上影线≥实体2倍
            shooting_star = 1.0 if (total_range > 0 and body > 0 and
                                     upper_shadow >= 2 * body and lower_shadow <= body * 0.3) else 0.0

            # 十字星: 实体极小
            doji = 1.0 if (total_range > 0 and _safe_div(body, total_range) < 0.05) else 0.0

            # 吞没形态
            engulfing_bull = 0.0
            engulfing_bear = 0.0
            if i >= 1:
                prev_body = abs(closes[i-1] - opens_[i-1])
                if body > 0 and prev_body > 0:
                    # 看涨吞没: 前阴后阳, 阳实体吞没前阴
                    if closes[i-1] < opens_[i-1] and close_i > open_i:
                        if close_i >= opens_[i-1] and open_i <= closes[i-1]:
                            engulfing_bull = 1.0
                    # 看跌吞没: 前阳后阴, 阴实体吞没前阳
                    if closes[i-1] > opens_[i-1] and close_i < open_i:
                        if open_i >= closes[i-1] and close_i <= opens_[i-1]:
                            engulfing_bear = 1.0

            # 黄昏星 (简化版): 大阳→小实体→大阴
            evening_star = 0.0
            if i >= 2:
                body1 = abs(closes[i-2] - opens_[i-2])
                body2 = abs(closes[i-1] - opens_[i-1])
                if body1 > 0 and body2 > 0 and body > 0:
                    is_big_up = (closes[i-2] > opens_[i-2]) and body1 > np.mean(closes[max(0,i-20):i+1]) * 0.02
                    is_small = body2 < body1 * 0.3
                    is_big_down = (close_i < open_i) and (close_i < closes[i-2] - body1 * 0.5)
                    if is_big_up and is_small and is_big_down:
                        evening_star = 1.0

            # 三白兵: 连续3根大阳
            three_soldiers = 0.0
            if i >= 2:
                c1, c2, c3 = closes[i-2], closes[i-1], close_i
                o1, o2, o3 = opens_[i-2], opens_[i-1], open_i
                if (c1 > o1 and c2 > o2 and c3 > o3 and
                    c1 > closes[i-3] if i >= 3 else True and
                    c2 > c1 and c3 > c2):
                    three_soldiers = 1.0

            # 三黑鸦: 连续3根大阴
            three_crows = 0.0
            if i >= 2:
                if (c1 < o1 and c2 < o2 and c3 < o3 and
                    c1 < closes[i-3] if i >= 3 else True and
                    c2 < c1 and c3 < c2):
                    three_crows = 1.0

            # 上影线比率
            upper_shadow_ratio_val = _safe_div(upper_shadow, total_range) if total_range > 0 else 0

            # ═════════════════════════════════════════════════════
            # 7. 时间序列统计（全序列值，每个日期重复）
            # ═════════════════════════════════════════════════════

            # ═════════════════════════════════════════════════════
            # 8. 流动性微观结构
            # ═════════════════════════════════════════════════════

            # Amihud 非流动性: |return| / amount * 1e6
            abs_ret = abs(daily_rets[i]) / 100  # 转换为小数
            amihud = _safe_div(abs_ret, amt_i) * 1e6 if amt_i > 0 else 0

            # Price impact: 单位成交额的价格冲击
            price_impact_val = _safe_div(abs_ret, amt_i) if amt_i > 0 else 0

            # Corwin-Schultz 买卖价差估计 (从H/L)
            if i >= 1:
                beta_hl = np.log(highs[i] / lows[i]) ** 2 + np.log(highs[i-1] / lows[i-1]) ** 2
                gamma_hl = np.log(max(highs[i], highs[i-1]) / min(lows[i], lows[i-1])) ** 2
                cs_spread = _safe_div(2 * (np.exp(beta_hl) - np.exp(gamma_hl)),
                                      2 + np.exp(beta_hl) - 2 * np.exp(gamma_hl))
                cs_spread = max(0, cs_spread) * 100
            else:
                cs_spread = 0.0

            # 零收益天数
            zero_returns = np.sum(np.abs(daily_rets[i-19:i+1]) < 0.01)

            # 高低比均值
            hl_ratio_20d = float(np.mean(np.log(highs[i-19:i+1] / lows[i-19:i+1])))

            # ═════════════════════════════════════════════════════
            # 9. 行业相对特征（需要行业数据）
            # ═════════════════════════════════════════════════════

            # excess_return_l1（沿用原逻辑）
            excess_l1 = compute_excess_return(mom20, industry_daily_df, code, date_str, stock_mapping, "l1")

            # sector persistence
            sp = compute_sector_persistence(code, stock_mapping, l1_persist, l2_persist)

            # ═════════════════════════════════════════════════════
            # 10. 交叉特征
            # ═════════════════════════════════════════════════════

            # vol_conviction (保留)
            vol_conviction_val = vol_ratio_val * ma20_dev

            # momentum_volume
            mom_vol = mom5 * vol_ratio_val

            # price_volume_diverg: 价格上涨但缩量
            pv_diverg = 1.0 if streak_val >= 2 and vol_ratio_val < 0.8 else 0.0

            # bb_streak
            bb_streak_val = bb_pct_b_val * streak_val

            # gap_momentum
            gap_mom = gap_pct_val * mom5

            # position_conviction
            pos_conviction = price_pos_20 * vol_ratio_val / 100

            # vol_drawdown
            vol_dd = vol_ratio_val * max_dd

            # ═════════════════════════════════════════════════════
            # 写入 feature dict
            # ═════════════════════════════════════════════════════

            feature_row = {
                COL_TS_CODE: code,
                COL_TRADE_DATE: date_str,

                # 动量
                "mom1": round(mom1, 4),
                "mom5": round(mom5, 4),
                "mom10": round(mom10, 4),
                "mom20": round(mom20, 4),
                "mom60": round(mom60, 4),
                "mom_ratio_5_20": round(mom_ratio, 4),
                "vol_adjusted_mom20": round(vol_adjusted_mom, 4),
                "exp_wgt_return_20d": round(exp_wgt_ret, 4),
                "mom_accel": round(mom_accel, 4),
                "upside_downside_ratio": round(ud_ratio, 4),
                "momentum_consistency": float(consistency),

                # 均线
                "ma5_dev": round(ma5_dev, 4),
                "ma10_dev": round(ma10_dev, 4),
                "ma20_dev": round(ma20_dev, 4),
                "ma60_dev": round(ma60_dev, 4),
                "ma120_dev": round(ma120_dev, 4),
                "ma250_dev": round(ma250_dev, 4),
                "ma_cross": ma_cross_flag,
                "ma_arrangement": float(ma_arr),

                # 量/额
                "vol_ratio": round(vol_ratio_val, 4),
                "vol_ma5_ratio": round(vol_ma5_ratio_val, 4),
                "vol_shock": round(vol_shock, 4),
                "volume_oscillator": round(vol_osc, 4),
                "amount_ratio": round(amount_ratio_val, 4),
                "amount_trend": round(amount_trend_val, 4),
                "turnover_ratio_5d_20d": round(turn_ratio, 4),
                "turnover_std_20d": round(turn_std, 4),
                "vol_amplitude_ratio": _safe_div(vol_ratio_val, atr_pct_val) if atr_pct_val > 0.01 else 0,
                "vol_conviction": round(vol_conviction_val, 4),

                # 波动率
                "atr_pct": round(atr_pct_val, 4),
                "close_position": round(close_pos, 4),
                "daily_range": round(daily_range_val, 4),
                "yang_zhang_vol": round(yz_vol, 4),
                "sharpe_20d": round(sharpe_20d, 4),
                "max_drawdown_20d": round(max_dd, 4),
                "skewness_20d": round(skew_20d, 4),
                "kurtosis_20d": round(kurt_20d, 4),
                "downside_vol": round(downside_vol_val, 4),

                # 价格形态
                "bb_pct_b": round(bb_pct_b_val, 4),
                "bb_width": round(bb_width_val, 4),
                "gap_pct": round(gap_pct_val, 4),
                "streak": float(streak_val),
                "price_position_20d": round(price_pos_20, 4),
                "price_position_60d": round(price_pos_60, 4),
                "higher_high_count": float(hh_count),
                "vwap_deviation": round(vwap_dev, 4),
                "rsi_14": round(rsi_val, 4),
                "ma_gap": round(ma_gap_val, 4),

                # K线形态
                "candle_hammer": hammer,
                "candle_shooting_star": shooting_star,
                "candle_doji": doji,
                "candle_engulfing_bull": engulfing_bull,
                "candle_engulfing_bear": engulfing_bear,
                "candle_evening_star": evening_star,
                "candle_three_soldiers": three_soldiers,
                "candle_three_crows": three_crows,
                "upper_shadow_ratio": round(upper_shadow_ratio_val, 4),

                # 行业
                "excess_return_l1": round(excess_l1, 4),
                "excess_return_l2": 0.0,   # 后续 fill
                "excess_return_l3": 0.0,   # 后续 fill
                "sector_persistence": round(sp, 4),
                "L3_L2_divergence": 0.0,
                "L2_L1_divergence": 0.0,
                "industry_cascade": 0.0,

                # 统计
                "autocorr_1d": round(autocorr_1d, 4),
                "autocorr_2d": round(autocorr_2d, 4),
                "autocorr_5d": round(autocorr_5d, 4),
                "variance_ratio_5_1": round(vratio_5_1, 4),
                "hurst_exponent": round(hurst_exp, 4),
                "runs_ratio": round(runs_r, 4),
                "gain_loss_consistency": round(gl_consistency, 4),

                # 流动性
                "amihud_illiq_20d": round(amihud, 6),
                "price_impact": round(price_impact_val, 8),
                "corwin_schultz_spread": round(cs_spread, 6),
                "zero_return_days_20d": float(zero_returns),
                "high_low_ratio_20d": round(hl_ratio_20d, 4),

                # 交叉特征（额外保存）
                "momentum_volume": round(mom_vol, 4),
                "price_volume_diverg": pv_diverg,
                "bb_streak": round(bb_streak_val, 4),
                "gap_momentum": round(gap_mom, 4),
                "position_conviction": round(pos_conviction, 4),
                "vol_drawdown": round(vol_dd, 4),

                "close": close_i,
            }

            # 截面排名和 Beta 占位（后续填补）
            for col in CROSS_SECTIONAL_FEATURES + MARKET_RELATIVE_FEATURES + EXTERNAL_FEATURES + COMPOSITE_FEATURES:
                if col not in feature_row:
                    feature_row[col] = 0.0

            features.append(feature_row)

    result_df = pd.DataFrame(features)

    # 确保类型正确
    for c in result_df.columns:
        if c not in (COL_TS_CODE, COL_TRADE_DATE):
            try:
                result_df[c] = pd.to_numeric(result_df[c], errors="coerce")
            except Exception:
                pass

    return result_df


# ═══════════════════════════════════════════════════════════════
# 第二阶段：行业层级特征填充
# ═══════════════════════════════════════════════════════════════

def add_industry_hierarchy_features(
    feature_df: pd.DataFrame,
    stock_mapping: dict,
    l1_daily: pd.DataFrame | None,
    l2_daily: pd.DataFrame | None,
    l3_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    """填充 L1/L2/L3 行业层级特征。

    包括：L3_excess_return, L2_excess_return, L3_L2_divergence,
          L2_L1_divergence, industry_cascade。
    """
    if feature_df is None or feature_df.empty:
        return feature_df

    df = feature_df.copy()

    # 预计算各层级的动量
    def _ind_momentum(ind_df: pd.DataFrame | None) -> dict:
        """返回 {(code, date): momentum} 字典。"""
        if ind_df is None or ind_df.empty:
            return {}
        result = {}
        for ind_code in ind_df[COL_TS_CODE].unique():
            grp = ind_df[ind_df[COL_TS_CODE] == ind_code].sort_values(COL_TRADE_DATE)
            closes = grp[COL_CLOSE].values
            dates = grp[COL_TRADE_DATE].values
            for j in range(20, len(closes)):
                mom = _safe_div(closes[j], closes[j-20]) * 100 - 100 if closes[j-20] > 0 else 0
                result[(ind_code, dates[j])] = mom
        return result

    l1_mom = _ind_momentum(l1_daily)
    l2_mom = _ind_momentum(l2_daily)
    l3_mom = _ind_momentum(l3_daily)

    for idx, row in df.iterrows():
        code = row[COL_TS_CODE]
        date = row[COL_TRADE_DATE]
        mapping = stock_mapping.get(code, {})
        l1_code = mapping.get("l1_code", "")
        l2_code = mapping.get("l2_code", "")
        l3_code = mapping.get("l3_code", "")

        # 未分类股票(l1_code为空): 用沪深300市场指数代理行业 (全量池扩池 2026-08-05)
        eff_l1 = l1_code or MARKET_INDEX_CODE
        eff_l2 = l2_code or MARKET_INDEX_CODE
        eff_l3 = l3_code or MARKET_INDEX_CODE

        l1_m = l1_mom.get((eff_l1, date), 0.0) if eff_l1 else 0.0
        l2_m = l2_mom.get((eff_l2, date), 0.0) if eff_l2 else 0.0
        l3_m = l3_mom.get((eff_l3, date), 0.0) if eff_l3 else 0.0

        stock_mom = row.get("mom20", 0.0)

        # L3 / L2 超额收益（未分类=相对市场指数）
        df.at[idx, "excess_return_l3"] = round(stock_mom - l3_m, 4)
        df.at[idx, "excess_return_l2"] = round(stock_mom - l2_m, 4)

        # 层级发散
        df.at[idx, "L3_L2_divergence"] = round(l3_m - l2_m, 4)
        df.at[idx, "L2_L1_divergence"] = round(l2_m - l1_m, 4)

        # 层级一致性: 个股/L3/L2/L1 方向一致数量
        signs = [np.sign(stock_mom), np.sign(l3_m), np.sign(l2_m), np.sign(l1_m)]
        cascade = sum(1 for s in signs if s != 0 and s == signs[0])
        df.at[idx, "industry_cascade"] = float(cascade) if signs[0] != 0 else 2.0

    return df


# ═══════════════════════════════════════════════════════════════
# 第三阶段：截面排名 + Z-Score
# ═══════════════════════════════════════════════════════════════

def add_cross_sectional_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """按每个交易日，对所有股票的关键特征计算截面排名和 Z-Score。"""
    if feature_df is None or feature_df.empty:
        return feature_df

    df = feature_df.copy()

    rank_config = {
        "rank_mom20": "mom20",
        "rank_vol_ratio": "vol_ratio",
        "rank_atr_pct": "atr_pct",
        "rank_excess_return_l1": "excess_return_l1",
        "rank_amount": "close",  # will use amount from raw but close as proxy
        "rank_price_position_20d": "price_position_20d",
        "rank_turnover_ratio": "turnover_ratio_5d_20d",
        "rank_exp_wgt_return": "exp_wgt_return_20d",
        "zscore_mom20": "mom20",
        "zscore_excess_return": "excess_return_l1",
        "zscore_vol_ratio": "vol_ratio",
    }

    # 按日期分组，对每组计算排名
    for date, group in df.groupby(COL_TRADE_DATE):
        idxs = group.index
        for rank_col, src_col in rank_config.items():
            if src_col not in df.columns:
                continue
            vals = df.loc[idxs, src_col].values
            if np.all(vals == vals[0]):  # 全部相同
                continue
            if rank_col.startswith("rank_"):
                # 百分位排名 (1-100)
                ranks = rankdata(vals, method="average")
                pct_ranks = (ranks / len(ranks)) * 100
                df.loc[idxs, rank_col] = pct_ranks
            elif rank_col.startswith("zscore_"):
                # Z-Score
                mu = np.mean(vals)
                sigma = np.std(vals, ddof=1)
                if sigma > 1e-10:
                    df.loc[idxs, rank_col] = (vals - mu) / sigma
                else:
                    df.loc[idxs, rank_col] = 0.0

    return df


# ═══════════════════════════════════════════════════════════════
# 第四阶段：Beta / 市场相对特征
# ═══════════════════════════════════════════════════════════════

def add_market_relative_features(
    feature_df: pd.DataFrame,
    stock_daily_df: pd.DataFrame,
    market_index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """计算个股 vs 大盘的 Beta、相关系数、特质收益（向量化版本）。

    大盘默认使用沪深300 (000300.SH)，回退到个股等权合成。
    全量向量化：用 numpy stride_tricks 替代 6.5M 次 Python 循环。
    """
    if feature_df is None or feature_df.empty:
        return feature_df

    df = feature_df.copy()

    # ── 获取市场日收益 ──
    market_ret_series = None
    if market_index_df is not None and not market_index_df.empty:
        if "market_return" in market_index_df.columns:
            market_ret_series = market_index_df.set_index(COL_TRADE_DATE)["market_return"]
        elif COL_CLOSE in market_index_df.columns:
            mkt = market_index_df.sort_values(COL_TRADE_DATE)
            rets = [0.0] + [mkt[COL_CLOSE].iloc[j] / mkt[COL_CLOSE].iloc[j-1] - 1
                             for j in range(1, len(mkt))]
            market_ret_series = pd.Series(rets, index=mkt[COL_TRADE_DATE].values)

    if market_ret_series is None and stock_daily_df is not None:
        # 等权合成市场收益
        ret_df = stock_daily_df.pivot_table(index=COL_TRADE_DATE, columns=COL_TS_CODE, values="pct_chg")
        market_ret_series = ret_df.mean(axis=1) / 100

    if market_ret_series is None or market_ret_series.empty:
        logger.warning("无市场数据，跳过 Beta 特征")
        return df

    # ── 将个股日收益和大盘日收益 merge 到特征矩阵 ──
    daily_ret = stock_daily_df.set_index([COL_TS_CODE, COL_TRADE_DATE])["pct_chg"] / 100
    df_index = df.set_index([COL_TS_CODE, COL_TRADE_DATE])
    df["stock_ret"] = df_index.index.map(daily_ret.to_dict()).fillna(0).values if hasattr(df_index.index, 'map') else 0
    # 更可靠的合并方式
    ret_dict = daily_ret.to_dict()
    df["stock_ret"] = df.apply(
        lambda r: ret_dict.get((r[COL_TS_CODE], r[COL_TRADE_DATE]), 0.0), axis=1
    )
    df["mkt_ret"] = df[COL_TRADE_DATE].map(market_ret_series.to_dict()).fillna(0)

    # ── 向量化逐股 Rolling Beta（numpy stride_tricks）──
    def _roll_beta_vec(g):
        n = len(g)
        if n < 21:
            cols = ["beta_20d","corr_market_20d","r_squared_market_20d",
                    "idio_return_20d","relative_strength"]
            z = pd.DataFrame(0.0, index=g.index, columns=cols)
            # 60d Beta
            cols60 = ["beta_60d"]
            z60 = pd.DataFrame(0.0, index=g.index, columns=cols60)
            return pd.concat([z, z60], axis=1)

        sr = g["stock_ret"].values.astype(np.float64)
        mr = g["mkt_ret"].values.astype(np.float64)

        def _window_beta(s, m, w=20, label="20d"):
            """滚动 w 天 Beta 及相关特征"""
            n_, w_ = len(s), w
            out = np.zeros((n_, 1 if label == "60d" else 5))
            if n_ < w_ + 1:
                return out

            # 用 stride_tricks 创建不拷贝数据的滑动窗口视图
            shape = (n_ - w_ + 1, w_)
            strides_s = (s.strides[0], s.strides[0])
            strides_m = (m.strides[0], m.strides[0])
            sw = np.lib.stride_tricks.as_strided(s, shape=shape, strides=strides_s)
            mw = np.lib.stride_tricks.as_strided(m, shape=shape, strides=strides_m)

            sm = np.mean(sw, axis=1)
            mm = np.mean(mw, axis=1)
            sd = sw - sm[:, None]
            md = mw - mm[:, None]

            cov = np.sum(sd * md, axis=1) / (w_ - 1)
            var_m = np.sum(md ** 2, axis=1) / (w_ - 1)
            std_s = np.sqrt(np.sum(sd ** 2, axis=1) / (w_ - 1))
            std_m = np.sqrt(var_m)

            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where((std_s > 0) & (std_m > 0), cov / (std_s * std_m), 0)
                beta = np.where(std_m > 0, corr * std_s / std_m, 0)

            if label == "60d":
                out[w_ - 1:, 0] = beta
                return out

            r2 = corr ** 2
            idio = sw[:, -1] - beta * mw[:, -1]
            rel_str = sw[:, -1] - mw[:, -1]
            out[w_ - 1:, :] = np.column_stack([beta, corr, r2, idio * 100, rel_str * 100])
            return out

        cols_20 = ["beta_20d","corr_market_20d","r_squared_market_20d",
                   "idio_return_20d","relative_strength"]
        cols_60 = ["beta_60d"]

        w20 = _window_beta(sr, mr, 20, "20d")
        w60 = _window_beta(sr, mr, 60, "60d")
        r20 = pd.DataFrame(w20, index=g.index, columns=cols_20)
        r60 = pd.DataFrame(w60, index=g.index, columns=cols_60)
        return pd.concat([r20, r60], axis=1)

    logger.info("   滚动 Beta（向量化逐股 %d 次）...", df[COL_TS_CODE].nunique())
    beta_df = df.groupby(COL_TS_CODE, group_keys=False).apply(_roll_beta_vec)
    for c in beta_df.columns:
        df[c] = beta_df[c].fillna(0)

    df.drop(columns=["stock_ret", "mkt_ret"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════
# 第五阶段：外部数据 Join
# ═══════════════════════════════════════════════════════════════

def add_external_data_features(
    feature_df: pd.DataFrame,
    margin_df: pd.DataFrame | None = None,
    moneyflow_df: pd.DataFrame | None = None,
    fundamental_df: pd.DataFrame | None = None,
    financial_quality_df: pd.DataFrame | None = None,
    stock_mapping: dict | None = None,
) -> pd.DataFrame:
    """将融资融券、资金流、基本面数据 join 到特征矩阵。"""
    if feature_df is None or feature_df.empty:
        return feature_df

    df = feature_df.copy()
    df_index = df.set_index([COL_TS_CODE, COL_TRADE_DATE])

    # ── 融资融券 (margin_cache) ──
    if margin_df is not None and not margin_df.empty:
        margin = margin_df.copy()
        # 计算融资余额变化率
        margin["margin_balance_change"] = margin.groupby(COL_TS_CODE)["rzye"].pct_change(1)
        # 融资买入强度 = rzmre / amount(需匹配)
        margin["margin_buy_intensity"] = margin["rzmre"] / (margin["rzye"] + 1e-10) * 100
        # 融资净买入变化 (有 rzmre 但没有直接的偿还额字段, 用 rzye 日变化近似)
        margin["rzye_lag"] = margin.groupby(COL_TS_CODE)["rzye"].shift(1)
        margin["margin_net_buy"] = (margin["rzye"] - margin["rzye_lag"]).fillna(0)
        margin["margin_buy_intensity"] = margin["margin_buy_intensity"].fillna(0)

        margin_feats = margin.set_index([COL_TS_CODE, COL_TRADE_DATE])[
            ["margin_balance_change", "margin_buy_intensity", "margin_net_buy"]
        ]
        for col in margin_feats.columns:
            df_index[col] = margin_feats[col].fillna(0)

    # ── 资金流向 (moneyflow_cache) ──
    if moneyflow_df is not None and not moneyflow_df.empty:
        mf = moneyflow_df.copy()
        mf["net_mf_ratio"] = mf["net_mf_amount"] / (mf["buy_elg_amount"] + mf["sell_elg_amount"] + 1e-10)
        mf["mf_buy_sell_ratio"] = mf["buy_elg_amount"] / (mf["sell_elg_amount"] + 1e-10)
        mf["mf_consecutive"] = 0

        # 连续净流入天数
        for code in mf[COL_TS_CODE].unique():
            grp = mf[mf[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
            consec = 0
            for idx in grp.index:
                if grp.at[idx, "net_mf_amount"] > 0:
                    consec += 1
                else:
                    consec = 0
                mf.at[idx, "mf_consecutive"] = consec

        mf_feats = mf.set_index([COL_TS_CODE, COL_TRADE_DATE])[
            ["net_mf_ratio", "mf_buy_sell_ratio", "mf_consecutive"]
        ]
        for col in mf_feats.columns:
            df_index[col] = mf_feats[col]

        # mf_intensity_rank: 截面排名
        for date in mf[COL_TRADE_DATE].unique():
            mask = df_index.index.get_level_values(COL_TRADE_DATE) == date
            vals = df_index.loc[mask, "net_mf_ratio"].values
            if len(vals) > 0 and not np.all(vals == vals[0]):
                ranks = rankdata(vals, method="average")
                df_index.loc[mask, "mf_intensity_rank"] = (ranks / len(ranks)) * 100

    # ── 基本面 (fundamental_cache) ──
    if fundamental_df is not None and not fundamental_df.empty:
        fund = fundamental_df.copy()
        fund = fund.sort_values([COL_TS_CODE, COL_TRADE_DATE])

        # 盈利收益率
        fund["earnings_yield"] = fund["pe_ttm"].apply(lambda x: _safe_div(1, x) if x and x > 0 else 0)

        # 行业内 PE/PB 百分位（先初始化默认值，避免列不存在）
        fund["pe_industry_percentile"] = 50.0
        fund["pb_industry_percentile"] = 50.0
        if stock_mapping:
            fund["l1_code"] = fund[COL_TS_CODE].map(
                lambda c: (stock_mapping.get(c, {}) or {}).get("l1_code", "")
            )
            for metric in ["pe_ttm", "pb"]:
                pct_col = f"{metric}_industry_percentile"
                for date in fund[COL_TRADE_DATE].unique():
                    for ind in fund[fund[COL_TRADE_DATE] == date]["l1_code"].unique():
                        if not ind:
                            continue
                        mask = (fund[COL_TRADE_DATE] == date) & (fund["l1_code"] == ind)
                        subset = fund.loc[mask, metric].values
                        if len(subset) > 2 and not np.all(subset == subset[0]):
                            ranks = rankdata(subset, method="average")
                            fund.loc[mask, pct_col] = (ranks / len(ranks)) * 100

        fund_cols = [c for c in [
            "pe_ttm", "pb", "roe", "earnings_yield",
            "pe_industry_percentile", "pb_industry_percentile",
            "gross_margin", "total_mv",
        ] if c in fund.columns]
        fund_feats = fund.set_index([COL_TS_CODE, COL_TRADE_DATE])[fund_cols]
        for col in fund_feats.columns:
            df_index[col] = fund_feats[col]

    # ── 财务质量 (financial_quality_cache) ──
    if financial_quality_df is not None and not financial_quality_df.empty:
        fq = financial_quality_df.copy()
        fq = fq[fq["report_type"] == "1"]  # 年报
        if not fq.empty:
            # 财务比率
            for col, num, den in [
                ("net_profit_margin", "n_income", "revenue"),
                ("cfo_margin", "cfo", "revenue"),
                ("fcf_margin", "fcf", "revenue"),
                ("capex_intensity", "capex", "revenue"),
            ]:
                fq[col] = fq.apply(lambda r: _safe_div(r[num], r[den]), axis=1)

            # cash_conversion = (cfo - capex) / n_income
            fq["cash_conversion"] = fq.apply(
                lambda r: _safe_div(r["cfo"] - r["capex"], r["n_income"]) if r["n_income"] != 0 else 0,
                axis=1
            )

            # 对每只股票取最新财务数据
            fq = fq.sort_values([COL_TS_CODE, "end_date"])
            fq_latest = fq.groupby(COL_TS_CODE).last().reset_index()
            fq_feats = fq_latest.set_index(COL_TS_CODE)[
                ["net_profit_margin", "cfo_margin", "fcf_margin",
                 "capex_intensity", "cash_conversion"]
            ]
            for idx in df.index:
                code = df.at[idx, COL_TS_CODE]
                if code in fq_feats.index:
                    for col in fq_feats.columns:
                        val = fq_feats.at[code, col]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            df.at[idx, col] = val

    df.reset_index(inplace=True) if df_index.index.name else None
    # 如果 df_index 用 set_index 改过，需要 reset
    try:
        df = df_index.reset_index()
    except Exception:
        pass

    # 填充 NaN
    for col in EXTERNAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════
# 第六阶段：复合因子
# ═══════════════════════════════════════════════════════════════

def add_composite_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """基于已有特征计算复合因子。"""
    if feature_df is None or feature_df.empty:
        return feature_df

    df = feature_df.copy()

    # Board type from ts_code
    def _board(code: str) -> float:
        if code.startswith("60"):
            return 1.0  # 沪主板
        elif code.startswith("00"):
            return 2.0  # 深主板
        elif code.startswith("30"):
            return 3.0  # 创业板
        elif code.startswith("68"):
            return 4.0  # 科创板
        elif code.startswith("4") or code.startswith("8"):
            return 5.0  # 北交所
        else:
            return 0.0

    df["board_type"] = df[COL_TS_CODE].apply(_board)

    # Quality score: ROE + CFO/NP + gross_margin
    has_q = all(c in df.columns for c in ["roe", "gross_margin"])
    if has_q:
        df["quality_score"] = (
            df["roe"].fillna(0).rank(pct=True) * 0.4 +
            df["gross_margin"].fillna(0).rank(pct=True) * 0.3 +
            df["cfo_margin"].fillna(0).rank(pct=True) * 0.3
        )
    else:
        df["quality_score"] = 0.0

    # Value score: -(PE percentile + PB percentile) + FCF yield
    has_v = all(c in df.columns for c in ["pe_industry_percentile", "pb_industry_percentile"])
    if has_v:
        df["value_score"] = (
            -df["pe_industry_percentile"].fillna(50) * 0.5 +
            -df["pb_industry_percentile"].fillna(50) * 0.3 +
            df["fcf_margin"].fillna(0).rank(pct=True) * 100 * 0.2
        )
        df["value_score"] = df["value_score"].fillna(0)
    else:
        df["value_score"] = 0.0

    # Momentum quality: momentum filtered by low vol
    df["momentum_quality"] = df["mom20"].fillna(0) * (1 - df["atr_pct"].rank(pct=True).fillna(0.5))

    # Composite crowding: high momentum + high crowding + declining volume
    has_crowd = all(c in df.columns for c in ["amount_ratio", "vol_ratio"])
    if has_crowd:
        df["composite_crowding"] = (
            df["mom20"].rank(pct=True) * 0.3 +
            df["amount_ratio"].rank(pct=True) * 0.4 +
            (1 - df["vol_ma5_ratio"].fillna(1).rank(pct=True)) * 0.3
        )
    else:
        df["composite_crowding"] = 0.0

    return df


# ═══════════════════════════════════════════════════════════════
# 第七阶段：V5 扩展特征（向量化）
# ═══════════════════════════════════════════════════════════════

def _rsi_vec(series: pd.Series, period: int) -> pd.Series:
    """向量化 RSI 计算。"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def add_v5_extension_features(
    feature_df: pd.DataFrame,
    stock_daily_df: pd.DataFrame | None = None,
    stock_mapping: dict | None = None,
) -> pd.DataFrame:
    """计算 V5 扩展特征（仅处理 feature_df 中已有股票）。

    用向量化 pandas 操作从 stock_daily_df 计算 V5 新增特征，
    然后 join 到 feature_df 上（(ts_code, trade_date) 精确对齐）。

    Args:
        feature_df: V4 特征矩阵（已有趋势交易的 118 个基础特征）
        stock_daily_df: 原始日线 OHLCV 数据（含 open/high/low/close/vol/amount/pct_chg）
        stock_mapping: 行业映射 {ts_code: {l1_code, l2_code}}

    Returns:
        补全了 V5 扩展列的 feature_df
    """
    if feature_df is None or feature_df.empty:
        return feature_df
    if stock_daily_df is None or stock_daily_df.empty:
        return feature_df

    df = feature_df.copy()
    codes_in_feat = set(df[COL_TS_CODE].unique())

    # 仅对 feature_df 中已有的股票处理
    raw = stock_daily_df[stock_daily_df[COL_TS_CODE].isin(codes_in_feat)].copy()
    raw = raw.sort_values([COL_TS_CODE, COL_TRADE_DATE])
    if raw.empty:
        return df

    raw[COL_TRADE_DATE] = raw[COL_TRADE_DATE].astype(str)
    result = raw[[COL_TS_CODE, COL_TRADE_DATE]].copy()
    grouped = raw.groupby(COL_TS_CODE)

    # ════════════════════════════════════════════════════
    # A2: 多尺度动量
    # ════════════════════════════════════════════════════
    for p in [2, 3, 4, 7, 12, 15, 30, 45, 75, 90, 120]:
        result[f"mom{p}"] = grouped["close"].pct_change(periods=p) * 100

    # mom_harmonic_mean (use only V5 computed momentums)
    mom_base = [c for c in result.columns if c.startswith("mom") and c not in ("mom_harmonic_mean", "mom_frequency_ratio")]
    if mom_base:
        mom_vals = result[mom_base].fillna(0).abs().clip(lower=0.01)
        result["mom_harmonic_mean"] = len(mom_base) / mom_vals.apply(
            lambda r: sum(1.0 / max(abs(v), 0.01) for v in r), axis=1)
    # mom_frequency_ratio
    all_moms = result[[c for c in result.columns if c.startswith("mom") and c not in ("mom_harmonic_mean", "mom_frequency_ratio")]]
    pos_sum = all_moms.clip(lower=0).sum(axis=1)
    neg_sum = all_moms.clip(upper=0).abs().sum(axis=1)
    result["mom_frequency_ratio"] = pos_sum / neg_sum.replace(0, np.nan)

    # ════════════════════════════════════════════════════
    # A3: 均线扩展
    # ════════════════════════════════════════════════════
    for p in [3, 8, 15, 30, 40, 80, 90]:
        ma = grouped["close"].transform(lambda x: x.rolling(p, min_periods=p).mean())
        result[f"ma{p}_dev"] = ((raw["close"] / ma) - 1) * 100

    ma5 = grouped["close"].transform(lambda x: x.rolling(5).mean())
    ma10 = grouped["close"].transform(lambda x: x.rolling(10).mean())
    ma20 = grouped["close"].transform(lambda x: x.rolling(20).mean())
    ma60 = grouped["close"].transform(lambda x: x.rolling(60).mean())
    ma120 = grouped["close"].transform(lambda x: x.rolling(120).mean())
    result["ma_short_arrangement"] = ((ma5 > ma10) & (ma10 > ma20)).astype(float)
    result["ma_long_arrangement"] = ((ma20 > ma60) & (ma60 > ma120)).astype(float)
    result["ma5_ma10_gap"] = ((ma5 - ma10) / raw["close"].replace(0, np.nan)) * 100
    result["ma20_ma60_gap"] = ((ma20 - ma60) / raw["close"].replace(0, np.nan)) * 100

    # ════════════════════════════════════════════════════
    # A4: MACD
    # ════════════════════════════════════════════════════
    macd_parts = []
    for code, grp in raw.groupby(COL_TS_CODE):
        grp = grp.sort_values(COL_TRADE_DATE)
        c = grp["close"]
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        signal = dif.ewm(span=9, adjust=False).mean()
        hist = dif - signal
        macd_parts.append(pd.DataFrame({
            "ts_code": code, "trade_date": grp[COL_TRADE_DATE].values,
            "macd_dif": dif.values, "macd_signal": signal.values,
            "macd_hist": hist.values,
        }, index=grp.index))
    if macd_parts:
        macd_df = pd.concat(macd_parts).sort_index()
        result["macd_dif"] = macd_df["macd_dif"]
        result["macd_signal"] = macd_df["macd_signal"]
        result["macd_hist"] = macd_df["macd_hist"]
        result["macd_cross_long"] = (macd_df["macd_dif"] > macd_df["macd_signal"]).astype(float)
        result["macd_histogram_accel"] = macd_df["macd_hist"].diff().fillna(0)
        zero_cross = ((macd_df["macd_dif"] >= 0) & (macd_df["macd_dif"].shift(1) < 0)).astype(float)
        result["macd_zero_cross"] = zero_cross.groupby(raw[COL_TS_CODE].values).transform(
            lambda x: x.rolling(10, min_periods=1).sum())

    # ════════════════════════════════════════════════════
    # A5: 成交量扩展
    # ════════════════════════════════════════════════════
    ma20_vol = grouped["vol"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    ma10_amt = grouped["amount"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    result["amount_ma10_ratio"] = raw["amount"] / ma10_amt.replace(0, np.nan)
    result["vol_shock_count_5d"] = (raw["vol"] > ma20_vol * 1.5).astype(float)
    result["vol_shock_count_5d"] = result["vol_shock_count_5d"].groupby(
        raw[COL_TS_CODE].values).transform(lambda x: x.rolling(5, min_periods=1).sum())
    ma5_vol = grouped["vol"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    vol_osc = (ma5_vol - ma20_vol) / ma20_vol.replace(0, np.nan)
    result["volume_oscillator_accel"] = vol_osc.diff()
    result["volume_dry_up_flag"] = ((raw["vol"] < ma20_vol * 0.5) & (vol_osc < -0.3)).astype(float)

    # ════════════════════════════════════════════════════
    # A6: 波动率扩展
    # ════════════════════════════════════════════════════
    pct_chg = raw["pct_chg"]
    result["realized_vol_5d"] = pct_chg.groupby(raw[COL_TS_CODE].values).transform(
        lambda x: x.rolling(5, min_periods=1).std())
    result["realized_vol_10d"] = pct_chg.groupby(raw[COL_TS_CODE].values).transform(
        lambda x: x.rolling(10, min_periods=1).std())
    atr_14 = pct_chg.groupby(raw[COL_TS_CODE].values).transform(
        lambda x: x.rolling(14).std())
    result["vol_surprise"] = ((atr_14 - atr_14.rolling(60).mean()) /
                               atr_14.rolling(60).std().replace(0, np.nan))
    result["vol_change_acceleration"] = atr_14.diff().diff()
    result["vol_regime_adaptive_ma"] = atr_14 / atr_14.rolling(60).mean().replace(0, np.nan)

    # ════════════════════════════════════════════════════
    # A7: BB 扩展
    # ════════════════════════════════════════════════════
    close_g = grouped["close"]
    ma20_close = close_g.transform(lambda x: x.rolling(20).mean())
    std20_close = close_g.transform(lambda x: x.rolling(20).std())
    upper_bb = ma20_close + 2 * std20_close
    lower_bb = ma20_close - 2 * std20_close
    result["bb_cross_upper"] = (raw["close"] > upper_bb).astype(float)
    result["bb_cross_lower"] = (raw["close"] < lower_bb).astype(float)
    # bb_squeeze_flag: 布林带宽度 < 历史均值 * 0.5
    bb_width = (2 * std20_close / ma20_close.replace(0, np.nan)) * 100
    bb_width_hist = bb_width.groupby(raw[COL_TS_CODE].values).transform(
        lambda x: x.rolling(60).mean())
    result["bb_squeeze_flag"] = (bb_width < bb_width_hist * 0.5).astype(float)
    # inside_bb_days
    inside = ((raw["close"] >= lower_bb) & (raw["close"] <= upper_bb)).astype(float)
    result["inside_bb_days"] = inside.groupby(raw[COL_TS_CODE].values).transform(
        lambda x: x.rolling(5, min_periods=1).sum())

    # ════════════════════════════════════════════════════
    # A9: RSI 扩展
    # ════════════════════════════════════════════════════
    for p in [3, 5, 7, 21, 60]:
        rsi_vals = grouped["close"].transform(lambda x, pp=p: _rsi_vec(x, pp))
        result[f"rsi_{p}"] = rsi_vals
    result["rsi_momentum"] = result["rsi_5"] - result["rsi_21"]
    result["rsi_divergence_bull"] = 0.0
    result["rsi_divergence_bear"] = 0.0
    # 保留 V4 RSI(14) 不变

    # ════════════════════════════════════════════════════
    # A10: VWAP/日内分析
    # ════════════════════════════════════════════════════
    result["intraday_vwap_proxy"] = raw["amount"] / (raw["vol"] * 100 + 1e-10)
    result["close_vs_intraday_ratio"] = ((raw["close"] / result["intraday_vwap_proxy"]) - 1) * 100
    result["open_vs_intraday_ratio"] = ((raw["open"] / result["intraday_vwap_proxy"]) - 1) * 100
    result["high_vs_close_spread"] = (raw["high"] - raw["close"]) / raw["close"].replace(0, np.nan) * 100
    result["low_vs_close_spread"] = (raw["low"] - raw["close"]) / raw["close"].replace(0, np.nan) * 100
    candle_range = raw["high"] - raw["low"]
    result["open_position_ratio"] = (raw["close"] - raw["open"]) / candle_range.replace(0, np.nan)
    body = (raw["close"] - raw["open"]).abs()
    result["body_to_range_ratio"] = body / candle_range.replace(0, np.nan) * 100
    result["open_position_accel"] = result["open_position_ratio"].diff()

    # ════════════════════════════════════════════════════
    # A12: 流动性扩展
    # ════════════════════════════════════════════════════
    abs_ret = raw["pct_chg"].abs() / 100
    amihud_daily = abs_ret / raw["amount"].replace(0, np.nan) * 1e6
    result["amihud_5d_mean"] = amihud_daily.groupby(raw[COL_TS_CODE].values).transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    ma20_vol_liq = raw.groupby(COL_TS_CODE)["vol"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    result["volume_shock_indicator"] = raw["vol"] / ma20_vol_liq.replace(0, np.nan)

    # ════════════════════════════════════════════════════
    # B: 行业涨停生态
    # ════════════════════════════════════════════════════
    if stock_mapping:
        raw["l1_code"] = raw[COL_TS_CODE].map(
            lambda c: stock_mapping.get(c, {}).get("l1_code", ""))
        raw["l2_code"] = raw[COL_TS_CODE].map(
            lambda c: stock_mapping.get(c, {}).get("l2_code", ""))
        def _board_v(code):
            return 3.0 if code.startswith("30") or code.startswith("68") else 1.0
        raw["board_type_v5"] = raw[COL_TS_CODE].apply(_board_v)
        raw["is_limit_up"] = raw.apply(
            lambda r: r["pct_chg"] >= (19.5 if r["board_type_v5"] >= 3 else 9.5), axis=1)
        raw["is_limit_down"] = raw.apply(
            lambda r: r["pct_chg"] <= (-19.5 if r["board_type_v5"] >= 3 else -9.5), axis=1)
        raw["is_up"] = raw["pct_chg"] > 0

        # 行业统计（向量化 groupby）
        for level, code_col in [("l1", "l1_code"), ("l2", "l2_code")]:
            lt_grp = raw[raw["is_limit_up"]].groupby([COL_TRADE_DATE, code_col]).size()
            dt_grp = raw[raw["is_limit_down"]].groupby([COL_TRADE_DATE, code_col]).size()
            up_grp = raw[raw["is_up"]].groupby([COL_TRADE_DATE, code_col]).size()
            total_grp = raw.groupby([COL_TRADE_DATE, code_col]).size()
            for name, grp in [("lt_count", lt_grp), ("dt_count", dt_grp),
                              ("up_count", up_grp), ("total", total_grp)]:
                if not grp.empty:
                    s = grp.reset_index(name=f"{name}_{level}")
                    raw = raw.merge(s, on=[COL_TRADE_DATE, code_col], how="left")
            # 补全空 group 导致缺失的列
            for prefix in ["lt_count", "dt_count", "up_count", "total"]:
                col = f"{prefix}_{level}"
                if col not in raw.columns:
                    raw[col] = 0.0
                else:
                    raw[col] = raw[col].fillna(0).astype(float)
            total_c = f"total_{level}"
            lt_c = f"lt_count_{level}"
            dt_c = f"dt_count_{level}"
            up_c = f"up_count_{level}"
            raw[f"lt_rate_{level}"] = raw[lt_c] / raw[total_c].replace(0, np.nan) * 100
            raw[f"up_pct_{level}"] = raw[up_c] / raw[total_c].replace(0, np.nan) * 100
            lt_dt_sum = raw[lt_c] + raw[dt_c]
            raw[f"lt_to_dt_ratio_{level}"] = raw[lt_c] / lt_dt_sum.replace(0, np.nan)
            for col in [f"lt_count_{level}", f"dt_count_{level}", f"lt_rate_{level}",
                        f"up_pct_{level}", f"lt_to_dt_ratio_{level}"]:
                result[col] = raw[col].fillna(0)

        # 行业排名
        for level, code_col in [("l1", "l1_code"), ("l2", "l2_code")]:
            ind_mean = raw.groupby([COL_TRADE_DATE, code_col])["pct_chg"].mean().reset_index()
            ind_mean[f"rank_{level}_in_all"] = ind_mean.groupby(COL_TRADE_DATE)["pct_chg"].rank(pct=True) * 100
            raw = raw.merge(ind_mean[[code_col, COL_TRADE_DATE, f"rank_{level}_in_all"]],
                            on=[code_col, COL_TRADE_DATE], how="left")
            result[f"rank_{level}_in_all"] = raw[f"rank_{level}_in_all"].fillna(50)

        # stock_vs_L1_alpha_daily
        l1_mean = raw.groupby([COL_TRADE_DATE, "l1_code"])["pct_chg"].transform("mean")
        raw["stock_vs_L1_alpha_daily"] = raw["pct_chg"] - l1_mean
        result["stock_vs_L1_alpha_daily"] = raw["stock_vs_L1_alpha_daily"].fillna(0)

    # ════════════════════════════════════════════════════
    # V4 交叉特征（已有但在 ALL_FEATURES 中缺失）
    # ════════════════════════════════════════════════════
    # 这些特征已在 build_feature_matrix 中计算并存入 df，
    # 但未被 ALL_FEATURES 收录，所以这里只确保列存在。
    # 如果 df 中已有，不再覆盖
    v4_cross = ["momentum_volume", "price_volume_diverg",
                "bb_streak", "gap_momentum",
                "position_conviction", "vol_drawdown"]
    for col in v4_cross:
        if col not in df.columns and col in result.columns:
            pass  # will be joined below

    # ── B9/B10: 日历 + 市场广度特征（从 trade_date 计算）──
    raw[COL_TRADE_DATE] = raw[COL_TRADE_DATE].astype(str)
    date_series = pd.to_datetime(raw[COL_TRADE_DATE], format="%Y%m%d", errors="coerce")
    result["day_of_week_effect"] = date_series.dt.dayofweek.map({0:1,4:5,1:2,2:3,3:4}).fillna(3)
    result["monthly_mean_reversion_signal"] = (date_series.dt.day <= 5).astype(float)
    result["annual_pattern_week_nr"] = date_series.dt.isocalendar().week.astype(float)
    result["quarter_earnings_window"] = date_series.dt.month.isin([1,4,7,10]).astype(float)
    result["month_end_effect"] = date_series.dt.is_month_end.astype(float)
    result["year_end_window_dressing"] = (date_series.dt.month == 12).astype(float)
    result["tax_loss_selling_window"] = ((date_series.dt.month == 12) & (date_series.dt.day >= 15)).astype(float)
    result["holiday_proximity_days"] = 0.0
    result["lunar_new_year_proximity"] = 0.0
    result["golden_week_effect"] = 0.0

    # 市场广度
    breadth_map = {}
    for date_val, day_df in raw.groupby(COL_TRADE_DATE):
        n_up = (day_df["pct_chg"] > 0).sum()
        n_down = (day_df["pct_chg"] < 0).sum()
        total = len(day_df)
        breadth_map[date_val] = {
            "advance_decline_ratio": _safe_div(n_up, n_down),
            "total_up_days": float(n_up),
            "market_extreme_signal": 1.0 if (n_up > total * 0.8 or n_down > total * 0.8) else 0.0,
            "market_regime_score": float(day_df["pct_chg"].mean()),
            "bull_bear_power": float((day_df["pct_chg"] > 2).sum() - (day_df["pct_chg"] < -2).sum()),
        }
    for col in ["advance_decline_ratio","total_up_days","market_extreme_signal",
                "market_regime_score","bull_bear_power"]:
        result[col] = raw[COL_TRADE_DATE].map(
            {k: v.get(col, 0) for k, v in breadth_map.items()}).fillna(0)

    # 剩余广度特征暂置 0（简化）
    for col in ["new_high_count_20d","new_low_count_20d","breadth_momentum",
                "market_fear_greed_proxy","breadth_thrust_flag","adtl_ratio_5d_ma",
                "breadth_confirmation_score","breadth_ratio_trend",
                "market_mcginley_dynamic","bulk_market_strength_index"]:
        result[col] = 0.0

    # ── B8: 行业扩展（简化占位）──
    for col in ["industry_rank_price_move","industry_vol_rank","industry_turnover_rank",
                "industry_moneyflow_net_5d","industry_moneyflow_share",
                "industry_price_divergence","L3_strength_index",
                "industry_mom_acceleration","industry_concentration_HHI"]:
        result[col] = 0.0

    # ── A14: 非线性变换（从 df 已有特征，post-join 计算）──
    # 这些需要在 df 上计算，标记一下
    _need_post = ["rank_mom20_winsor","rank_vol_ratio_winsor","rank_atr_pct_winsor",
                  "zscore_mom20_robust","zscore_excess_return_log",
                  "log1p_total_mv","huber_delta_mom20","huber_delta_vol_ratio"]

    # ── A15/A16: 交互/自适应特征（占位，post-join 计算）──
    for col in ["mom20_x_low_vol","vol_ratio_x_mom1","mom20_x_bb_width",
                "vol_ratio_x_atr_pct","turnover_x_price_pos","streak_x_vol_ratio",
                "rsi_x_mom20","gap_x_mom5","amount_x_pct_chg",
                "bb_pct_b_x_vol_ratio","macd_hist_x_mom5",
                "quality_x_mom20","fcf_yield_x_mom120",
                "excess_return_x_industry_rank","industry_alpha_x_vol_surprise",
                "rsi_x_bb_width_direction","macd_hist_x_vwap_dev",
                "atr_x_vol_ratio_diverg","streak_x_mf_buy_sell",
                "turnover_x_mom20_diverg","price_pos_x_amount_surge",
                "gap_x_rsi_divergence","quality_x_momentum_consistency",
                "beta_x_idio_vol_ratio","sector_momentum_x_lt_rate",
                "mom20_x_industry_rank","quality_x_value_residual",
                "leverage_x_mom20","float_mv_x_turnover","mf_x_mom_convergence",
                "adaptive_ma_short","adaptive_ma_long","vol_regime_flag",
                "trend_strength_score","adaptive_atr_stop","adaptive_bb_window",
                "adaptive_mom_window","vol_regime_streak",
                "adaptive_vol_target_exposure","dynamic_stop_loss_pct",
                "regime_adaptive_momentum_threshold","adaptive_position_sizing_signal",
                "market_residual_momentum","idiosyncratic_vol_20d",
                "market_impact_coeff","residual_skewness_20d","alpha_persistence",
                "growth_score","efficiency_score","quality_value_residual",
                "composite_regime_factor","quality_growth_accel",
                "value_momentum_divergence","low_vol_anomaly_score",
                "momentum_crowding_divergence"]:
        result[col] = 0.0

    # ── Join V5 新特征到 feature_df ──
    result[COL_TS_CODE] = result[COL_TS_CODE].astype(str)
    result[COL_TRADE_DATE] = result[COL_TRADE_DATE].astype(str)

    v5_join_cols = [c for c in V5_EXT_FEATURES if c in result.columns and c not in df.columns]
    if v5_join_cols:
        join_data = result[[COL_TS_CODE, COL_TRADE_DATE] + v5_join_cols]
        df = df.merge(join_data, on=[COL_TS_CODE, COL_TRADE_DATE], how="left")

    # 统一填充
    for c in V5_EXT_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ── Post-join: 全部剩余 V5 计算 ──
    # A6: 波动率扩展
    if "atr_pct" in df.columns:
        grp = df.groupby(COL_TS_CODE)["atr_pct"]
        df["atr_ratio_14_7"] = (grp.transform(lambda x: x.rolling(14).mean()) /
                                 grp.transform(lambda x: x.rolling(7).mean()).replace(0, np.nan)).fillna(0)
        atr_ma20 = grp.transform(lambda x: x.rolling(20).mean())
        atr_ma60 = grp.transform(lambda x: x.rolling(60).mean())
        df["atr_regime_short_term"] = (atr_ma20 / atr_ma60.replace(0, np.nan)).fillna(1)
        df["realized_vol_half_month"] = df.get("downside_vol", pd.Series(0)).fillna(0) * np.sqrt(252)
    # A7: BB 扩展
    if "bb_pct_b" in df.columns:
        df["bb_convergence"] = (df["bb_pct_b"] - 50).abs()
        df["bb_expansion_rate"] = df.groupby(COL_TS_CODE)["bb_width"].pct_change(5).fillna(0)
        df["bb_position_accel"] = df.groupby(COL_TS_CODE)["bb_pct_b"].diff().fillna(0)
    # A10: VWAP 扩展
    if "vwap_deviation" in df.columns:
        df["vwap_ma20_divergence"] = (df["vwap_deviation"] -
                                       df.groupby(COL_TS_CODE)["vwap_deviation"].transform(
                                           lambda x: x.rolling(20).mean())).fillna(0)
    # B: 行业涨停
    if "lt_count_l1" in df.columns and "dt_count_l1" in df.columns:
        s = df["lt_count_l1"].fillna(0) + df["dt_count_l1"].fillna(0)
        df["lt_to_dt_ratio"] = (df["lt_count_l1"].fillna(0) / s.replace(0, np.nan)).fillna(0)
    if "lt_rate_l1" in df.columns and "rank_l1_in_all" in df.columns:
        df["lt_rate_mul_L1_rank"] = (df["lt_rate_l1"] * df["rank_l1_in_all"] / 100).fillna(0)
    if "lt_count_l1" in df.columns:
        df["lt_rate_trend_5d"] = df.groupby(COL_TS_CODE)["lt_count_l1"].diff(5).fillna(0)
        df["lt_count_mom_accel"] = df.groupby(COL_TS_CODE)["lt_count_l1"].diff().diff().fillna(0)
    df["is_20cm_eligible"] = df[COL_TS_CODE].apply(
        lambda c: 1.0 if c.startswith("30") or c.startswith("68") else 0.0)
    # C1: 融资融券
    for src in ["margin_buy_intensity", "margin_balance_change"]:
        if src in df.columns:
            df["margin_buy_intensity_ratio"] = df["margin_buy_intensity"]
            df["margin_price_divergence"] = df["margin_balance_change"]
            break
    # C2: 资金流
    if "buy_elg_amount" in df.columns and "sell_elg_amount" in df.columns:
        denom = (df["buy_elg_amount"] + df["sell_elg_amount"]).replace(0, np.nan)
        df["mf_buy_sell_rate_change"] = (df["buy_elg_amount"] / denom).diff().fillna(0)
        df["mf_sustained_divergence_5d"] = (
            abs(df["buy_elg_amount"] - df["sell_elg_amount"]).fillna(0) / denom.replace(0, np.nan)).fillna(0)
    # A5: 成交量扩展
    if "turnover_ratio_5d_20d" in df.columns:
        df["turnover_std_rank"] = df.groupby(COL_TRADE_DATE)["turnover_ratio_5d_20d"].rank(pct=True).fillna(0.5) * 100
    if "vol_ratio" in df.columns:
        df["volume_ratio_zscore_20d"] = df.groupby(COL_TS_CODE)["vol_ratio"].transform(
            lambda x: ((x - x.rolling(20).mean()) / x.rolling(20).std().replace(0, np.nan))).fillna(0)
    # D1
    if "roe" in df.columns:
        df["roa"] = df["roe"].fillna(0) * 0.6
    # A15/A16 extra
    if "mom3" in df.columns and "mom20" in df.columns:
        df["mom3_mom20_ratio"] = (df["mom3"].fillna(0) /
                                   df["mom20"].fillna(0.01).abs().clip(lower=0.01)).fillna(0).clip(-10, 10)
    if "atr_pct" in df.columns:
        df["regime_adaptive_weight"] = (1 / (1 + df["atr_pct"].fillna(0))).fillna(0.5)
        df["vol_regime_zscore_position"] = df.groupby(COL_TS_CODE)["atr_pct"].transform(
            lambda x: ((x - x.rolling(60).mean()) / x.rolling(60).std().replace(0, np.nan))).fillna(0)
    # A14 截面变换
    for date_val in df[COL_TRADE_DATE].unique():
        mask = df[COL_TRADE_DATE] == date_val
        for src, tgt in [("mom20","rank_mom20_winsor"),("vol_ratio","rank_vol_ratio_winsor"),
                         ("atr_pct","rank_atr_pct_winsor")]:
            if tgt in df.columns and src in df.columns:
                vals = df.loc[mask, src].values
                if len(vals) > 2 and not np.all(vals == vals[0]):
                    from scipy.stats import rankdata
                    df.loc[mask, tgt] = (rankdata(vals, method="average") / len(vals)) * 100
        for src, tgt in [("mom20","zscore_mom20_robust"),("excess_return_l1","zscore_excess_return_log")]:
            if tgt in df.columns and src in df.columns:
                vals = df.loc[mask, src].values
                mu, sigma = np.mean(vals), np.std(vals, ddof=1)
                if sigma > 1e-10:
                    df.loc[mask, tgt] = (vals - mu) / sigma
        if "log1p_total_mv" in df.columns and "total_mv" in df.columns:
            df.loc[mask, "log1p_total_mv"] = np.log1p(df.loc[mask, "total_mv"].fillna(0).abs())
    # A15 交互
    _pairs = [("mom20",lambda df: df["atr_pct"].rank(pct=True),"mom20_x_low_vol"),
              ("vol_ratio","mom1","vol_ratio_x_mom1"),
              ("bb_pct_b","vol_ratio","bb_pct_b_x_vol_ratio"),
              ("macd_hist","mom5","macd_hist_x_mom5"),
              ("gap_pct","mom5","gap_x_mom5"),
              ("rsi_14","mom20","rsi_x_mom20"),
              ("streak","vol_ratio","streak_x_vol_ratio"),
              ("turnover_ratio_5d_20d","price_position_20d","turnover_x_price_pos"),
              ("mom20","bb_width","mom20_x_bb_width"),
              ("vol_ratio","atr_pct","vol_ratio_x_atr_pct")]
    for a,b,t in _pairs:
        if t in df.columns:
            c1 = a(df) if callable(a) else (df[a] if a in df.columns else None)
            c2 = b(df) if callable(b) else (df[b] if b in df.columns else None)
            if c1 is not None and c2 is not None:
                df[t] = pd.to_numeric(c1 * c2, errors="coerce").fillna(0)
    # A16 自适应
    if "trend_strength_score" in df.columns and "mom20" in df.columns and "atr_pct" in df.columns:
        df["trend_strength_score"] = ((df["mom20"].fillna(0).abs() - df["atr_pct"].fillna(0) * 2) /
                                       df["atr_pct"].fillna(1).clip(lower=0.1)).fillna(0)
    if "vol_regime_flag" in df.columns and "atr_pct" in df.columns:
        atr_ma60 = df.groupby(COL_TS_CODE)["atr_pct"].transform(lambda x: x.rolling(60).mean())
        df["vol_regime_flag"] = (df["atr_pct"] > atr_ma60 * 0.8).astype(float).fillna(0)
    # D4 复合
    if "growth_score" in df.columns and "fcf_margin" in df.columns:
        df["growth_score"] = (df["fcf_margin"].fillna(0).rank(pct=True) * 0.6 +
                               (1 - df.get("revenue_cfo_gap", pd.Series(0)).fillna(0).rank(pct=True)) * 0.4)
    if "value_momentum_divergence" in df.columns and "value_score" in df.columns and "mom20" in df.columns:
        df["value_momentum_divergence"] = df["value_score"].fillna(0).rank(pct=True) - \
                                           df["mom20"].fillna(0).rank(pct=True)
    if "low_vol_anomaly_score" in df.columns and "atr_pct" in df.columns and "mom20" in df.columns:
        df["low_vol_anomaly_score"] = df["mom20"].fillna(0).rank(pct=True) * \
                                       (1 - df["atr_pct"].fillna(0).rank(pct=True))

    # 再次统一填充所有 V5 列
    for c in V5_EXT_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    logger.info("  V5 扩展特征: %d 列已添加", len(v5_join_cols))
    return df


def _compute_stock_sequential(sdf: pd.DataFrame) -> pd.DataFrame:
    """计算需要逐股时序处理的特征。
    每只股票调用一次，用 numpy 批处理而非逐日 Python 循环。
    """
    n = len(sdf)
    c = sdf[COL_CLOSE].values
    h = sdf[COL_HIGH].values
    l = sdf[COL_LOW].values
    o = sdf[COL_OPEN].values

    r = np.zeros(n)
    r[1:] = c[1:] / c[:-1] * 100 - 100

    streak = np.zeros(n, dtype=np.float64)
    hh_count = np.zeros(n, dtype=np.float64)
    cs_spread = np.zeros(n, dtype=np.float64)
    hammer = np.zeros(n, dtype=np.float64)
    shooting = np.zeros(n, dtype=np.float64)
    doji = np.zeros(n, dtype=np.float64)
    engulf_bull = np.zeros(n, dtype=np.float64)
    engulf_bear = np.zeros(n, dtype=np.float64)
    evening_star = np.zeros(n, dtype=np.float64)
    three_soldiers = np.zeros(n, dtype=np.float64)
    three_crows = np.zeros(n, dtype=np.float64)
    usr = np.zeros(n, dtype=np.float64)
    mom_cons = np.zeros(n, dtype=np.float64)
    exp_wgt = np.zeros(n, dtype=np.float64)
    yz_vol = np.zeros(n, dtype=np.float64)

    w20 = np.exp(-0.2 * np.arange(20))
    w20 /= w20.sum()

    for i in range(n):
        ci = c[i]; oi = o[i]; hi = h[i]; li = l[i]

        # streak (10-day lookback, most-recent-first, match old)
        if i >= 1:
            sv = 0
            for k in range(1, min(i, 10) + 1):  # k=1: most recent pair
                if c[i - k + 1] > c[i - k]:
                    sv = sv + 1 if sv >= 0 else 1
                elif c[i - k + 1] < c[i - k]:
                    sv = sv - 1 if sv <= 0 else -1
                else:
                    break
            streak[i] = sv

        # HH count (match old: 10-day lookback, strict max equals)
        cnt = 0
        for k in range(1, min(11, i+1)):
            if i >= k and c[i-k+1] == np.max(c[i-k:i+1]): cnt += 1
        hh_count[i] = cnt

        # Corwin-Schultz
        if i >= 1:
            beta = np.log(hi/li)**2 + np.log(h[i-1]/l[i-1])**2
            gamma = np.log(max(hi, h[i-1]) / min(li, l[i-1]))**2
            cs = 2*(np.exp(beta)-np.exp(gamma)) / max(1e-10, 2+np.exp(beta)-2*np.exp(gamma))
            cs_spread[i] = max(0, cs) * 100

        # Exp weighted return (20d)
        if i >= 20:
            exp_wgt[i] = np.dot(r[i-19:i+1], w20)

        # momentum_consistency
        if i >= 20:
            ret5 = (ci / c[i-5] - 1) * 100 if c[i-5] > 0 else 0
            if abs(ret5) >= 0.01:
                ref = ret5 >= 0
                cons = 0
                for k in range(1, 11):
                    rk = (c[i-k+1] / c[i-k] - 1) * 100 if c[i-k] > 0 else 0
                    if (rk >= 0) == ref: cons += 1
                    else: break
                mom_cons[i] = cons

        # Yang-Zhang vol (20d rolling)
        if i >= 20:
            yz_vol[i] = _compute_yang_zhang_vol(o[i-19:i+1], h[i-19:i+1],
                                                 l[i-19:i+1], c[i-19:i+1])

        # Candlestick (all row-wise)
        body = abs(ci - oi)
        tr_ = hi - li
        if tr_ > 0:
            us = hi - max(ci, oi)
            ls_ = min(ci, oi) - li
            usr[i] = us / tr_
            if body > 0:
                if ls_ >= 2*body and us <= body*0.3:  hammer[i] = 1.0
                if us >= 2*body and ls_ <= body*0.3:  shooting[i] = 1.0
                if body / tr_ < 0.05:  doji[i] = 1.0
                if i >= 1:
                    pb = abs(c[i-1]-o[i-1])
                    if pb > 0:
                        if c[i-1] < o[i-1] and ci > oi and ci >= o[i-1] and oi <= c[i-1]:
                            engulf_bull[i] = 1.0
                        if c[i-1] > o[i-1] and ci < oi and oi >= c[i-1] and ci <= o[i-1]:
                            engulf_bear[i] = 1.0
                if i >= 2:
                    b1 = abs(c[i-2]-o[i-2]); b2 = abs(c[i-1]-o[i-1])
                    if b1 > 0 and b2 > 0 and body > 0:
                        avg20 = np.mean(c[max(0,i-20):i+1])
                        if c[i-2] > o[i-2] and b1 > avg20*0.02 and b2 < b1*0.3 and ci < oi and ci < c[i-2]-b1*0.5:
                            evening_star[i] = 1.0
                    if i >= 2 and c[i-2] > o[i-2] and c[i-1] > o[i-1] and ci > oi and c[i-1] > c[i-2] and ci > c[i-1]:
                        if i < 3 or c[i-2] > c[i-3]: three_soldiers[i] = 1.0
                    if i >= 2 and c[i-2] < o[i-2] and c[i-1] < o[i-1] and ci < oi and c[i-1] < c[i-2] and ci < c[i-1]:
                        if i < 3 or c[i-2] < c[i-3]: three_crows[i] = 1.0

    # RSI(14) — vectorized ewm
    gains = np.maximum(r, 0); losses = np.maximum(-r, 0)
    ag = pd.Series(gains).ewm(span=14, adjust=False, min_periods=7).mean().values
    al = pd.Series(losses).ewm(span=14, adjust=False, min_periods=7).mean().values
    rs = np.divide(ag, al, out=np.ones(n), where=al>1e-10)
    rsi_arr = 100 - 100/(1+rs)

    return pd.DataFrame({
        "streak": streak, "higher_high_count": hh_count,
        "candle_hammer": hammer, "candle_shooting_star": shooting,
        "candle_doji": doji, "candle_engulfing_bull": engulf_bull,
        "candle_engulfing_bear": engulf_bear, "candle_evening_star": evening_star,
        "candle_three_soldiers": three_soldiers, "candle_three_crows": three_crows,
        "upper_shadow_ratio": usr, "corwin_schultz_spread": cs_spread,
        "exp_wgt_return_20d": exp_wgt, "rsi_14": rsi_arr,
        "momentum_consistency": mom_cons, "yang_zhang_vol": yz_vol,
    }, index=sdf.index)


def build_feature_matrix_vectorized(
    stock_daily_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame | None = None,
    stock_mapping: dict | None = None,
    persistence_scores: dict | None = None,
    multi_day: bool = False,
) -> pd.DataFrame:
    """向量化 V4 特征构建（替代 build_feature_matrix 的 per-stock 循环）。

    全量 GroupBy + Rolling 操作 + 单次逐股 apply(仅时序特征)。
    覆盖 momentum/MA/volume/volatility/BB/K线/统计/流动性 ~80 列。
    """
    if stock_daily_df is None or stock_daily_df.empty:
        return pd.DataFrame()
    df = stock_daily_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    codes = df[COL_TS_CODE].unique()
    logger.info("   向量化特征: %d 只股票, %d 行", len(codes), len(df))
    grp = df.groupby(COL_TS_CODE)

    # 日收益
    df["_ret"] = grp[COL_CLOSE].pct_change().fillna(0) * 100

    # ── 动量 ──
    for p in [1, 5, 10, 20, 60]:
        df[f"mom{p}"] = grp[COL_CLOSE].pct_change(p) * 100
    df["mom_ratio_5_20"] = (df["mom5"] / df["mom20"].replace(0, np.nan)).fillna(0)
    df["mom_accel"] = df["mom5"].fillna(0) - df["mom20"].fillna(0)

    # upside_downside_ratio
    tmp_up = grp["_ret"].rolling(20, min_periods=2).apply(
        lambda x: np.mean(x[x>0]) if np.any(x>0) else 0, raw=True).values
    tmp_dn = grp["_ret"].rolling(20, min_periods=2).apply(
        lambda x: abs(np.mean(x[x<0])) if np.any(x<0) else 1, raw=True).values
    df["upside_downside_ratio"] = np.where(tmp_dn > 0.01, tmp_up / tmp_dn, 0)

    # ── 均线 ──
    for p in [5, 10, 20, 60, 120, 250]:
        ma = grp[COL_CLOSE].transform(lambda x, pp=p: x.rolling(pp, min_periods=1).mean())
        df[f"_ma{p}"] = ma
        df[f"ma{p}_dev"] = (df[COL_CLOSE] / ma - 1) * 100
    df["ma_cross"] = (df["_ma5"] > df["_ma20"]).astype(float)
    # ma_arrangement: -3~+3 → +2 → clip 0-4 (match old per-stock logic)
    _arr = ((df["_ma5"] > df["_ma10"]).astype(int) - (df["_ma5"] < df["_ma10"]).astype(int)
          + (df["_ma10"] > df["_ma20"]).astype(int) - (df["_ma10"] < df["_ma20"]).astype(int)
          + (df["_ma20"] > df["_ma60"]).astype(int) - (df["_ma20"] < df["_ma60"]).astype(int))
    df["ma_arrangement"] = (_arr + 2).clip(0, 4)
    df["ma_gap"] = (df["_ma5"] - df["_ma20"]) / df[COL_CLOSE] * 100

    # ── 成交量/额 ──
    mv5 = grp[COL_VOL].transform(lambda x: x.rolling(5, min_periods=1).mean())
    mv20 = grp[COL_VOL].transform(lambda x: x.rolling(20, min_periods=1).mean())
    ma5 = grp[COL_AMOUNT].transform(lambda x: x.rolling(5, min_periods=1).mean())
    ma20 = grp[COL_AMOUNT].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["vol_ratio"]  = (df[COL_VOL] / mv20.replace(0, np.nan)).fillna(0)
    df["vol_ma5_ratio"] = (df[COL_VOL] / mv5.replace(0, np.nan)).fillna(0)
    df["vol_shock"] = (grp[COL_VOL].pct_change() + 1).fillna(1)
    df["volume_oscillator"] = ((mv5 - mv20) / mv20.replace(0, np.nan)).fillna(0)
    df["amount_ratio"] = (df[COL_AMOUNT] / ma20.replace(0, np.nan)).fillna(0)
    df["amount_trend"] = (ma5 / ma20.replace(0, np.nan)).fillna(0)
    df["turnover_ratio_5d_20d"] = (mv5 / mv20.replace(0, np.nan)).fillna(0)
    df["turnover_std_20d"] = grp[COL_VOL].transform(
        lambda x: x.rolling(20, min_periods=5).std()
                  / x.rolling(20, min_periods=1).mean().replace(0, np.nan)).fillna(0)

    # ── 波动率 ──
    prev_c = grp[COL_CLOSE].shift(1)
    tr = np.maximum(np.maximum(
        df[COL_HIGH].values - df[COL_LOW].values,
        np.abs(df[COL_HIGH].values - prev_c.values)),
        np.abs(df[COL_LOW].values - prev_c.values))
    df["_tr"] = tr
    atr = grp["_tr"].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["atr_pct"] = (atr / df[COL_CLOSE] * 100).fillna(0)
    df["vol_adjusted_mom20"] = (df["mom20"] / df["atr_pct"].replace(0, np.nan)).fillna(0)

    cr_ = df[COL_HIGH].values - df[COL_LOW].values
    df["close_position"] = np.where(cr_ > 0, (df[COL_CLOSE].values - df[COL_LOW].values) / cr_, 0.5)
    df["daily_range"] = pd.Series(cr_ / df[COL_CLOSE].values * 100, index=df.index).fillna(0)

    r20_mu = grp["_ret"].rolling(20, min_periods=5).mean().values
    r20_sd = grp["_ret"].rolling(20, min_periods=5).std().values
    df["sharpe_20d"] = np.where(r20_sd > 1e-10, r20_mu / r20_sd * np.sqrt(252), 0)
    df["max_drawdown_20d"] = grp[COL_CLOSE].rolling(20, min_periods=5).apply(
        lambda x: np.min(x / np.maximum.accumulate(x) - 1) * 100, raw=True).values
    df["skewness_20d"] = grp["_ret"].rolling(20, min_periods=5).skew().values
    df["kurtosis_20d"] = grp["_ret"].rolling(20, min_periods=5).kurt().values
    df["downside_vol"] = grp["_ret"].rolling(20, min_periods=5).apply(
        lambda x: np.std(x[x<0], ddof=1) if np.sum(x<0) > 1 else 0, raw=True).values

    # ── 价格形态 ──
    # gap_pct = (open / prev_close - 1) * 100 (prev_close 是昨收, 不是昨开)
    prev_close = grp[COL_CLOSE].shift(1)
    df["gap_pct"] = (df[COL_OPEN] / prev_close - 1).fillna(0) * 100

    std20 = grp[COL_CLOSE].transform(lambda x: x.rolling(20, min_periods=5).std())
    ma20bb = grp[COL_CLOSE].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["bb_pct_b"] = np.where(std20 > 0,
        (df[COL_CLOSE].values - (ma20bb - 2*std20)) / (4*std20) * 100, 50)
    df["bb_width"] = np.where(ma20bb > 0, 2*std20 / ma20bb * 100, 0)

    lo20 = grp[COL_CLOSE].transform(lambda x: x.rolling(20, min_periods=1).min())
    hi20 = grp[COL_CLOSE].transform(lambda x: x.rolling(20, min_periods=1).max())
    df["price_position_20d"] = np.where(hi20 > lo20,
        (df[COL_CLOSE].values - lo20) / (hi20 - lo20) * 100, 50)
    lo60 = grp[COL_CLOSE].transform(lambda x: x.rolling(60, min_periods=1).min())
    hi60 = grp[COL_CLOSE].transform(lambda x: x.rolling(60, min_periods=1).max())
    df["price_position_60d"] = np.where(hi60 > lo60,
        (df[COL_CLOSE].values - lo60) / (hi60 - lo60) * 100, 50)

    # VWAP (rolling sum ratio)
    cum_pv = grp.apply(lambda g: (g[COL_CLOSE]*g[COL_AMOUNT]).rolling(20, min_periods=1).sum())
    cum_v = grp[COL_AMOUNT].transform(lambda x: x.rolling(20, min_periods=1).sum())
    cum_pv = cum_pv.reset_index(level=0, drop=True)
    vwap = cum_pv / cum_v.replace(0, np.nan)
    df["vwap_deviation"] = ((df[COL_CLOSE] / vwap - 1) * 100).fillna(0)

    # ── 统计特征（每只股票一次，广播到所有行）──
    def _ss(sdf):
        out = dict.fromkeys(["autocorr_1d","autocorr_2d","autocorr_5d",
                             "variance_ratio_5_1","hurst_exponent",
                             "runs_ratio","gain_loss_consistency"], 0.0)
        r_ = sdf["_ret"].values[1:]
        if len(r_) >= 30:
            if np.var(r_) > 1e-10:
                out["autocorr_1d"] = float(np.corrcoef(r_[:-1],r_[1:])[0,1]) if len(r_)>2 else 0
                out["autocorr_2d"] = float(np.corrcoef(r_[:-2],r_[2:])[0,1]) if len(r_)>3 else 0
                out["autocorr_5d"] = float(np.corrcoef(r_[:-5],r_[5:])[0,1]) if len(r_)>6 else 0
            v1 = np.var(r_, ddof=1)
            r5 = np.array([np.sum(r_[max(0,t-4):t+1]) for t in range(5,len(r_))])
            v5 = np.var(r5, ddof=1) if len(r5)>1 else 0
            out["variance_ratio_5_1"] = v5/(5*v1) if v1>0 else 1.0
            out["hurst_exponent"] = _compute_hurst(r_)
            med = np.median(r_); ab = r_ > med
            runs = 1 + np.sum(ab[1:] != ab[:-1])
            exp_r = 1 + 2*np.sum(ab)*np.sum(~ab)/len(ab)
            out["runs_ratio"] = runs/exp_r if exp_r>0 else 1.0
            g = r_[r_>0]; l_ = r_[r_<0]
            out["gain_loss_consistency"] = (np.std(g) if len(g)>0 else 0)/max(1, np.std(l_) if len(l_)>0 else 1)
        return pd.DataFrame(out, index=sdf.index)
    stat_df = df.groupby(COL_TS_CODE, group_keys=False).apply(_ss)
    for c in stat_df.columns:
        df[c] = stat_df[c].fillna(0)

    # ── 流动性 ──
    abr = df["_ret"].abs().values / 100
    amt = df[COL_AMOUNT].values
    df["amihud_illiq_20d"] = np.where(amt > 0, abr/amt*1e6, 0)
    df["amihud_illiq_20d"] = grp["amihud_illiq_20d"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    df["price_impact"] = np.where(amt > 0, abr/amt, 0)
    df["price_impact"] = grp["price_impact"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    df["zero_return_days_20d"] = grp["_ret"].rolling(20, min_periods=1).apply(
        lambda x: np.sum(np.abs(x) < 0.01), raw=True).values
    df["_hl_log"] = np.log(df[COL_HIGH].values / df[COL_LOW].values).clip(0)
    df["high_low_ratio_20d"] = grp["_hl_log"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()).fillna(0)

    # ── 逐股时序特征（一次 apply，2992 次调用）──
    logger.info("   逐股时序特征...")
    seq_df = df.groupby(COL_TS_CODE, group_keys=False).apply(_compute_stock_sequential)
    for c in seq_df.columns:
        df[c] = seq_df[c].fillna(0)

    # ── 行业占位 + 交叉特征 ──
    for c in ["excess_return_l1","excess_return_l2","excess_return_l3",
              "sector_persistence","L3_L2_divergence","L2_L1_divergence","industry_cascade"]:
        df[c] = 0.0
    # 全量池扩池 (2026-08-05): 是否申万映射（0=未分类/东财兜底，行业特征为占位）
    def _sw_flag(c: str) -> float:
        if not stock_mapping:
            return 1.0  # 旧行为: 无映射时默认全部为申万
        info = stock_mapping.get(c, {})
        flag = info.get("is_sw", "1" if info.get("l1_code", "") else "0")
        return 1.0 if str(flag) in ("1", "True", "true") else 0.0
    df["has_sw_mapping"] = df[COL_TS_CODE].map(_sw_flag)
    df["vol_amplitude_ratio"] = (df["vol_ratio"]/df["atr_pct"].replace(0,np.nan)).fillna(0)
    df["vol_conviction"] = (df["vol_ratio"]*(df[COL_CLOSE]/df["_ma20"]-1)*100).fillna(0)
    df["momentum_volume"] = (df["mom5"]*df["vol_ratio"]).fillna(0)
    df["price_volume_diverg"] = ((df["streak"]>=2)&(df["vol_ratio"]<0.8)).astype(float)
    df["bb_streak"] = (df["bb_pct_b"]*df["streak"]).fillna(0)
    df["gap_momentum"] = (df["gap_pct"]*df["mom5"]).fillna(0)
    df["position_conviction"] = (df["price_position_20d"]*df["vol_ratio"]/100).fillna(0)
    df["vol_drawdown"] = (df["vol_ratio"]*df["max_drawdown_20d"]).fillna(0)

    # ── 截面排名/外部/Beta 占位 ──
    for col in (CROSS_SECTIONAL_FEATURES + MARKET_RELATIVE_FEATURES
                + EXTERNAL_FEATURES + COMPOSITE_FEATURES):
        if col not in df.columns:
            df[col] = 0.0

    # ── 过滤前 20 天 + 数值类型 ──
    df = df.groupby(COL_TS_CODE, group_keys=False).apply(lambda g: g.iloc[20:].copy())
    df["close"] = df[COL_CLOSE].values
    for c in df.columns:
        if c not in (COL_TS_CODE, COL_TRADE_DATE):
            try: df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception: pass
    df = df.fillna(0)
    drop_c = [c for c in df.columns if c.startswith("_")]
    if drop_c: df.drop(columns=drop_c, inplace=True)
    # 去掉原始输入列（旧版不包含）
    for c in [COL_OPEN, COL_HIGH, COL_LOW, COL_VOL, COL_AMOUNT, "pre_close", "pct_chg"]:
        if c in df.columns: df.drop(columns=[c], inplace=True)

    logger.info("  → %d 行 × %d 列", len(df), len(df.columns))
    return df


# ═══════════════════════════════════════════════════════════════
# 流水线：全量特征构建
# ═══════════════════════════════════════════════════════════════

def build_full_feature_matrix(
    stock_daily_df: pd.DataFrame,
    l1_daily: pd.DataFrame | None = None,
    l2_daily: pd.DataFrame | None = None,
    l3_daily: pd.DataFrame | None = None,
    stock_mapping: dict | None = None,
    persistence_scores: dict | None = None,
    market_index_df: pd.DataFrame | None = None,
    margin_df: pd.DataFrame | None = None,
    moneyflow_df: pd.DataFrame | None = None,
    fundamental_df: pd.DataFrame | None = None,
    financial_quality_df: pd.DataFrame | None = None,
    multi_day: bool = False,
) -> pd.DataFrame:
    """执行全量特征构建流水线。

    阶段：
      1. build_feature_matrix — OHLCV 基础特征
      2. add_industry_hierarchy_features — L1/L2/L3 行业特征
      3. add_cross_sectional_features — 截面排名
      4. add_market_relative_features — Beta/市场相对
      5. add_external_data_features — 外部数据 join
      6. add_composite_features — 复合因子
    """
    logger.info("阶段 1/6: 构建 OHLCV 基础特征...")
    df = build_feature_matrix_vectorized(stock_daily_df, l1_daily, stock_mapping, persistence_scores, multi_day)
    if df.empty:
        return df
    logger.info("  → %d 行 × %d 列", len(df), len(df.columns))

    logger.info("阶段 2/6: 添加行业层级特征...")
    df = add_industry_hierarchy_features(df, stock_mapping, l1_daily, l2_daily, l3_daily)
    logger.info("  → %d 列", len(df.columns))

    logger.info("阶段 3/6: 添加截面排名特征...")
    df = add_cross_sectional_features(df)
    logger.info("  → %d 列", len(df.columns))

    logger.info("阶段 4/6: 添加 Beta/市场相对特征...")
    df = add_market_relative_features(df, stock_daily_df, market_index_df)
    logger.info("  → %d 列", len(df.columns))

    logger.info("阶段 5/6: 添加外部数据特征...")
    df = add_external_data_features(df, margin_df, moneyflow_df, fundamental_df, financial_quality_df, stock_mapping)
    logger.info("  → %d 列", len(df.columns))

    logger.info("阶段 6/6: 添加复合因子...")
    df = add_composite_features(df)
    logger.info("  → %d 列", len(df.columns))

    logger.info("阶段 7/7: 添加 V5 扩展特征...")
    df = add_v5_extension_features(df, stock_daily_df, stock_mapping)
    logger.info("  → %d 列", len(df.columns))

    # 统一填充
    for c in ALL_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════

def train_model(feature_df: pd.DataFrame, group: str, forward_days: int = 20):
    """训练指定分组的 LightGBM 模型。"""
    cfg = GROUP_CONFIG.get(group)
    if cfg is None:
        logger.error("未知分组: %s", group)
        return None, None

    feats = [c for c in cfg["features"] if c in feature_df.columns]
    if len(feats) < 3 or feature_df.empty or len(feature_df) < 100:
        logger.warning("分组 %s: 训练数据不足", group)
        return None, None

    # 构建 label: forward N-day return
    labels = []
    for code in feature_df[COL_TS_CODE].unique():
        sdf = feature_df[feature_df[COL_TS_CODE] == code].sort_values(COL_TRADE_DATE)
        closes = sdf["close"].values
        for i in range(len(sdf) - forward_days):
            if closes[i] > 0:
                labels.append({
                    COL_TS_CODE: code,
                    COL_TRADE_DATE: sdf[COL_TRADE_DATE].iloc[i],
                    "label": (closes[i + forward_days] / closes[i] - 1) * 100,
                })
    if not labels:
        return None, None

    label_df = pd.DataFrame(labels)
    train_df = feature_df.merge(label_df, on=[COL_TS_CODE, COL_TRADE_DATE], how="inner")
    if "close" in train_df.columns:
        train_df = train_df.drop(columns=["close"])
    if len(train_df) < 100:
        return None, None

    X = train_df[feats].fillna(0)
    y = train_df["label"]

    model = lgb.LGBMRegressor(**cfg["params"])
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": feats, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("分组 %s 训练完成: %d 样本, %d 特征, Top=%s",
                 group, len(train_df), len(feats),
                 ", ".join(importance["feature"].head(3).tolist()))

    return model, importance


# ═══════════════════════════════════════════════════════════════
# Walk-Forward 训练（时间序列滚动交叉验证）
# ═══════════════════════════════════════════════════════════════

def compute_forward_labels(
    feature_df: pd.DataFrame,
    forward_days: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """为特征数据集预计算前向收益标签。

    对每只股票的每个交易日，计算未来 N 日收益。
    最后 forward_days 行的标签为 NaN（数据不足）。

    Returns:
        带 fwd_return 列的 DataFrame
    """
    if feature_df is None or feature_df.empty:
        return feature_df

    df = feature_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).copy()
    codes = df[COL_TS_CODE].unique()
    all_labels = []

    for code in codes:
        idx = df[COL_TS_CODE] == code
        sdf = df[idx].sort_values(COL_TRADE_DATE)
        prices = sdf[price_col].values
        dates = sdf[COL_TRADE_DATE].values
        n = len(prices)

        for i in range(n - forward_days):
            if prices[i] > 0:
                fwd = (prices[i + forward_days] / prices[i] - 1) * 100
                all_labels.append({
                    COL_TS_CODE: code,
                    COL_TRADE_DATE: dates[i],
                    "fwd_return": round(fwd, 4),
                })

    if not all_labels:
        return df

    label_df = pd.DataFrame(all_labels)
    result = df.merge(label_df, on=[COL_TS_CODE, COL_TRADE_DATE], how="left")
    return result


def _get_group_config_override(group: str, n_samples: int) -> dict:
    """根据样本量动态调整 LightGBM 参数。"""
    cfg = dict(GROUP_CONFIG[group]["params"])
    # 样本量少时减少复杂度
    if n_samples < 5000:
        cfg["num_leaves"] = min(cfg.get("num_leaves", 31), 7)
        cfg["n_estimators"] = min(cfg.get("n_estimators", 200), 80)
        cfg["min_child_samples"] = 10
    elif n_samples < 20000:
        cfg["num_leaves"] = min(cfg.get("num_leaves", 31), 15)
        cfg["n_estimators"] = min(cfg.get("n_estimators", 200), 120)
    return cfg


def train_model_on_window(
    train_df: pd.DataFrame,
    group: str,
    feats: list[str],
) -> tuple:
    """在给定的训练集上训练一个模型。

    Returns:
        (model, importance_df) 或 (None, None)
    """
    if train_df.empty or len(train_df) < 200:
        return None, None

    X = train_df[feats].fillna(0)
    y = train_df["fwd_return"].values

    if len(X) < 200:
        return None, None

    cfg = _get_group_config_override(group, len(X))
    model = lgb.LGBMRegressor(**cfg)
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": feats,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, importance


def walk_forward_train(
    feature_df: pd.DataFrame,
    group: str,
    initial_train_days: int = 150,
    val_days: int = 25,
    step_days: int = 25,
    forward_days: int = 20,
    min_train_samples: int = 500,
    skip_validation: bool = False,
) -> dict:
    """Walk-Forward 时间序列滚动训练。

    方法：
      1. 时间顺序分割数据
      2. 每折：前 150 天训练 → 后 25 天验证（向前滚动）
      3. 每次验证用真实前向收益评估（非 close 代理）
      4. 所有折完成后，用全量数据训练最终模型

    Args:
        feature_df: 带 close 列的 V4 特征数据集
        group: "micro" 或 "mid"
        initial_train_days: 初始训练窗口（交易日）
        val_days: 每折验证窗口大小
        step_days: 滚动步长
        forward_days: 前向收益预测天数
        min_train_samples: 最少训练样本数
        skip_validation: 为 True 时跳过 98 折验证循环，仅拟合全量最终模型
                          （每日增量训练用，验证指标由每周六全量运行刷新）

    Returns:
        dict:
          folds: [{train_end, val_start, val_end, spearman_r,
                   top20_median, bot20_median, spread, n_train, n_val}]
          final_model: 全量数据训练的最终模型
          final_importance: 全量特征重要性
          feature_rank_stability: 特征排名在不同折之间的稳定性
          overall: 总体性能汇总
    """
    cfg = GROUP_CONFIG.get(group)
    if cfg is None:
        logger.error("未知分组: %s", group)
        return {"error": f"未知分组: {group}"}

    # ── 1. 筛选分组 ──
    if "group" in feature_df.columns:
        gdf = feature_df[feature_df["group"] == group].copy()
    else:
        gdf = feature_df.copy()

    if gdf.empty:
        logger.warning("分组 %s: 无数据", group)
        return {"error": "无数据"}

    # ── 2. 计算前向收益标签 ──
    logger.info("Walk-Forward %s: 计算前向收益标签 (N=%d)...", group, forward_days)
    gdf = compute_forward_labels(gdf, forward_days=forward_days)

    # 丢弃没有标签的行（最后 forward_days 天）
    gdf = gdf.dropna(subset=["fwd_return"])
    if gdf.empty:
        logger.error("分组 %s: 前向标签全部为空", group)
        return {"error": "无标签"}

    # ── 3. 获取可用特征 ──
    feats = [c for c in cfg["features"] if c in gdf.columns]
    if len(feats) < 3:
        logger.warning("分组 %s: 特征不足", group)
        return {"error": "特征不足"}

    logger.info("  %d 样本, %d 特征", len(gdf), len(feats))

    # ── 3.5 快速刷新路径（skip_validation=True）──
    # 每日增量训练用：跳过 98 折验证，直接拟合全量最终模型。
    # 最终模型与完整 walk-forward 的最终模型等价（同为「截止最新日期的全量拟合」），
    # 但报告中的 98 折验证指标保留上次全量运行的结果（每周六 cron 刷新）。
    if skip_validation:
        logger.info("=" * 60)
        logger.info("快速刷新 %s: 跳过 98 折验证，直接拟合最终模型 (N=%d)", group, len(gdf))
        logger.info("=" * 60)
        final_model, final_imp = train_model_on_window(gdf, group, feats)
        result = {
            "group": group,
            "folds": [],
            "fold_models": [],
            "overall": {
                "n_folds": 0,
                "n_features": len(feats),
                "fast_refresh": True,
                "initial_train_days": initial_train_days,
                "val_days": val_days,
                "step_days": step_days,
                "forward_days": forward_days,
            },
            "final_model": final_model,
            "final_importance": final_imp,
            "feature_rank_stability": {},
            "feature_list": feats,
        }
        if final_imp is not None:
            logger.info("快速刷新 %s: 最终模型 Top-5 特征:", group)
            for _, r in final_imp.head(5).iterrows():
                logger.info("  %-30s %4d", r["feature"], int(r["importance"]))
        return result

    # ── 4. 时间序列分割 ──
    all_dates = sorted(gdf[COL_TRADE_DATE].unique())
    n_dates = len(all_dates)

    if n_dates < initial_train_days + val_days + forward_days:
        logger.warning("分组 %s: 数据天数不足 (%d < %d)",
                       group, n_dates, initial_train_days + val_days + forward_days)
        return {"error": "天数不足"}

    # ── 5. Walk-Forward ──
    logger.info("=" * 60)
    logger.info("Walk-Forward %s: %d 个交易日", group, n_dates)
    logger.info("  初始训练: %d 天, 验证: %d 天, 步长: %d 天",
                initial_train_days, val_days, step_days)
    logger.info("=" * 60)

    folds = []
    fold_models = []
    all_importances = []
    train_end = initial_train_days
    fold_idx = 0

    while train_end + val_days <= n_dates - forward_days:
        fold_idx += 1
        train_dates = all_dates[:train_end]
        val_dates = all_dates[train_end:train_end + val_days]

        train_df = gdf[gdf[COL_TRADE_DATE].isin(train_dates)]
        val_df = gdf[gdf[COL_TRADE_DATE].isin(val_dates)]

        if len(train_df) < min_train_samples or val_df.empty:
            train_end += step_days
            continue

        # 训练
        model, imp = train_model_on_window(train_df, group, feats)
        if model is None:
            train_end += step_days
            continue

        # 预测
        X_val = val_df[feats].fillna(0)
        preds = model.predict(X_val)
        actuals = val_df["fwd_return"].values

        # 评估
        from scipy.stats import spearmanr
        r_valid = ~(np.isnan(preds) | np.isnan(actuals))
        r, r_p = spearmanr(preds[r_valid], actuals[r_valid]) if r_valid.sum() > 3 else (0.0, 1.0)

        # Top/Bottom 20 前向收益
        val_result = val_df.copy()
        val_result["pred"] = preds
        top20 = val_result.nlargest(20, "pred")["fwd_return"].median() if len(val_result) >= 20 else 0
        bot20 = val_result.nsmallest(20, "pred")["fwd_return"].median() if len(val_result) >= 20 else 0
        spread = top20 - bot20

        fold_info = {
            "fold": fold_idx,
            "train_end": all_dates[train_end - 1],
            "val_start": val_dates[0],
            "val_end": val_dates[-1],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "train_dates": f"{all_dates[0]}~{train_dates[-1]}",
            "val_dates": f"{val_dates[0]}~{val_dates[-1]}",
            "spearman_r": round(r, 4),
            "spearman_p": round(r_p, 4),
            "top20_fwd_return": round(top20, 2),
            "bot20_fwd_return": round(bot20, 2),
            "spread": round(spread, 2),
        }
        folds.append(fold_info)

        if imp is not None:
            all_importances.append(imp.set_index("feature")["importance"])

        fold_models.append(model)

        logger.info(
            "  折%2d: 训练~%s | 验证 %s~%s | "
            "r=%+.4f (p=%.3f) | spread=%.2f%% | train=%d val=%d",
            fold_idx, all_dates[train_end - 1],
            val_dates[0], val_dates[-1],
            r, r_p, spread,
            len(train_df), len(val_df),
        )

        train_end += step_days

    if not folds:
        logger.warning("分组 %s: 无有效折", group)
        return {"error": "无有效折"}

    # ── 6. 特征排名稳定性 ──
    rank_stability = {}
    if len(all_importances) >= 2:
        imp_df = pd.DataFrame(all_importances).fillna(0)
        rank_df = imp_df.rank(axis=1, ascending=False)
        # 特征排名的标准差 — 越小越稳定
        rank_std = rank_df.std(axis=1).sort_values()
        for feat in rank_std.head(10).index:
            rank_stability[feat] = round(rank_std[feat], 2)

    # ── 7. 全量数据训练最终模型 ──
    logger.info("=" * 60)
    logger.info("训练最终模型（全量数据）...")
    logger.info("=" * 60)

    final_model, final_imp = train_model_on_window(gdf, group, feats)

    # ── 8. 汇总 ──
    r_values = [f["spearman_r"] for f in folds]
    spreads = [f["spread"] for f in folds]
    positive_folds = sum(1 for r in r_values if r > 0)

    overall = {
        "n_folds": len(folds),
        "mean_spearman_r": round(float(np.mean(r_values)), 4),
        "median_spearman_r": round(float(np.median(r_values)), 4),
        "std_spearman_r": round(float(np.std(r_values, ddof=1)), 4),
        "positive_folds": positive_folds,
        "positive_ratio": round(positive_folds / len(folds), 3) if folds else 0,
        "best_fold_spearman": round(float(np.max(r_values)), 4),
        "worst_fold_spearman": round(float(np.min(r_values)), 4),
        "mean_spread": round(float(np.mean(spreads)), 2),
        "max_spread": round(float(np.max(spreads)), 2),
        "min_spread": round(float(np.min(spreads)), 2),
        "n_features": len(feats),
        "initial_train_days": initial_train_days,
        "val_days": val_days,
        "step_days": step_days,
        "forward_days": forward_days,
    }

    result = {
        "group": group,
        "folds": folds,
        "fold_models": fold_models,
        "overall": overall,
        "final_model": final_model,
        "final_importance": final_imp,
        "feature_rank_stability": rank_stability,
        "feature_list": feats,
    }

    logger.info("=" * 60)
    logger.info("Walk-Forward %s 汇总", group)
    logger.info("  有效折数: %d", len(folds))
    logger.info("  平均 Spearman r: %.4f (正比例: %.1f%%)",
                overall["mean_spearman_r"], overall["positive_ratio"] * 100)
    logger.info("  平均 Top-Bottom Spread: %.2f%%", overall["mean_spread"])
    logger.info("  最佳折: r=%.4f", overall["best_fold_spearman"])
    logger.info("  最差折: r=%.4f", overall["worst_fold_spearman"])
    logger.info("  特征数: %d", len(feats))
    logger.info("=" * 60)

    if final_imp is not None:
        logger.info("最终模型 Top-10 特征:")
        for _, r in final_imp.head(10).iterrows():
            logger.info("  %-30s %4d", r["feature"], int(r["importance"]))

    return result


def save_walk_forward_report(results: dict, output_path: str):
    """将 walk-forward 结果保存为可读报告。

    Args:
        results: {"micro": {...}, "mid": {...}} 或单个 result dict
        output_path: 输出路径
    """
    import json
    from datetime import datetime

    if not results:
        return

    # 兼容单个 result
    if "group" in results and "folds" in results:
        results = {"single": results}

    serializable = {}
    for grp, r in results.items():
        if not r or "error" in r:
            serializable[grp] = {"error": r.get("error", "unknown") if r else "empty"}
            continue

        imp = {}
        if r.get("final_importance") is not None:
            for _, row in r["final_importance"].iterrows():
                imp[row["feature"]] = int(row["importance"])

        # 确保所有值为 Python 原生类型
        def _to_native(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.ndarray,)):
                return v.tolist()
            if isinstance(v, pd.Series):
                return v.to_dict()
            return v

        stability = {str(k): _to_native(v) for k, v in r.get("feature_rank_stability", {}).items()}

        serializable[grp] = {
            "group": r.get("group", grp),
            "overall": {k: _to_native(v) for k, v in r.get("overall", {}).items()},
            "folds": [
                {k: _to_native(v) for k, v in f.items() if k != "train_dates"}
                for f in (r.get("folds") or [])
            ],
            "feature_rank_stability": stability,
            "final_feature_importance": {k: int(v) for k, v in imp.items()},
            "n_features": len(r.get("feature_list", [])),
        }

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": serializable,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Walk-forward 报告已保存: %s", output_path)


def save_model(model, group: str):
    path = MODEL_PATHS.get(group)
    if path and model:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("模型已保存: %s (%s)", path, group)


def load_model(group: str):
    path = MODEL_PATHS.get(group)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# 推理
# ═══════════════════════════════════════════════════════════════

def assign_group(avg_amount: float, amount_thresholds: tuple[float, float]) -> str:
    if avg_amount <= amount_thresholds[0]:
        return "micro"
    elif avg_amount <= amount_thresholds[1]:
        return "mid"
    else:
        return "large"


def _compute_amount_thresholds(stock_daily_df: pd.DataFrame) -> tuple[float, float]:
    amounts = stock_daily_df.groupby(COL_TS_CODE)["amount"].mean()
    return (amounts.quantile(1/3), amounts.quantile(2/3))


FEATURE_CACHE_PATH = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")


def rerank_with_ml(
    stock_picks: list[dict],
    stock_daily_df: pd.DataFrame,
    stock_mapping: dict | None = None,
    industry_daily_df: pd.DataFrame | None = None,
    persistence_scores: dict | None = None,
    models: dict | None = None,
    # ── 全量特征推理参数（传入则使用 build_full_feature_matrix）──
    l2_daily: pd.DataFrame | None = None,
    l3_daily: pd.DataFrame | None = None,
    market_index_df: pd.DataFrame | None = None,
    margin_df: pd.DataFrame | None = None,
    moneyflow_df: pd.DataFrame | None = None,
    fundamental_df: pd.DataFrame | None = None,
    financial_quality_df: pd.DataFrame | None = None,
) -> list[dict]:
    """三分组 ML 重排 — 全量特征在线推理（优先），V4 缓存推理（回退）。

    全量在线路径（首选，当所有数据源传入时）:
      - 对精选股用 build_full_feature_matrix 在线构建 118 特征
      - 分组加载 LightGBM 模型 -> predict

    V4 缓存路径（次选，当全量数据未传入或缓存最新日期匹配当前日期时）:
      - 加载 feature_matrix_v4.parquet 缓存
      - 对每只精选股取缓存中当日特征行
      - 按三分组加载模型预测

    V3 路径（最终回退）:
      - 当缓存不可用时在线构建基本 OHLCV 特征
    """
    if not stock_picks:
        return stock_picks

    # ── 全量在线路径（精度最高，所有数据源就绪时使用）──
    has_full_data = all([
        stock_mapping is not None,
        industry_daily_df is not None,
        l2_daily is not None,
    ])
    if has_full_data:
        try:
            return _rerank_full_online(
                stock_picks, stock_daily_df, industry_daily_df,
                l2_daily, l3_daily, stock_mapping, persistence_scores,
                market_index_df, margin_df, moneyflow_df,
                fundamental_df, financial_quality_df, models,
            )
        except Exception as e:
            logger.warning("全量在线推理失败: %s，回退到缓存/精简路径", e)

    # ── V4 缓存路径（精度折中，数据陈旧时静默降级）──
    if os.path.exists(FEATURE_CACHE_PATH):
        try:
            return _rerank_via_cache(stock_picks, stock_daily_df, models)
        except Exception as e:
            logger.warning("V4 缓存推理失败: %s，回退到在线构建", e)

    # ── 回退 V3 精简在线路径 ──
    return _rerank_v3_online(stock_picks, stock_daily_df, stock_mapping,
                              industry_daily_df, persistence_scores, models)


def _rerank_via_cache(
    stock_picks: list[dict],
    stock_daily_df: pd.DataFrame,
    models: dict | None = None,
) -> list[dict]:
    """使用 V4 特征缓存进行推理（118 特征全量）。

    若缓存日期与当前数据日期不匹配（缓存在上次训练时产生，
    不会自动更新），则回退到在线构建特征以确保每日评分更新。
    """
    cache = pd.read_parquet(FEATURE_CACHE_PATH)
    cache = cache.sort_values([COL_TS_CODE, COL_TRADE_DATE])

    # ── 日期新鲜度检查 ──
    latest_data_date = stock_daily_df[COL_TRADE_DATE].max() if stock_daily_df is not None else None
    cache_dates = cache[COL_TRADE_DATE].unique()
    cache_latest_date = sorted(cache_dates, reverse=True)[0] if len(cache_dates) > 0 else None

    if latest_data_date and cache_latest_date and latest_data_date != cache_latest_date:
        logger.info("V4 缓存最新日期 %s ≠ 当前 %s，回退到在线构建特征",
                     cache_latest_date, latest_data_date)
        return _rerank_v3_online(stock_picks, stock_daily_df,
                                  stock_mapping=None, industry_daily_df=None,
                                  persistence_scores=None, models=models)

    # 获取每只股票在缓存中最新的特征行
    latest_features = cache.groupby(COL_TS_CODE).last().reset_index()

    # 三分组
    thresholds = _compute_amount_thresholds(stock_daily_df)
    codes_list = []
    for p in stock_picks:
        code = p["ts_code"]
        sd = stock_daily_df[stock_daily_df[COL_TS_CODE] == code]
        avg_amt = sd["amount"].mean() if not sd.empty else 0
        grp = assign_group(avg_amt, thresholds)
        p["_group"] = grp
        codes_list.append(code)

    # 按分组预测
    feature_cols = [c for c in ALL_FEATURES if c in latest_features.columns]
    for grp in ["micro", "mid"]:
        grp_picks = [p for p in stock_picks if p.get("_group") == grp and p["ts_code"] in latest_features[COL_TS_CODE].values]
        if not grp_picks:
            continue
        grp_codes = [p["ts_code"] for p in grp_picks]
        grp_feats = latest_features[latest_features[COL_TS_CODE].isin(grp_codes)]

        model = models.get(grp) if models else load_model(grp)
        if model is None or grp_feats.empty:
            continue

        # 仅使用模型训练时见过的特征
        available = [c for c in feature_cols if c in grp_feats.columns]
        if not available:
            continue

        X = grp_feats[available].fillna(0)
        preds = model.predict(X)

        code_to_pred = dict(zip(grp_feats[COL_TS_CODE].values, preds))
        for p in grp_picks:
            ms = code_to_pred.get(p["ts_code"])
            if ms is not None:
                p["ml_score"] = round(float(ms), 3)

    # ── 截面百分位归一化 (替代固定 raw*5+5，消除模型系统性偏差) ──
    all_ml = [p["ml_score"] for p in stock_picks if p.get("ml_score") is not None]
    if all_ml:
        min_ml, max_ml = min(all_ml), max(all_ml)
        rng = max_ml - min_ml
        for p in stock_picks:
            ms = p.get("ml_score")
            if ms is not None:
                if rng > 0:
                    pct = (ms - min_ml) / rng  # 0~1
                else:
                    pct = 0.5
                p["score"] = round(pct * 10, 2)

    # 排序: ML 评分在前，线性评分在后
    ml_scored = [p for p in stock_picks if p.get("ml_score") is not None]
    linear_only = [p for p in stock_picks if p.get("ml_score") is None]
    ml_scored.sort(key=lambda x: x["score"], reverse=True)

    # 清理临时字段
    for p in stock_picks:
        p.pop("_group", None)

    return ml_scored + linear_only


def _rerank_v3_online(
    stock_picks: list[dict],
    stock_daily_df: pd.DataFrame,
    stock_mapping: dict | None = None,
    industry_daily_df: pd.DataFrame | None = None,
    persistence_scores: dict | None = None,
    models: dict | None = None,
) -> list[dict]:
    """在线构建特征进行推理（基于当前 stock_daily_df，每日评分更新）。

    内部调用 build_feature_matrix 为精选股构建全量 OHLCV 衍生特征，
    取最新日期行 + 模型预测。
    """
    thresholds = _compute_amount_thresholds(stock_daily_df)
    stocks_info = {}
    for p in stock_picks:
        code = p["ts_code"]
        sd = stock_daily_df[stock_daily_df[COL_TS_CODE] == code]
        avg_amt = sd["amount"].mean() if not sd.empty else 0
        grp = assign_group(avg_amt, thresholds)
        stocks_info[code] = {"group": grp, "features": {}}

    for grp in ["micro", "mid"]:
        codes_in_grp = [c for c, info in stocks_info.items() if info["group"] == grp]
        if not codes_in_grp:
            continue
        target_df = stock_daily_df[stock_daily_df[COL_TS_CODE].isin(codes_in_grp)]
        if target_df.empty:
            continue
        feat_df = build_feature_matrix(
            target_df, industry_daily_df, stock_mapping, persistence_scores, multi_day=False,
        )
        model = models.get(grp) if models else load_model(grp)
        if model is None or feat_df.empty:
            continue
        # 使用模型实际训练过的特征子集
        try:
            model_feats = [c for c in model.booster_.feature_name() if c in feat_df.columns]
        except Exception:
            model_feats = [c for c in ALL_FEATURES if c in feat_df.columns]
        if not model_feats:
            continue

        # 每只股票只取最新日期的特征行（避免多日期多只股票时的 preds 索引错位）
        latest_idx = feat_df.groupby(COL_TS_CODE)[COL_TRADE_DATE].idxmax()
        latest_feat = feat_df.loc[latest_idx].copy()

        X = latest_feat[model_feats].fillna(0)
        preds = model.predict(X)

        for i, (_, row) in enumerate(latest_feat.iterrows()):
            code = row.get(COL_TS_CODE)
            if code and code in stocks_info:
                stocks_info[code]["features"]["ml_score"] = float(preds[i])

    ml_scored = [p for p in stock_picks if stocks_info.get(p["ts_code"], {}).get("features", {}).get("ml_score") is not None]
    linear_only = [p for p in stock_picks if stocks_info.get(p["ts_code"], {}).get("features", {}).get("ml_score") is None]

    for p in ml_scored:
        ms = stocks_info[p["ts_code"]]["features"]["ml_score"]
        p["ml_score"] = round(ms, 3)

    # ── 截面百分位归一化（与 V4 路径一致，消除系统性偏差）──
    all_ml = [p["ml_score"] for p in ml_scored if p.get("ml_score") is not None]
    if all_ml:
        min_ml, max_ml = min(all_ml), max(all_ml)
        rng = max_ml - min_ml
        for p in ml_scored:
            ms = p.get("ml_score")
            if ms is not None:
                if rng > 0:
                    pct = (ms - min_ml) / rng  # 0~1
                else:
                    pct = 0.5
                p["score"] = round(pct * 10, 2)

    ml_scored.sort(key=lambda x: x["score"], reverse=True)
    return ml_scored + linear_only


def _rerank_full_online(
    stock_picks: list[dict],
    stock_daily_df: pd.DataFrame,
    l1_daily: pd.DataFrame | None,
    l2_daily: pd.DataFrame | None,
    l3_daily: pd.DataFrame | None,
    stock_mapping: dict | None,
    persistence_scores: dict | None,
    market_index_df: pd.DataFrame | None,
    margin_df: pd.DataFrame | None,
    moneyflow_df: pd.DataFrame | None,
    fundamental_df: pd.DataFrame | None,
    financial_quality_df: pd.DataFrame | None,
    models: dict | None = None,
) -> list[dict]:
    """在线全量特征推理 — 用 build_full_feature_matrix（118 特征）为精选股构建当日特征。

    与 V4 缓存路径的区别：使用当前 stock_daily_df 实时构建特征，
    确保每日更新的行情数据反映到 ML 评分中。

    特征涵盖：动量(11) | 均线(8) | 量/额(10) | 波动率(10) | 价格形态(13)
    K线(9) | 行业(7) | 截面排名(11) | Beta/市场相对(6) | 统计(7)
    流动性(5) | 融资融券(3) | 资金流向(4) | 基本面(13) | 财务质量(6)
    复合因子(5) | 板块属性(1)
    """
    if not stock_picks:
        return stock_picks

    # 对每只精选股取日线数据
    picked_codes = set(p["ts_code"] for p in stock_picks)
    target_df = stock_daily_df[stock_daily_df[COL_TS_CODE].isin(picked_codes)].copy()
    if target_df.empty:
        logger.warning("全量在线推理: 精选股无日线数据")
        return stock_picks

    # 三分组
    thresholds = _compute_amount_thresholds(stock_daily_df)
    code_to_group = {}
    for p in stock_picks:
        sd = target_df[target_df[COL_TS_CODE] == p["ts_code"]]
        avg_amt = sd["amount"].mean() if not sd.empty else 0
        code_to_group[p["ts_code"]] = assign_group(avg_amt, thresholds)

    # 构建 118 特征全量矩阵（仅对精选股计算，含外部数据 join）
    feats = build_full_feature_matrix(
        stock_daily_df=target_df,
        l1_daily=l1_daily,
        l2_daily=l2_daily,
        l3_daily=l3_daily,
        stock_mapping=stock_mapping,
        persistence_scores=persistence_scores,
        market_index_df=market_index_df,
        margin_df=margin_df,
        moneyflow_df=moneyflow_df,
        fundamental_df=fundamental_df,
        financial_quality_df=financial_quality_df,
    )
    if feats.empty:
        logger.warning("全量在线推理: 特征矩阵为空")
        return stock_picks

    # 每只股票取最新特征行
    latest_idx = feats.groupby(COL_TS_CODE)[COL_TRADE_DATE].idxmax()
    latest_feat = feats.loc[latest_idx].copy()
    logger.info("全量在线特征: %d 只股票, %d 特征",
                 len(latest_feat), len(latest_feat.columns))

    # 按分组预测
    feature_cols = [c for c in ALL_FEATURES if c in latest_feat.columns]
    for grp in ["micro", "mid"]:
        grp_codes = [c for c in picked_codes if code_to_group.get(c) == grp and c in latest_feat[COL_TS_CODE].values]
        if not grp_codes:
            continue
        grp_feats = latest_feat[latest_feat[COL_TS_CODE].isin(grp_codes)]

        model = models.get(grp) if models else load_model(grp)
        if model is None or grp_feats.empty:
            continue

        available = [c for c in feature_cols if c in grp_feats.columns]
        if not available:
            continue

        X = grp_feats[available].fillna(0)
        preds = model.predict(X)

        code_to_pred = dict(zip(grp_feats[COL_TS_CODE].values, preds))
        for p in stock_picks:
            ms = code_to_pred.get(p["ts_code"])
            if ms is not None:
                p["ml_score"] = round(float(ms), 3)

    # ── 截面百分位归一化 ──
    all_ml = [p["ml_score"] for p in stock_picks if p.get("ml_score") is not None]
    if all_ml:
        min_ml, max_ml = min(all_ml), max(all_ml)
        rng = max_ml - min_ml
        for p in stock_picks:
            ms = p.get("ml_score")
            if ms is not None:
                p["score"] = round(((ms - min_ml) / rng) * 10, 2) if rng > 0 else 5.0

    # 排序: ML 评分在前
    ml_scored = [p for p in stock_picks if p.get("ml_score") is not None]
    linear_only = [p for p in stock_picks if p.get("ml_score") is None]
    ml_scored.sort(key=lambda x: x["score"], reverse=True)

    logger.info("全量在线推理: %d/20 只有 ML 评分", len(ml_scored))
    return ml_scored + linear_only
