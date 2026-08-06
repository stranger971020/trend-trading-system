#!/usr/bin/env python3
"""V6 Master Plan — 特征分段 (Feature Segmentation)。

废除「全市场统一回归」的单一目标模型，将特征矩阵按三套独立策略引擎分段：

  Engine         语义            特征池                           目标
  ─────────────  ─────────────  ───────────────────────────────  ──────────────
  momentum       趋势持续性      动量/均线乖离/板块强度/时序结构    捕捉持续性
  reversion      超跌反弹        偏离度/RSI背离/布林触底/超跌形态   捕捉超跌反弹
  breakout       变盘捕捉        波动压缩/量能异动/流动性枯竭      捕捉变盘突破

每个引擎独立训练自己的二分类 WinRate 模型（见 winrate_engine.py），
互不混合梯度。本模块只负责「输入掩码」——严格把每个引擎能看到的
特征列锁死。

历史背景 (2026-08-01, HERMES-20260801-001)：
V5 在单一 LGBMRegressor 上混合全部 306 特征 + 连续 fwd_return 目标，
导致「动量梯度 + 反转梯度 + 突破梯度」互相抵消，胜率 ~20%。V6 从输入
层面拆分梯度，让每个引擎只学自己的形态。

# ── Changelog ──
# 2026-08-01 Claude: HERMES-20260801-006 Fix-3 — apply_volume_shock_adjustment
#               新增 Breakout 引擎量能确认：Breakout Engine 触发必须伴随量能放大
#               (>1.5×) 才给 +5~8分绝对加分，否则视为假突破不确认。
#               修正 004/005 版仅动量/反转有量能确认、Breakout 引擎裸奔的缺口。
#               告警: 需 fold_scores 含 vol_20d_spike_t1 列（由 load_t1_daily 提供）。
# 2026-08-01 Claude: HERMES-20260801-005 (PATCH) — 注入「严格 T-1 时效约束」
#               的 Volume Shock Factor。新增 compute_volume_shock_factor /
#               apply_volume_shock_engine_bonus：Vol_20d_Spike 分子=昨日(T-1)
#               整日实际成交量、分母=过去20交易日整日平均量，绝不引用盘中/
#               分钟线数据（Tushare stk_mins 禁用，见 alpha_decay_risk.py）。
#               告警: 突破加分(>1.5 → +5~8分) / 缩量回踩(<0.7 → ×1.1) 只对
#               T-1 快照行生效；盘中调用请先过 load_t1_daily。
# ─────────────
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

# 项目根路径引导（支持独立运行 + 流水线内导入）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_decay_risk import (
    T1_SNAPSHOT_ROW_TAG,
    VOL_SPIKE_FEATURE_COL,
    VOL_SPIKE_BREAKOUT_THRESHOLD,
    VOL_SPIKE_BREAKOUT_BONUS_RANGE,
    VOL_SPIKE_REVERSION_THRESHOLD,
    VOL_SPIKE_REVERSION_BOOST,
    add_volume_shock_features,
    apply_volume_shock_bonus as _apply_volume_shock_bonus,
)

logger = logging.getLogger(__name__)

# 引擎常量（对外契约，dynamic_allocation.py / winrate_engine.py 复用）
ENGINE_MOMENTUM = "momentum"
ENGINE_REVERSION = "reversion"
ENGINE_BREAKOUT = "breakout"
ALL_ENGINES = [ENGINE_MOMENTUM, ENGINE_REVERSION, ENGINE_BREAKOUT]

ENGINE_CN = {
    ENGINE_MOMENTUM: "动量引擎",
    ENGINE_REVERSION: "均值回归引擎",
    ENGINE_BREAKOUT: "变盘引擎",
}

# ═══════════════════════════════════════════════════════════════
# Volume Shock Factor (HERMES-20260801-004) —— 绝对成交量验证
#
#   Vol_20d_Spike = Current_Volume / MA_Volume_20D
#     - 动量突破：Breakout 触发必须伴随量能放大 (>1.5) 才给 +5~8 分绝对加分
#       （确认资金真实入场，否则假突破）
#     - 均值回归：缩量回踩关键均线 (Vol<0.7×MA_20D) → Reversion 置信度大幅提升
#       （放量下跌往往没到底，缩量回踩才是超跌反弹的安全形态）
# ═══════════════════════════════════════════════════════════════
VOL_20D_SPIKE_COL = "vol_20d_spike"      # 量能异动比值列名
VOL_SPIKE_FLAG_COL = "vol_spike_flag"    # 放量标记 (vol_20d_spike > 1.5)
VOL_SHRINK_FLAG_COL = "vol_shrink_flag"  # 缩量标记 (vol_20d_spike < 0.7)
VOL_SPIKE_MULT = 1.5                     # 放量阈值（倍）
VOL_SHRINK_MULT = 0.7                    # 缩量阈值（倍）
VOL_BONUS_MIN = 0.05                     # 绝对加分下界（+5 分，P(Win) 0-1 标尺）
VOL_BONUS_MAX = 0.08                     # 绝对加分上界（+8 分）
MOMENTUM_BREAKOUT_PX_POS = 0.7           # 动量突破需价格位于 20 日区间上部（≥0.7）
REVERSION_MA_DEV_BELOW = 0.0             # 回踩均线判定: ma20_dev < 0（价格低于 MA20）

# 每个引擎必须保留的非特征列（路由/标签需要）
BASE_KEEP_COLUMNS = ["ts_code", "trade_date", "close"]

# ═══════════════════════════════════════════════════════════════
# 三引擎特征池（基于 feature_matrix_v5.parquet 的实际列名构建）
#
# 任务书中示例特征名 → 本系统实际特征名对照：
#   ma5_ratio                     → ma5_dev / ma5_ma10_gap / ma_gap
#   vol_14d_std                   → realized_vol_10d / turnover_std_20d
#   sector_strength_rank          → rank_l1_in_all / rank_l2_in_all / sector_persistence
#   rsi_14_deviation              → rsi_14 / rsi_divergence_bull / rsi_divergence_bear
#   bollinger_band_lower_touch_depth → bb_pct_b / bb_cross_lower / bb_streak
#   volatility_contraction_index  → bb_squeeze_flag / vol_regime_adaptive_ma / realized_vol_5d
#   key_ma_adhesion_score         → ma_short_arrangement / ma_long_arrangement / ma20_ma60_gap
# ═══════════════════════════════════════════════════════════════

MOMENTUM_FEATURES = [
    # 多周期动量
    "mom1", "mom2", "mom3", "mom4", "mom5", "mom7", "mom10", "mom12",
    "mom15", "mom20", "mom30", "mom45", "mom60", "mom75", "mom90", "mom120",
    "mom_harmonic_mean", "mom_frequency_ratio", "mom_ratio_5_20",
    "vol_adjusted_mom20", "exp_wgt_return_20d", "mom_accel",
    "upside_downside_ratio", "momentum_consistency",
    # 均线趋势结构
    "ma5_ma10_gap", "ma20_ma60_gap", "ma_gap", "ma_cross", "ma_arrangement",
    "ma_short_arrangement", "ma_long_arrangement",
    "macd_dif", "macd_signal", "macd_hist", "macd_cross_long",
    "macd_histogram_accel", "macd_zero_cross",
    # 趋势位置/强度
    "price_position_20d", "price_position_60d", "higher_high_count",
    "streak", "gain_loss_consistency", "momentum_volume", "gap_momentum",
    # 板块强度
    "sector_persistence", "excess_return_l1", "excess_return_l2",
    "excess_return_l3", "L3_L2_divergence", "L2_L1_divergence",
    "industry_cascade", "relative_strength", "stock_vs_L1_alpha_daily",
    "up_pct_l1", "rank_l1_in_all", "rank_l2_in_all",
    # 截面动量排名
    "rank_mom20", "rank_exp_wgt_return", "zscore_mom20", "momentum_quality",
    # 绝对成交量验证 (HERMES-20260801-004)：突破必须量能放大
    VOL_20D_SPIKE_COL, VOL_SPIKE_FLAG_COL,
    # 全量池扩池 (2026-08-05): 是否有申万映射（0=未分类/东财兜底，行业特征为占位）
    # 仅 momentum 引擎加入——行业特征最多的引擎，让模型学"该股行业特征是占位"
    "has_sw_mapping",
]

REVERSION_FEATURES = [
    # 均线乖离（超跌/超涨偏离）
    "ma3_dev", "ma5_dev", "ma8_dev", "ma10_dev", "ma15_dev", "ma20_dev",
    "ma30_dev", "ma40_dev", "ma60_dev", "ma80_dev", "ma90_dev",
    "ma120_dev", "ma250_dev",
    # 布林带触底
    "bb_pct_b", "bb_width", "bb_streak", "bb_cross_upper", "bb_cross_lower",
    "bb_squeeze_flag", "inside_bb_days",
    # RSI 偏离/背离
    "rsi_3", "rsi_5", "rsi_7", "rsi_14", "rsi_21", "rsi_60",
    "rsi_divergence_bull", "rsi_divergence_bear",
    # VWAP / 日内偏离
    "vwap_deviation", "intraday_vwap_proxy", "close_vs_intraday_ratio",
    "open_vs_intraday_ratio",
    # 超跌特征（回撤/偏度/下行波动）
    "max_drawdown_20d", "skewness_20d", "kurtosis_20d", "downside_vol",
    "vol_drawdown", "position_conviction", "price_volume_diverg",
    "idio_return_20d", "zscore_excess_return",
    # 反转 K 线形态
    "candle_hammer", "candle_shooting_star", "candle_doji",
    "candle_engulfing_bull", "candle_engulfing_bear",
    "candle_evening_star", "candle_three_soldiers", "candle_three_crows",
    "upper_shadow_ratio", "open_position_ratio", "open_position_accel",
    "body_to_range_ratio", "high_low_ratio_20d",
    # 绝对成交量验证 (HERMES-20260801-004)：缩量回踩才确认超跌反弹
    VOL_SHRINK_FLAG_COL,
]

BREAKOUT_FEATURES = [
    # 波动压缩/收缩
    "bb_squeeze_flag", "bb_width", "inside_bb_days",
    "vol_regime_adaptive_ma", "realized_vol_5d", "realized_vol_10d",
    "vol_surprise", "vol_change_acceleration",
    # 量能异动
    "vol_ratio", "vol_ma5_ratio", "vol_shock", "vol_shock_count_5d",
    "volume_oscillator", "volume_oscillator_accel", "volume_dry_up_flag",
    "amount_ratio", "amount_trend", "amount_ma10_ratio",
    "turnover_ratio_5d_20d", "turnover_std_20d",
    "vol_amplitude_ratio", "vol_conviction", "volume_shock_indicator",
    # 波动率水平
    "atr_pct", "daily_range", "yang_zhang_vol", "sharpe_20d",
    "close_position",
    # 时序结构（变盘前兆）
    "autocorr_1d", "autocorr_2d", "autocorr_5d", "variance_ratio_5_1",
    "hurst_exponent", "runs_ratio", "zero_return_days_20d",
    # 流动性枯竭
    "amihud_illiq_20d", "amihud_5d_mean", "price_impact",
    "corwin_schultz_spread",
    # 缺口/振幅
    "gap_pct", "gap_momentum", "high_vs_close_spread", "low_vs_close_spread",
]

# 引擎 → 特征池（主表）
ENGINE_FEATURE_MASKS: dict[str, list[str]] = {
    ENGINE_MOMENTUM: MOMENTUM_FEATURES,
    ENGINE_REVERSION: REVERSION_FEATURES,
    ENGINE_BREAKOUT: BREAKOUT_FEATURES,
}

# 引擎 → 关键词兜底分类器（当主表未命中新特征时自动归类）
ENGINE_KEYWORDS: dict[str, tuple[str, ...]] = {
    ENGINE_MOMENTUM: (
        "mom", "momentum", "ma_cross", "ma_arrangement", "macd", "persistence",
        "strength", "excess_return", "cascade", "trend", "gap_momentum",
        "higher_high", "price_position", "up_pct", "alpha", "relative_strength",
    ),
    ENGINE_REVERSION: (
        "dev", "bollinger", "bb_", "rsi", "divergence", "deviation", "vwap",
        "intraday", "drawdown", "skew", "kurt", "downside", "hammer",
        "shooting", "doji", "engulf", "evening", "soldiers", "crows",
        "shadow", "overbought", "oversold", "body_to_range", "open_position",
    ),
    ENGINE_BREAKOUT: (
        "vol", "squeeze", "atr", "range", "amplitude", "autocorr", "variance_ratio",
        "hurst", "runs_ratio", "amihud", "impact", "corwin", "spread", "gap_pct",
        "contraction", "adhesion", "shock", "oscillator", "dry_up", "realized_vol",
    ),
}

# 冗余特征集合（用于 cross-check 引擎间的特征重叠，避免两引擎喂同一梯度）
_OVERLAP_ALLOWED = {"bb_squeeze_flag", "bb_width", "inside_bb_days", "gap_momentum"}


# ═══════════════════════════════════════════════════════════════
# 核心 API
# ═══════════════════════════════════════════════════════════════

def get_engine_features(engine: str) -> list[str]:
    """返回指定引擎的完整特征池（拷贝，防止外部误改）。"""
    if engine not in ENGINE_FEATURE_MASKS:
        raise ValueError(f"未知引擎: {engine!r}，可选 {ALL_ENGINES}")
    return list(ENGINE_FEATURE_MASKS[engine])


def mask_feature_matrix(
    feature_df: pd.DataFrame,
    engine: str,
    include_close: bool = True,
    extra_keep: Iterable[str] = (),
) -> pd.DataFrame:
    """按引擎掩码裁剪特征矩阵，只保留该引擎可见的特征。

    硬性隔离：返回的 DataFrame 只含该引擎特征池中「实际存在」的列，
    外加路由所需的基础列。任何不属于该引擎的输入特征都会被丢弃。

    Args:
        feature_df: 全量特征矩阵（至少含 ts_code/trade_date）
        engine: ENGINE_MOMENTUM / ENGINE_REVERSION / ENGINE_BREAKOUT
        include_close: 是否保留 close（标签计算需要）
        extra_keep: 额外需要透传的列（如 group）

    Returns:
        裁剪后的 DataFrame
    """
    if feature_df is None or feature_df.empty:
        return feature_df
    if engine not in ENGINE_FEATURE_MASKS:
        raise ValueError(f"未知引擎: {engine!r}，可选 {ALL_ENGINES}")

    feats = [c for c in ENGINE_FEATURE_MASKS[engine] if c in feature_df.columns]
    keep = list(BASE_KEEP_COLUMNS if include_close else ["ts_code", "trade_date"])
    keep += [c for c in extra_keep if c in feature_df.columns]

    result = feature_df[keep + feats].copy()
    missing = [c for c in ENGINE_FEATURE_MASKS[engine] if c not in feature_df.columns]
    if missing:
        logger.debug("引擎 %s: %d/%d 特征不在当前矩阵 (%d 个缺失)",
                     engine, len(feats), len(ENGINE_FEATURE_MASKS[engine]), len(missing))
    return result


def compute_volume_shock_features(stock_daily_df: pd.DataFrame) -> pd.DataFrame:
    """从原始日线（含 vol）计算 Vol_20d_Spike 与量能异动标记。

    vol_20d_spike  = vol / MA20(vol)   （即任务书公式 Current_Volume / MA_Volume_20D）
    vol_spike_flag = vol_20d_spike > VOL_SPIKE_MULT  （放量 >1.5×）
    vol_shrink_flag= vol_20d_spike < VOL_SHRINK_MULT （缩量 <0.7×）

    Args:
        stock_daily_df: 原始日线，须含 ts_code / trade_date / vol，按 ts_code 聚合升序
    Returns:
        per-(ts_code,trade_date) 的 vol_20d_spike / vol_spike_flag / vol_shrink_flag
    """
    if (stock_daily_df is None or stock_daily_df.empty
            or "vol" not in stock_daily_df.columns):
        return pd.DataFrame(columns=["ts_code", "trade_date",
                                     VOL_20D_SPIKE_COL, VOL_SPIKE_FLAG_COL, VOL_SHRINK_FLAG_COL])
    df = stock_daily_df[["ts_code", "trade_date", "vol"]].copy()
    df = df.sort_values(["ts_code", "trade_date"])
    ma20 = df.groupby("ts_code")["vol"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    df[VOL_20D_SPIKE_COL] = (df["vol"] / ma20.replace(0, np.nan)).fillna(1.0)
    df[VOL_SPIKE_FLAG_COL] = (df[VOL_20D_SPIKE_COL] > VOL_SPIKE_MULT).astype(float)
    df[VOL_SHRINK_FLAG_COL] = (df[VOL_20D_SPIKE_COL] < VOL_SHRINK_MULT).astype(float)
    return df


def apply_volume_shock_adjustment(
    scored_df: pd.DataFrame,
    prob_col_template: str = "win_prob_{engine}",
    vol_spike_col: str = VOL_20D_SPIKE_COL,
    spike_mult: float = VOL_SPIKE_MULT,
    shrink_mult: float = VOL_SHRINK_MULT,
    bonus_min: float = VOL_BONUS_MIN,
    bonus_max: float = VOL_BONUS_MAX,
    breakout_px_pos: float = MOMENTUM_BREAKOUT_PX_POS,
    reversion_ma_below: float = REVERSION_MA_DEV_BELOW,
) -> pd.DataFrame:
    """绝对成交量验证（Volume Shock Factor）叠加到引擎 P(Win)。

    Momentum 突破确认：price_position_20d ≥ breakout_px_pos 且 vol_20d_spike>spike_mult
        → 加分 bonus = (spike-spike_mult)/spike × (bonus_max-bonus_min) + bonus_min ∈ [5,8] 分
        （量能放大越猛加分越多，确认资金真实入场）
    Reversion 缩量回踩确认：ma20_dev < reversion_ma_below 且 vol_20d_spike<shrink_mult
        → 置信度提升 boost = (shrink_mult-spike)/shrink_mult × (bonus_max-bonus_min) + bonus_min
        （缩量越充分置信度越高，防止「放量下跌没到底」的假反弹）

    纯规则叠加、不改训练数据；引擎 P(Win) 列缺失时自动跳过对应引擎。
    """
    df = scored_df.copy()
    if vol_spike_col not in df.columns or df.empty:
        return df
    spike = pd.to_numeric(df[vol_spike_col], errors="coerce").fillna(1.0)

    # ── Momentum：量能放大确认突破 ──
    mom_col = prob_col_template.format(engine=ENGINE_MOMENTUM)
    if mom_col in df.columns:
        px_pos = pd.to_numeric(
            df.get("price_position_20d", pd.Series(1.0, index=df.index)),
            errors="coerce").fillna(1.0)
        breakout = (spike > spike_mult) & (px_pos >= breakout_px_pos)
        bonus = np.clip((spike - spike_mult) / spike, 0, 1) * (bonus_max - bonus_min) + bonus_min
        df[mom_col] = df[mom_col].astype(float) + np.where(breakout, bonus, 0.0)
        df[mom_col] = df[mom_col].clip(0.0, 1.0)
        df["vol_momentum_confirmed"] = breakout.astype(float)

    # ── Breakout：量能放大确认变盘突破 (006 Fix-3) ──
    #    任务硬要求：Breakout Engine 触发必须伴随量能放大 (>1.5×) 才给 +8分绝对加分，
    #    确认资金真实入场，否则视为假突破不确认。
    bk_col = prob_col_template.format(engine=ENGINE_BREAKOUT)
    if bk_col in df.columns:
        bk_break = (spike > spike_mult)
        bk_bonus = np.clip((spike - spike_mult) / spike, 0, 1) * (bonus_max - bonus_min) + bonus_min
        df[bk_col] = df[bk_col].astype(float) + np.where(bk_break, bk_bonus, 0.0)
        df[bk_col] = df[bk_col].clip(0.0, 1.0)
        df["vol_breakout_confirmed"] = bk_break.astype(float)

    # ── Reversion：缩量回踩确认超跌反弹 ──
    rev_col = prob_col_template.format(engine=ENGINE_REVERSION)
    if rev_col in df.columns:
        ma_dev = pd.to_numeric(
            df.get("ma20_dev", pd.Series(-1.0, index=df.index)),
            errors="coerce").fillna(-1.0)
        shrink_pullback = (spike < shrink_mult) & (ma_dev < reversion_ma_below)
        boost = np.clip((shrink_mult - spike) / shrink_mult, 0, 1) * (bonus_max - bonus_min) + bonus_min
        df[rev_col] = df[rev_col].astype(float) + np.where(shrink_pullback, boost, 0.0)
        df[rev_col] = df[rev_col].clip(0.0, 1.0)
        df["vol_reversion_confirmed"] = shrink_pullback.astype(float)

    return df


def validate_masks(feature_df: pd.DataFrame) -> dict:
    """校验三引擎掩码在当前特征矩阵上的覆盖情况。

    Returns:
        {engine: {"defined": n, "available": n, "coverage": float, "missing": [...]}}
    """
    report = {}
    for engine in ALL_ENGINES:
        defined = ENGINE_FEATURE_MASKS[engine]
        avail = [c for c in defined if c in feature_df.columns]
        report[engine] = {
            "defined": len(defined),
            "available": len(avail),
            "coverage": round(len(avail) / len(defined), 4) if defined else 0.0,
            "missing": [c for c in defined if c not in feature_df.columns][:10],
        }
    return report


def auto_classify(feature_name: str) -> str | None:
    """兜底：按关键词把未知特征名归类到某个引擎。

    命中多个引擎时按列表顺序返回第一个；无命中返回 None。
    """
    name = feature_name.lower()
    for engine in ALL_ENGINES:
        for kw in ENGINE_KEYWORDS[engine]:
            if kw.lower() in name:
                return engine
    return None


def compute_overlap() -> dict[str, list[str]]:
    """计算三引擎特征池之间的交叉重叠（用于审查梯度泄漏）。"""
    overlap: dict[str, list[str]] = {}
    for i, e1 in enumerate(ALL_ENGINES):
        for e2 in ALL_ENGINES[i + 1:]:
            common = sorted(set(ENGINE_FEATURE_MASKS[e1]) & set(ENGINE_FEATURE_MASKS[e2]))
            overlap[f"{e1}∩{e2}"] = common
    return overlap


def ensure_no_gradient_leak() -> bool:
    """硬校验：除白名单外，三引擎特征池不得共享同一输入特征。

    任务硬性要求「停止在单模型里混合所有梯度」，此函数在流水线入口
    强制保证各引擎的输入掩码互斥。

    Returns:
        True 表示无泄漏；False 表示存在白名单外的重叠特征
    """
    ok = True
    for pair, common in compute_overlap().items():
        leak = [c for c in common if c not in _OVERLAP_ALLOWED]
        if leak:
            logger.warning("梯度泄漏 %s: %s", pair, leak)
            ok = False
    return ok


# ═══════════════════════════════════════════════════════════════
# Volume Shock Factor（严格 T-1 量能异动，HERMES-20260801-004/005）
# ═══════════════════════════════════════════════════════════════

# 引擎内可用列（breakout 已含 vol_shock 系列；这里补显式 T-1 快照列）
VOLUME_SHOCK_FEATURES = [
    VOL_SPIKE_FEATURE_COL,          # vol_20d_spike_t1
    "vol_shock",                    # vol[i]/vol[i-1]（整日）
    "volume_shock_indicator",       # vol / ma20_vol（整日）
    "vol_ratio", "vol_ma5_ratio", "vol_shock_count_5d",
]


def compute_volume_shock_factor(
    feature_df: pd.DataFrame,
    vol_col: str = "vol",
    asof_date: str | None = None,
    ma_window: int = 20,
) -> pd.DataFrame:
    """计算每只股票 T-1 的 Vol_20d_Spike 并追加到特征矩阵（严格 T-1）。

    Vol_20d_Spike = vol_{T-1} / mean(vol_{T-20 .. T-1})
      分子 = T-1 收盘当天的实际整日成交量（非实时流盘推演）
      分母 = 过去 20 个交易日的整日平均成交量
    排序/打分只依赖 T-1 及更早整日收盘快照；盘中/分钟线数据会触发
    alpha_decay_risk._assert_closing_snapshot_only 拒绝。

    Args:
        feature_df: 特征矩阵（含 ts_code/trade_date/vol，整日收盘口径）
        vol_col: 成交量列名（默认 vol）
        asof_date: T-1 截止日（YYYYMMDD），None → 取数据内最新
        ma_window: 量能均线窗口（默认 20）

    Returns:
        追加 vol_20d_spike_t1 与 is_t1_snapshot 列的 DataFrame。
    """
    return add_volume_shock_features(feature_df, vol_col=vol_col,
                                     asof_date=asof_date, ma_window=ma_window)


def apply_volume_shock_engine_bonus(
    scored_df: pd.DataFrame,
    engine: str,
    spike_col: str = VOL_SPIKE_FEATURE_COL,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """给指定引擎的评分施加绝对量能验证加分（只作用于 T-1 快照行）。

    规则（任务硬要求）：
      - momentum / breakout 突破：vol_20d_spike > 1.5 → 绝对加分
        +5 ~ 8 分（量能越大分越高），确认资金真实入场。
      - reversion 缩量回踩：vol_20d_spike < 0.7 → 置信度 ×1.1，
        防止「放量下跌没到底」的假反弹。

    非 T-1 快照行（is_t1_snapshot=False）保持原分不动——绝不在盘中
    流盘数据上打分。

    Args:
        scored_df: 已含 spike_col 与 score_col 的评分表
        engine: ENGINE_MOMENTUM / ENGINE_REVERSION / ENGINE_BREAKOUT
        spike_col: Vol_20d_Spike 列名
        score_col: 绝对评分列名

    Returns:
        带 vol_shock_bonus / vol_shock_conf_boost 列的 DataFrame。
    """
    if scored_df is None or scored_df.empty:
        return scored_df
    if T1_SNAPSHOT_ROW_TAG not in scored_df.columns:
        # 未标记快照行 → 视为全部是 T-1 快照（由 load_t1_daily 保证）
        out = scored_df.copy()
        out[T1_SNAPSHOT_ROW_TAG] = True
    else:
        out = scored_df.copy()
    snap_mask = out[T1_SNAPSHOT_ROW_TAG].fillna(False).astype(bool)

    # 只对 T-1 快照行做量能加分；盘中行保持原分
    snap = out[snap_mask].copy()
    if not snap.empty:
        if engine == ENGINE_REVERSION:
            snap = _apply_volume_shock_bonus(
                snap, spike_col=spike_col, score_col=score_col,
                reversion_threshold=VOL_SPIKE_REVERSION_THRESHOLD,
                reversion_boost=VOL_SPIKE_REVERSION_BOOST,
                breakout_threshold=1e9,          # reversion 不触发突破加分
            )
        else:
            snap = _apply_volume_shock_bonus(
                snap, spike_col=spike_col, score_col=score_col,
                breakout_threshold=VOL_SPIKE_BREAKOUT_THRESHOLD,
                breakout_bonus_range=VOL_SPIKE_BREAKOUT_BONUS_RANGE,
                reversion_threshold=-1.0,        # momentum/breakout 不触发缩量加成
            )
    out.loc[snap_mask, [c for c in snap.columns if c in out.columns]] = snap
    # 补充新增列
    for c in ("vol_shock_bonus", "vol_shock_conf_boost"):
        if c in snap.columns and c not in out.columns:
            out[c] = 0.0
            out.loc[snap_mask, c] = snap[c].values
    return out


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

def _self_test() -> dict:
    """最小自检：用合成矩阵验证掩码逻辑。"""
    fake_cols = (
        ["ts_code", "trade_date", "close"] + MOMENTUM_FEATURES[:3]
        + REVERSION_FEATURES[:3] + BREAKOUT_FEATURES[:3]
    )
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(rng.normal(size=(n, len(fake_cols))), columns=fake_cols)
    df["ts_code"] = "000001.SZ"
    df["trade_date"] = [f"20260{i%9:02d}{i%28:02d}" for i in range(n)]

    result = {}
    for engine in ALL_ENGINES:
        masked = mask_feature_matrix(df, engine)
        result[engine] = {
            "n_cols": masked.shape[1],
            "only_engine_features": set(masked.columns) <= (set(ENGINE_FEATURE_MASKS[engine]) | set(BASE_KEEP_COLUMNS)),
        }
    result["no_gradient_leak"] = ensure_no_gradient_leak()
    result["coverage_report"] = validate_masks(df)

    # ── Volume Shock Factor（严格 T-1，HERMES-20260801-005）──
    rng2 = np.random.default_rng(5)
    n2 = 25
    dates2 = pd.bdate_range("2026-06-01", periods=n2).strftime("%Y%m%d").tolist()
    vols_boom = np.clip(rng2.normal(1_000_000, 150_000, n2), 500_000, 1_500_000)
    vols_boom[-1] = 2_200_000                                   # T-1 放量
    vols_quiet = np.clip(rng2.normal(1_000_000, 150_000, n2), 500_000, 1_500_000)
    vols_quiet[-1] = 500_000                                    # T-1 缩量
    vol_df = pd.DataFrame({
        "ts_code": ["BOOM"] * n2 + ["QUIET"] * n2,
        "trade_date": dates2 + dates2,
        "close": 10.0,
        "vol": np.concatenate([vols_boom, vols_quiet]).astype(float),
    })
    vol_feat = compute_volume_shock_factor(vol_df, asof_date=dates2[-1])
    assert VOL_SPIKE_FEATURE_COL in vol_feat.columns
    assert vol_feat[T1_SNAPSHOT_ROW_TAG].sum() == 2, "应恰有 2 行 T-1 快照"
    t1_rows = vol_feat[vol_feat[T1_SNAPSHOT_ROW_TAG]]
    spike_by_code = t1_rows.set_index("ts_code")[VOL_SPIKE_FEATURE_COL]
    assert spike_by_code["BOOM"] > 1.5, "放量票 T-1 spike 应 >1.5"
    assert spike_by_code["QUIET"] < 0.7, "缩量票 T-1 spike 应 <0.7"

    # 引擎加分：breakout 放量 → 绝对加分 ≥5；reversion 缩量 → 置信度 ×1.1
    scored = pd.DataFrame({
        "ts_code": ["BOOM", "QUIET"],
        VOL_SPIKE_FEATURE_COL: [float(spike_by_code["BOOM"]), float(spike_by_code["QUIET"])],
        T1_SNAPSHOT_ROW_TAG: [True, True],
        "composite_score": [80.0, 75.0],
    })
    bk = apply_volume_shock_engine_bonus(scored.copy(), ENGINE_BREAKOUT)
    rv = apply_volume_shock_engine_bonus(scored.copy(), ENGINE_REVERSION)
    result["volume_shock"] = {
        "boom_spike_t1": float(spike_by_code["BOOM"]),
        "quiet_spike_t1": float(spike_by_code["QUIET"]),
        "breakout_bonus_boom": float(bk.set_index("ts_code").loc["BOOM"]["vol_shock_bonus"]),
        "reversion_boost_quiet": float(rv.set_index("ts_code").loc["QUIET"]["vol_shock_conf_boost"]),
        "t1_snapshot_rows": int(vol_feat[T1_SNAPSHOT_ROW_TAG].sum()),
    }
    assert result["volume_shock"]["breakout_bonus_boom"] >= 5.0
    assert result["volume_shock"]["reversion_boost_quiet"] == 1.1
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import json
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))
