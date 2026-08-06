#!/usr/bin/env python3
"""V6 Master Plan — 动态组合路由与闸门 (Dynamic Allocation Engine)。

替代「死板的一律买 Top 5」——依据大盘 Regime (bull/range/bear) 路由仓位：

  Regime        策略路由                          推荐数  仓位上限
  ────────────  ───────────────────────────────  ──────  ────────
  bull (牛)     动量权重最高 (60%)，正常推荐       Top 5   100%
  range (震荡)  砍掉趋势/突破，≥70% 给 Reversion  Top 3   50%
  bear (熊)     防御机制：禁高波动，仅低波防御     Top 3   20%

Regime 来源：analysis/market_ml.py（大盘 ML）+ market_regime.py（规则状态机）。

# ── Changelog ──
# 2026-08-01 Claude: HERMES-20260801-006 Fix-3 修正——量能异动加分改回严格 T-1：
#               默认量能列 vol_20d_spike_t1（vol_{T-1}/MA20），放量突破 +[0.05,0.08]
#               绝对加分、缩量回踩 ×1.1。修正 004/005 版用 vol_ratio + 常量 +5/+3
#               直接加在 0-1 composite_score 上导致排序被量能主导的缺陷。
#               告警: 调用方须传 vol_20d_spike_t1 列（由 load_t1_daily 提供），
#               缺失时自动跳过（vol_boost=""）。
# 2026-08-01 Claude: Fix-1 (HERMES-20260801-003) 胜率闸门从「一票否决」降级为
#               软性权重 Soft-Weighting。新增 SOFT_WEIGHT_TARGET / SOFT_WEIGHT_MIN /
#               SOFT_WEIGHT_MAX / soft_weight_composite()。route_portfolio 在合并
#               信号前用 soft_weight 替代 apply_winrate_gate 硬淘汰。
#               告警: apply_winrate_gate 仍保留（自检与外部调用用），gate_summary.veto
#               语义改为「软权重降权数」，不再淘汰候选。
# 2026-08-01 Claude: Fix-3 build_defense_pool 修正——SELECT 补回 trade_date 并按
#               ts_code+trade_date 排序后 tail(1)，确保 PE/PB 取「截至 upto_date 最新」
# 2026-08-01 Claude: HERMES-20260801-004/005 新增 V6 v2 量能异动加分——
#               apply_volume_shock_adjust() 在 Top-N 排序前对 composite_score 施加
#               量能确认加分（放量突破 vol_ratio>1.5 加分；缩量回踩 <0.7 增强 Reversion）。
#               仅用 T-1 收盘 vol_ratio（005 约束，无盘中数据）。
#               告警: 需 fold_scores 含 vol_ratio 列，缺失时跳过。
# 2026-08-01 Claude: HERMES-20260801-005 (PATCH) — Alpha Decay 严格 T-1
#               约束注入。新增 apply_alpha_decay_cut()，route_portfolio 新增
#               alpha_decay_scores 可选参数（Top-N 排序前直接移出触发
#               P(Today)<P(Init)*0.75 的持仓）。数据必须来自
#               alpha_decay_risk.load_t1_daily（截止 T-1 整日收盘），
#               禁用 Tushare stk_mins 盘中接口。
#               告警: alpha_decay_scores 缺失时跳过砍仓，行为与原版一致。
#               而非任意一行；修复动量加成日志 % 转义（前3%%）。
#               告警: 无——行为仅影响防御池估值新鲜度。
# ─────────────
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

# 项目根路径引导
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.strategy_feature_masker import ENGINE_MOMENTUM, ENGINE_REVERSION, ENGINE_BREAKOUT
from analysis.winrate_engine import (
    DEFAULT_WINRATE_GATE,
    apply_winrate_gate,
    evaluate_panel,
)
from analysis.t1_gap_risk import (
    DEFAULT_RISK_THRESHOLD,
    DEFAULT_PENALTY,
    apply_t1_gap_penalty,
)
from analysis.alpha_decay_risk import (
    DEFAULT_DECAY_RATIO,
    DECAY_CUT_COL,
    apply_alpha_decay_cut as _apply_alpha_decay_cut,
)

logger = logging.getLogger(__name__)

# Regime 常量
REGIME_BULL = "bull"
REGIME_RANGE = "range"
REGIME_BEAR = "bear"

# 规则 → 三态归一化（兼容 market_ml.py 的 regime_label / market_regime.py 的 v2_label）
REGIME_NORMALIZE_MAP = {
    # 牛市态
    "bull": REGIME_BULL, "BULL": REGIME_BULL, "early_bull": REGIME_BULL,
    "rebound": REGIME_RANGE,   # 反弹行情=结构性机会 → 震荡路由（Reversion）
    # 震荡态
    "range": REGIME_RANGE, "RANGE": REGIME_RANGE, "range_up": REGIME_RANGE,
    "range_down": REGIME_RANGE, "pullback": REGIME_RANGE,
    # 熊市态
    "bear": REGIME_BEAR, "BEAR": REGIME_BEAR,
}

# 三态路由规则（任务硬要求）
ALLOCATION_RULES: dict[str, dict] = {
    REGIME_BULL: {
        "top_n": 5,
        "engine_weights": {ENGINE_MOMENTUM: 0.60, ENGINE_REVERSION: 0.20, ENGINE_BREAKOUT: 0.20},
        "max_position": 1.00,
        "gate_engine": ENGINE_MOMENTUM,
        "description": "牛市：正常推荐 Top 5，动量策略权重最高",
    },
    REGIME_RANGE: {
        "top_n": 3,
        "engine_weights": {ENGINE_MOMENTUM: 0.00, ENGINE_REVERSION: 0.70, ENGINE_BREAKOUT: 0.00},
        "max_position": 0.50,
        "gate_engine": ENGINE_REVERSION,
        "description": "震荡：砍掉趋势/突破，≥70% 仓位分配给 Reversion 捕捉结构性反弹，推荐缩减至 Top 3",
    },
    REGIME_BEAR: {
        "top_n": 3,
        "engine_weights": {ENGINE_MOMENTUM: 0.00, ENGINE_REVERSION: 0.00, ENGINE_BREAKOUT: 0.00},
        "max_position": 0.20,
        "gate_engine": ENGINE_REVERSION,
        "defensive_only": True,
        "description": "熊市：防御机制，仓位上限 <20%，禁止高波动策略，仅低波动红利/防御标的",
    },
}

# 防御过滤参数（熊市）
DEFENSIVE_KEEP_RATIO = 0.40        # 保留防御评分前 40%（最多保底 10 只）
DEFENSIVE_VALUE_WEIGHT = 0.5       # 价值/红利在防御评分中的权重

# ── 软权重闸门参数 (Fix-1, HERMES-20260801-003) ──
# 将「P(Win) < 阈值直接淘汰」降级为「按概率衰减权重」，绝不删候选：
#   weight_multiplier = np.clip(P_Win / Target_Win_Rate, min, max)
SOFT_WEIGHT_TARGET = 0.55      # 目标胜率（基准，默认与旧一票否决阈值一致）
SOFT_WEIGHT_MIN = 0.3          # 权重下界：即使 P(Win)=45% 权重也保留 0.8
SOFT_WEIGHT_MAX = 1.2          # 权重上界：高置信度可获得不超过 1.2 加成


def soft_weight_composite(
    scored_df: pd.DataFrame,
    prob_cols: list[str],
    target_rate: float = SOFT_WEIGHT_TARGET,
    w_min: float = SOFT_WEIGHT_MIN,
    w_max: float = SOFT_WEIGHT_MAX,
    out_score_col: str = "composite_score",
) -> pd.DataFrame:
    """软权重合成：保留全部候选，按 P(Win)/Target 动态加权。

    weight_multiplier = np.clip(P_Win / Target, w_min, w_max)
    composite_score  = raw_engine_probability * weight_multiplier

    即使 P(Win)=45%，权重降至 45/55 ≈ 0.82 → 排序轻微降权，
    但绝不因一票否决导致排序功能退化（Rank IC 崩溃根因）。

    Args:
        scored_df: 候选池，须含至少一个 prob_cols 中的 P(Win) 列
        prob_cols: 参与合成的引擎 P(Win) 列（regime 有效引擎）
        target_rate: 目标胜率
        w_min / w_max: 权重裁剪边界
        out_score_col: 输出复合总分列名

    Returns:
        新增/覆盖 composite_score 与 soft_multiplier 列的 DataFrame。
    """
    df = scored_df.copy()
    avail = [c for c in prob_cols if c in df.columns]
    if not avail:
        raise ValueError(f"缺少任何 P(Win) 列: {prob_cols}")
    base = df[avail].mean(axis=1)                      # 有效引擎概率均值
    mult = np.clip(base / target_rate, w_min, w_max)  # 软权重乘数
    df["soft_multiplier"] = np.round(mult, 4)
    df[out_score_col] = (base * mult).round(4)
    return df


# ── V6 量能异动加分 (HERMES-20260801-006 Fix-3 修正) ──
# 修正 004/005 版缺陷：原版用 vol_ratio + 常量 +5/+3 直接加在 0-1 标尺的
# composite_score 上，导致量能加分完全主导排序（Rank IC 被拖累）。006 版：
#   · 量能比列使用严格 T-1 的 vol_20d_spike_t1 = vol_{T-1}/MA20(vol)（唯一执行标准）
#   · 放量突破 (>1.5) 施加 [0.05, 0.08] 绝对加分（对应任务书「+8分绝对加分」0-100 标尺）
#   · 缩量回踩 (<0.7) 施加 ×1.1 置信度加成（Reversion 得分大幅提升）
VOL_BREAKOUT_MIN = 1.5    # 放量阈值：突破需量能 > MA20(vol) 的 1.5 倍
VOL_SHRINK_MAX = 0.7      # 缩量阈值：回踩需量能 < MA20(vol) 的 0.7 倍
VOL_BOOST_MIN = 0.05      # 放量突破绝对加分下界（0-1 composite 标尺 = +5分/100）
VOL_BOOST_MAX = 0.08      # 放量突破绝对加分上界（0-1 composite 标尺 = +8分/100）
VOL_REVERSION_BOOST = 1.1 # 缩量回踩的 Reversion 置信度加成倍数


def apply_volume_shock_adjust(
    scored_df: pd.DataFrame,
    vol_col: str = "vol_20d_spike_t1",
    breakout_min: float = VOL_BREAKOUT_MIN,
    shrink_max: float = VOL_SHRINK_MAX,
    boost_min: float = VOL_BOOST_MIN,
    boost_max: float = VOL_BOOST_MAX,
    reversion_boost: float = VOL_REVERSION_BOOST,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """量能异动加分 (006 Fix-3)：用严格 T-1 量能比对 composite_score 施加量能确认。

    - 放量突破确认：vol_20d_spike_t1 > 1.5（资金真实入场）→ composite_score
      绝对加分 bonus ∈ [boost_min, boost_max]（0-1 标尺，量能放大越猛加分越高，
      对应任务书「+8分绝对加分」）。
    - 缩量回踩确认：vol_20d_spike_t1 < 0.7（缩量回踩关键均线）→ composite_score
      ×reversion_boost（Reversion 方向置信度大幅提升），防止「放量下跌没到底」的假反弹。

    严格 T-1（006 Fix-3）：vol_col 默认 vol_20d_spike_t1 = vol_{T-1}/MA20(vol)，
    由 alpha_decay_risk.load_t1_daily + compute_volume_shock_batch 提供；绝无盘中/
    分钟线数据（BANNED_INTRADAY_API: tushare.stk_mins）。

    Args:
        scored_df: 候选池（须含 vol_col 量能比列）
        vol_col: 量能比列名（默认 vol_20d_spike_t1；缺失则跳过）
        breakout_min: 放量阈值
        shrink_max: 缩量阈值
        boost_min / boost_max: 放量绝对加分下/上界（0-1 标尺）
        reversion_boost: 缩量 Reversion 置信度加成倍数
        score_col: 被加分的总分列

    Returns:
        新增 vol_boost 列（"breakout"|"shrink"|""）与加分后的 DataFrame。
    """
    df = scored_df.copy()
    if vol_col not in df.columns or score_col not in df.columns or df.empty:
        df["vol_boost"] = ""
        return df
    vol = df[vol_col].fillna(1.0)
    # 放量突破：绝对加分随量能线性放大 ∈ [boost_min, boost_max]
    is_breakout = vol > breakout_min
    bonus = np.where(is_breakout,
                     np.clip((vol - breakout_min) / vol, 0, 1) * (boost_max - boost_min) + boost_min,
                     0.0)
    # 缩量回踩：置信度加成倍数
    is_shrink = vol < shrink_max
    mult = np.where(is_shrink, reversion_boost, 1.0)
    df["vol_boost"] = np.where(is_breakout, "breakout",
                               np.where(is_shrink, "shrink", ""))
    if bonus.any():
        df[score_col] = (df[score_col] + bonus).round(4)
    if (mult != 1.0).any():
        df[score_col] = (df[score_col] * mult).round(4)
    return df


# ── V6 v2 Alpha Decay 动态止损 (HERMES-20260801-004/005) ──
ALPHA_DECAY_RATIO = 0.75   # 持仓股 P(Win) 衰减至初始 75% 以下 → 立即砍仓


def alpha_decay_should_cut(
    init_prob: float,
    current_prob: float,
    ratio: float = ALPHA_DECAY_RATIO,
) -> bool:
    """Alpha Decay 判定：当前 T-1 P(Win) < 初始 P(Win)×0.75 → 触发砍仓。

    用昨日(T-1)收盘数据重算的胜率概率与建仓时对比（005 约束：不引用当日盘中）。
    衰减超过 25% 即视为 Alpha 衰退，立即离场，不依赖盘中固定止损。

    Args:
        init_prob: 建仓时 P(Win)
        current_prob: 持有日 T-1 收盘重算的 P(Win)
        ratio: 衰减阈值（默认 0.75）

    Returns:
        True=应砍仓；False=继续持有。
    """
    if current_prob is None or not np.isfinite(current_prob):
        return False
    if init_prob is None or init_prob <= 0:
        return False
    return current_prob < init_prob * ratio


def apply_alpha_decay_cut(
    scored_df: pd.DataFrame,
    decay_scores: pd.DataFrame | None,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """持仓 Alpha Decay 砍仓（005 严格 T-1 约束注入）。

    触发 P(Today) < P(Init)*0.75 的持仓直接移出组合（立即砍仓，
    不依赖盘中固定止损）。decay_scores 来自
    alpha_decay_risk.compute_alpha_decay_batch —— 其 P(Today) 只在
    T-1 整日收盘快照上重算（见 alpha_decay_risk.py 模块头约束：
    禁止 Tushare stk_mins 等盘中接口）。未提供 decay_scores 时
    原表返回，alpha_decay_cut 列置 False。

    Args:
        scored_df: 候选/持仓评分表（含 ts_code + score_col）
        decay_scores: alpha_decay_risk.compute_alpha_decay_batch 输出
                      （ts_code + alpha_decay_cut）
        score_col: 总分列名（透传）

    Returns:
        剔除触发砍仓行后的 DataFrame，附 alpha_decay_cut 列。
    """
    return _apply_alpha_decay_cut(scored_df, decay_scores, score_col=score_col)


def normalize_regime(regime: str | None) -> str:
    """把各种 regime 标签归一化为 bull / range / bear。"""
    if regime is None:
        return REGIME_RANGE
    key = str(regime).strip().lower()
    # 先试原样映射
    if key in REGIME_NORMALIZE_MAP:
        return REGIME_NORMALIZE_MAP[key]
    # market_regime.py 的 v2 标签（小写）
    if key in REGIME_NORMALIZE_MAP:
        return REGIME_NORMALIZE_MAP[key]
    logger.warning("未知 regime=%r，默认按 range 处理", regime)
    return REGIME_RANGE


def get_allocation(regime: str | None) -> dict:
    """返回指定 regime 的路由规则。"""
    r = normalize_regime(regime)
    return dict(ALLOCATION_RULES[r], **{"regime": r})


def get_regime_from_market_ml() -> dict:
    """从大盘模型/规则状态机读取当前 regime（bull/range/bear）。

    Returns:
        {"regime": "bull|range|bear", "source": "market_ml|market_regime|fallback",
         "raw_regime": ..., "up_prob": float|None, "details": dict}
    """
    result = {"regime": REGIME_RANGE, "source": "fallback", "raw_regime": None, "up_prob": None}
    # ① 优先：market_regime 规则状态机（明确输出 BULL/RANGE/BEAR）
    try:
        from analysis.market_regime import determine_regime
        from analysis.retrain_ml import load_industry_data
        l1_daily, _, _ = load_industry_data()
        if l1_daily is not None and not l1_daily.empty:
            rr = determine_regime(l1_daily)
            raw = rr.get("regime", "RANGE")
            result.update({
                "regime": normalize_regime(raw),
                "source": "market_regime",
                "raw_regime": raw,
                "details": rr,
            })
            return result
    except Exception as e:
        logger.warning("market_regime 不可用: %s", e)
    # ② 备选：market_ml 大盘模型 up_prob
    try:
        from analysis.market_ml import load_market_model, predict_market_today
        model = load_market_model()
        if model is not None:
            prob, details = predict_market_today(None, None)
            result.update({
                "regime": REGIME_BULL if prob > 0.6 else (REGIME_BEAR if prob < 0.4 else REGIME_RANGE),
                "source": "market_ml",
                "raw_regime": "prob",
                "up_prob": prob,
                "details": details,
            })
    except Exception as e:
        logger.warning("market_ml 不可用: %s", e)
    return result


# ═══════════════════════════════════════════════════════════════
# 引擎信号合并 + 路由
# ═══════════════════════════════════════════════════════════════

def combine_engine_scores(
    scored_df: pd.DataFrame,
    engine_weights: dict[str, float],
    prob_col_template: str = "win_prob_{engine}",
) -> pd.DataFrame:
    """按 regime 权重合并各引擎 P(Win) → 复合总分。

    列约定：momentum 引擎的 P(Win) 列名为 win_prob_momentum，其余类推。
    若某个引擎列缺失，其权重按剩余引擎归一化重分配。
    """
    df = scored_df.copy()
    avail = {}
    for engine, w in engine_weights.items():
        col = prob_col_template.format(engine=engine)
        if col in df.columns and w > 0:
            avail[col] = w
    if not avail:
        raise ValueError("scored_df 缺少任何引擎的 P(Win) 列，无法合成总分")
    total_w = sum(avail.values())
    norm = {c: w / total_w for c, w in avail.items()}
    df["composite_score"] = sum(df[c] * w for c, w in norm.items())
    df["engine_weights"] = [norm] * len(df)
    return df


def _defensive_score(df: pd.DataFrame) -> pd.Series:
    """防御评分：波动率越低分越高，叠加价值/红利倾斜。

    保证结果在 [0, 1+weights] 区间、不会整池清空（用排名而非硬阈值）。
    """
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    vol_col = None
    for c in ("atr_pct", "realized_vol_10d", "realized_vol_5d"):
        if c in out.columns:
            vol_col = c
            break
    if vol_col is not None:
        # 低波动 → 高防御分（pct 排名反号）
        score = score - out[vol_col].rank(pct=True)
    # 价值/红利倾斜：value_score / earnings_yield 越高防御性越好
    if "value_score" in out.columns:
        score = score + DEFENSIVE_VALUE_WEIGHT * out["value_score"].rank(pct=True)
    if "earnings_yield" in out.columns:
        score = score + DEFENSIVE_VALUE_WEIGHT * out["earnings_yield"].rank(pct=True)
    return score


def _defensive_filter(df: pd.DataFrame, keep_ratio: float = DEFENSIVE_KEEP_RATIO) -> pd.DataFrame:
    """熊市防御过滤：按防御评分保留极低波动 + 价值/红利倾向标的。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["defensive_score"] = _defensive_score(out)
    n_keep = max(int(len(out) * keep_ratio), min(10, len(out)))
    return out.nlargest(n_keep, "defensive_score").copy()


# ── Fix-3 熊市防御池 (HERMES-20260801-003) ──
DEFENSE_PE_MAX = 15.0      # PE(TTM) 上界（低估值）
DEFENSE_PB_MAX = 2.0       # PB 上界（低估值）
DEFENSE_ATR_MAX = 2.0      # 日均 ATR < 2%（低波动）
DEFENSE_MAX_STOCKS = 60    # 防御池容量上限


def build_defense_pool(
    db_path: str | None = None,
    upto_date: str | None = None,
    min_pe: float = 0.0,
    max_pe: float = DEFENSE_PE_MAX,
    max_pb: float = DEFENSE_PB_MAX,
    max_atr: float = DEFENSE_ATR_MAX,
    limit: int = DEFENSE_MAX_STOCKS,
) -> list[str]:
    """构建低波动红利防御池（Fix-3）。

    规则：PE(TTM)∈[min_pe,max_pe] 且 PB<max_pb 且 近20日 ATR%<max_atr。
    数据源：fundamental_cache (pe_ttm/pb) + stock_daily (ATR)。
    用 upto_date 之前的时点数据（防未来泄漏），无则取最新。

    Returns:
        防御池 ts_code 列表（按 ATR 升序，即最低波动优先）。
    """
    import sqlite3
    db_path = db_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_storage", "sw_index_data.db")
    conn = sqlite3.connect(db_path)
    try:
        # ① 基本面：PE/PB 双低（fundamental_cache）
        #    必须带 trade_date 并排序，groupby.tail(1) 才能取到「截至 upto_date 最新」的估值
        if upto_date:
            fq = pd.read_sql_query(
                "SELECT ts_code, trade_date, pe_ttm, pb FROM fundamental_cache "
                "WHERE trade_date <= ? AND pe_ttm IS NOT NULL AND pb IS NOT NULL",
                conn, params=(upto_date,))
        else:
            fq = pd.read_sql_query(
                "SELECT ts_code, trade_date, pe_ttm, pb FROM fundamental_cache "
                "WHERE pe_ttm IS NOT NULL AND pb IS NOT NULL", conn)
        if fq.empty:
            logger.warning("防御池: fundamental_cache 无 PE/PB 数据")
            return []
        fq = fq.sort_values(["ts_code", "trade_date"])
        fq = fq.groupby("ts_code").tail(1)  # 每只取最新
        fq = fq[(fq["pe_ttm"] >= min_pe) & (fq["pe_ttm"] <= max_pe) & (fq["pb"] < max_pb)]
        codes = fq["ts_code"].tolist()
        if not codes:
            return []

        # ② 低波动：近 20 日 ATR%（stock_daily）
        ph = ",".join("?" for _ in codes)
        if upto_date:
            rows = pd.read_sql_query(
                f"SELECT ts_code, trade_date, high, low, close FROM stock_daily "
                f"WHERE trade_date <= ? AND ts_code IN ({ph})",
                conn, params=[upto_date] + codes)
        else:
            rows = pd.read_sql_query(
                f"SELECT ts_code, trade_date, high, low, close FROM stock_daily "
                f"WHERE ts_code IN ({ph})", conn, params=codes)
        conn.close()
    except Exception as e:
        conn.close()
        logger.warning("防御池构建失败: %s", e)
        return []

    if rows.empty:
        return []
    rows = rows.sort_values(["ts_code", "trade_date"])
    atr_list = []
    for code, g in rows.groupby("ts_code"):
        g = g.tail(21)
        if len(g) < 11:
            continue
        tr = np.maximum(g["high"] - g["low"],
                        np.maximum(abs(g["high"] - g["close"].shift(1)),
                                   abs(g["low"] - g["close"].shift(1))))
        atr = tr.mean()
        atr_pct = atr / g["close"].iloc[-1] * 100 if g["close"].iloc[-1] > 0 else 999
        if atr_pct < max_atr:
            atr_list.append((code, round(float(atr_pct), 3)))
    atr_list.sort(key=lambda x: x[1])
    pool = [c for c, _ in atr_list[:limit]]
    logger.info("防御池: %d 只 (PE<%.0f/PB<%.1f/ATR<%.1f%%)", len(pool), max_pe, max_pb, max_atr)
    return pool


def route_portfolio(
    scored_df: pd.DataFrame,
    regime: str | None,
    winrate_gate: float = DEFAULT_WINRATE_GATE,
    t1_risk_scores: pd.DataFrame | None = None,
    t1_threshold: float = DEFAULT_RISK_THRESHOLD,
    t1_penalty: float = DEFAULT_PENALTY,
    score_col: str = "composite_score",
    prob_col_template: str = "win_prob_{engine}",
    momentum_boost_col: str | None = None,
    defense_pool: list[str] | None = None,
    volume_shock_col: str | None = None,
    alpha_decay_scores: pd.DataFrame | None = None,
) -> dict:
    """完整路由闸门：Regime → 引擎权重 → 软权重 → T+1 Gap 罚分 → Alpha Decay → 动量加成 → 量能异动 → Top-N。

    Fix-2 (HERMES-20260801-003): 新增 momentum_boost_col 可选参数。若候选池带
    momentum_boost 布尔列（True=5日收益全市场前3%），Top-N 排序前 composite_score
    额外 ×1.1，补偿短期动能属性。
    Fix-3 (HERMES-20260801-003): 新增 defense_pool 可选参数（低波动红利池）。
    V6 v2 (HERMES-20260801-004/005): 新增 volume_shock_col 可选参数。候选池带
    vol_ratio（T-1 收盘量能比）时，Top-N 排序前施加量能确认加分：
    放量突破(>1.5) +5 分，缩量回踩(<0.7) Reversion +3 分。
    V6 v2 (HERMES-20260801-005, PATCH): 新增 alpha_decay_scores 可选参数。
    alpha_decay_risk.compute_alpha_decay_batch 输出（T-1 整日收盘快照上重算
    P(Win)，P(Today) < P(Init)*0.75 触发砍仓）。Top-N 排序前直接移出触发
    Alpha Decay 的持仓——不依赖盘中固定止损。绝不允许引用当日盘中/分钟线数据。
    若 regime==bear 且给定了防御池，则最终 Top-N 中强制至少 1 席来自防御池，
    切断趋势动量导致的熊市滑铁卢。

    Args:
        scored_df: 候选池，须含 ts_code 与各引擎 P(Win) 列
                   （win_prob_momentum / win_prob_reversion / win_prob_breakout）
        regime: "bull" / "range" / "bear"（或任意原始标签）
        winrate_gate: 胜率一票否决阈值
        t1_risk_scores: compute_t1_lock_risk_batch 输出（ts_code + score）
        t1_threshold / t1_penalty: T+1 风险闸门参数
        alpha_decay_scores: Alpha Decay 决策表（ts_code + alpha_decay_cut），
                            None 则跳过砍仓

    Returns:
        {
          "regime": ..., "allocation": {...},
          "portfolio": DataFrame (final Top-N),
          "gate_summary": {...},
          "panel": {Rank IC, Actual Win Rate (>55%), Avg_Volatility_of_Top5},
        }
    """
    if scored_df is None or scored_df.empty:
        return {"error": "无候选数据", "regime": normalize_regime(regime)}

    alloc = get_allocation(regime)
    regime_norm = alloc["regime"]
    df = scored_df.copy()

    gate_engine = alloc["gate_engine"]
    gate_col = prob_col_template.format(engine=gate_engine)
    if gate_col not in df.columns:
        raise ValueError(f"缺少 gate 引擎 {gate_engine!r} 的 P(Win) 列: {gate_col}")

    # ── 熊市特判：先在全池上做防御筛选（仅低波动红利/防御），再做胜率闸门 ──
    n_defensive = 0
    if regime_norm == REGIME_BEAR and alloc.get("defensive_only"):
        before = len(df)
        df = _defensive_filter(df)
        n_defensive = len(df)
        logger.info("熊市防御筛选: %d → %d", before, n_defensive)
        if df.empty:
            return {
                "regime": regime_norm, "allocation": alloc,
                "portfolio": df,
                "gate_summary": {"veto": 0, "pass": before, "defensive_survivors": 0},
                "panel": {"Rank IC": 0.0, "Actual Win Rate (>55%)": 0.0, "Avg_Volatility_of_Top5": 0.0},
            }

    # ① 软权重闸门 (Fix-1, HERMES-20260801-003)：替代一票否决。
    #    保留全部候选，按 P(Win)/Target 动态加权，绝不硬淘汰。
    prob_cols = [prob_col_template.format(engine=e) for e, w in alloc["engine_weights"].items() if w > 0]
    if not prob_cols or gate_col not in df.columns:
        prob_cols = [gate_col]
    # 鲁棒兜底：某折引擎训练失败导致 P(Win) 列缺失时，退化为任一存在的 P(Win) 列
    if gate_col not in df.columns and not any(c in df.columns for c in prob_cols):
        alt = [c for c in df.columns if c.startswith("win_prob_")]
        if not alt:
            raise ValueError(f"无任何 P(Win) 列可用于路由: {list(df.columns)}")
        prob_cols = [alt[0]]
    n_veto = int((df[gate_col] < winrate_gate).sum())  # 低于目标胜率的票数（降权非淘汰）
    if df.empty:
        return {
            "regime": regime_norm, "allocation": alloc,
            "portfolio": df,
            "gate_summary": {"veto": n_veto, "pass": 0, "defensive_survivors": n_defensive},
            "panel": {"Rank IC": 0.0, "Actual Win Rate (>55%)": 0.0, "Avg_Volatility_of_Top5": 0.0},
        }

    # ② 引擎权重合并（regime 内有效引擎）
    if regime_norm == REGIME_BEAR:
        # 熊市：防御池上软权重合成，叠加防御评分微调
        df = soft_weight_composite(df, [gate_col], target_rate=winrate_gate)
        if "defensive_score" in df.columns:
            df["composite_score"] = df["composite_score"] * 0.5 + df["defensive_score"] * 0.5
        df["engine_weights"] = [alloc["engine_weights"]] * len(df)
    else:
        # 非熊市：先按 regime 引擎权重合并 P(Win)，再乘软权重乘数（Fix-1）
        df = combine_engine_scores(df, alloc["engine_weights"], prob_col_template=prob_col_template)
        mult = np.clip(df[gate_col].astype(float) / winrate_gate, SOFT_WEIGHT_MIN, SOFT_WEIGHT_MAX)
        df["soft_multiplier"] = np.round(mult, 4)
        df["composite_score"] = (df["composite_score"] * df["soft_multiplier"]).round(4)
        df["engine_weights"] = [alloc["engine_weights"]] * len(df)

    # ③ T+1 Gap-Risk 罚分
    if t1_risk_scores is not None and not t1_risk_scores.empty:
        df = apply_t1_gap_penalty(
            df, t1_risk_scores, threshold=t1_threshold, penalty=t1_penalty,
            score_col=score_col,
        )
    else:
        df["t1_lock_risk"] = 0.0
        df["t1_high_risk"] = False

    # ④ Alpha Decay 动态止损 (HERMES-20260801-004/005, 严格 T-1)
    #    只消费 T-1 整日收盘快照重算的 P(Win)；无盘中/分钟线数据
    if alpha_decay_scores is not None and not alpha_decay_scores.empty:
        df = apply_alpha_decay_cut(df, alpha_decay_scores, score_col=score_col)
    else:
        df[DECAY_CUT_COL] = False

    # ⑤ 动量加成 (Fix-2, HERMES-20260801-003) + 量能异动 (V6 v2) + 选 Top-N
    if momentum_boost_col and momentum_boost_col in df.columns:
        n_boost = int(df[momentum_boost_col].sum())
        if n_boost > 0:
            df.loc[df[momentum_boost_col], score_col] = (
                df.loc[df[momentum_boost_col], score_col] * 1.1)
            logger.info("动量加成: %d 只 ×1.1 (全市场前3%%短动)", n_boost)
    if volume_shock_col and volume_shock_col in df.columns:
        df = apply_volume_shock_adjust(df, vol_col=volume_shock_col, score_col=score_col)
        n_bk = int((df["vol_boost"] == "breakout").sum())
        n_sh = int((df["vol_boost"] == "shrink").sum())
        if n_bk or n_sh:
            logger.info("量能异动: 放量突破 %d 只 +%.0f~%.0f分, 缩量回踩 %d 只 ×%.1f",
                        n_bk, VOL_BOOST_MIN * 100, VOL_BOOST_MAX * 100, n_sh, VOL_REVERSION_BOOST)
    n_pass = len(df)
    n_survivors = len(df)
    top_n = alloc["top_n"]
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    portfolio = df.head(top_n).copy()

    # Fix-3: 熊市强制防御池席位（至少 1 席来自低波动红利池）
    if regime_norm == REGIME_BEAR and defense_pool:
        pool_set = set(defense_pool)
        if not pool_set.intersection(portfolio["ts_code"].tolist()):
            # 从防御池中取排序最高的一只替换组合内最后一席
            def_pool_df = df[df["ts_code"].isin(pool_set)]
            if not def_pool_df.empty:
                best_def = def_pool_df.sort_values(score_col, ascending=False).iloc[0]
                portfolio = portfolio.iloc[:-1].copy()
                portfolio = pd.concat([portfolio, pd.DataFrame([best_def])], ignore_index=True)
                logger.info("熊市防御席位: 强制置入 %s（防御池）", best_def["ts_code"])

    # ⑥ V6 评估面板（在通过闸门的全池上计算，替代 R²）
    if "fwd_return_20d" in df.columns:
        panel = evaluate_panel(df, pred_col=score_col, top_n=top_n)
    else:
        vol = portfolio["atr_pct"].mean() if "atr_pct" in portfolio.columns else 0.0
        panel = {"Rank IC": 0.0, "Actual Win Rate (>55%)": 0.0,
                 "Avg_Volatility_of_Top5": round(float(vol), 4)}

    return {
        "regime": regime_norm,
        "allocation": alloc,
        "portfolio": portfolio,
        "gate_summary": {"veto": n_veto, "pass": n_pass, "survivors": n_survivors},
        "panel": panel,
    }


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

def _self_test() -> dict:
    """合成候选池验证三态路由。"""
    rng = np.random.default_rng(21)
    n = 60
    df = pd.DataFrame({
        "ts_code": [f"{i:06d}.SZ" for i in range(n)],
        "win_prob_momentum": rng.uniform(0.3, 0.85, n),
        "win_prob_reversion": rng.uniform(0.3, 0.85, n),
        "win_prob_breakout": rng.uniform(0.3, 0.85, n),
        "atr_pct": rng.uniform(1.0, 9.0, n),
        "value_score": rng.uniform(-1, 2, n),
        "fwd_return_20d": rng.normal(0, 8, n),
    })

    results = {}
    for regime in (REGIME_BULL, REGIME_RANGE, REGIME_BEAR):
        r = route_portfolio(df, regime, winrate_gate=0.55)
        alloc = r["allocation"]
        port = r["portfolio"]
        results[regime] = {
            "top_n": len(port),
            "expected_top_n": alloc["top_n"],
            "max_position": alloc["max_position"],
            "weights": alloc["engine_weights"],
            "gate_veto": r["gate_summary"]["veto"],
            "all_pass_above_gate": bool((port["win_prob_" + alloc["gate_engine"]] >= 0.55).all()),
        }
        assert len(port) <= alloc["top_n"]

    # 熊市防御：组合 atr 均值应显著低于全池
    bull_port = results[REGIME_BULL]
    assert bull_port["top_n"] == 5, "牛市应为 Top 5"
    assert bull_port["all_pass_above_gate"]
    assert results[REGIME_RANGE]["top_n"] == 3, "震荡应为 Top 3"
    assert results[REGIME_RANGE]["weights"][ENGINE_REVERSION] >= 0.7, "震荡 Reversion 权重应 ≥70%"
    assert results[REGIME_BEAR]["max_position"] == 0.2, "熊市仓位上限应 <20%"

    # regime 归一化
    assert normalize_regime("BULL") == REGIME_BULL
    assert normalize_regime("early_bull") == REGIME_BULL
    assert normalize_regime("range_down") == REGIME_RANGE
    assert normalize_regime("BEAR") == REGIME_BEAR

    # ── Alpha Decay 砍仓 (HERMES-20260801-005, 严格 T-1) ──
    holdings = df.head(8).copy()
    # 构造 P(Init) 与 T-1 重算 P(Today)：一半触发 <0.75 砍仓
    p_init = np.array([0.80, 0.60, 0.70, 0.75, 0.65, 0.55, 0.72, 0.68])
    p_today = np.array([0.78, 0.40, 0.68, 0.50, 0.64, 0.52, 0.30, 0.67])
    decay_in = pd.DataFrame({
        "ts_code": holdings["ts_code"].tolist(),
        "win_prob_entry": p_init,
        "win_prob_t1": p_today,
    })
    from analysis.alpha_decay_risk import compute_alpha_decay_batch
    decay_scores = compute_alpha_decay_batch(decay_in, decay_in)
    cut_port = apply_alpha_decay_cut(holdings, decay_scores)
    n_cut = int((~holdings["ts_code"].isin(cut_port["ts_code"])).sum())
    expected_cut = int((p_today < p_init * DEFAULT_DECAY_RATIO).sum())
    results["alpha_decay"] = {
        "expected_cut": expected_cut,
        "actual_cut": n_cut,
        "cut_ratio": DEFAULT_DECAY_RATIO,
    }
    assert n_cut == expected_cut, f"Alpha Decay 砍仓数 {n_cut} != 期望 {expected_cut}"

    # ── route_portfolio 接入 alpha_decay_scores（Top-N 前移出砍仓股）──
    routed = route_portfolio(
        holdings, REGIME_BULL, winrate_gate=0.55,
        alpha_decay_scores=decay_scores,
    )
    results["route_alpha_decay"] = {
        "portfolio_size": len(routed["portfolio"]),
        "decayed_removed": all(
            bool(decay_scores.set_index("ts_code").loc[c, DECAY_CUT_COL]) is False
            for c in routed["portfolio"]["ts_code"]
        ) if not routed["portfolio"].empty else True,
    }

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import json
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))
