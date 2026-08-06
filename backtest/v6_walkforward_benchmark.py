#!/usr/bin/env python3
"""V6 Walk-Forward Backtest & Win Rate Benchmarking (HERMES-20260801-002).

实战级验证：2024 年至今 A 股数据按日切分 98 折时间序列 Walk-Forward，
三引擎（momentum/reversion/breakout）各自独立训练二分类胜率模型，
组合层施加 T+1 Gap-Risk 罚分 + Regime 动态路由，输出 V6 指标看板：
  Strategy_Rank_IC         (> +0.3 期望)
  Actual_Win_Rate_Verified (T+1 与 5-Day 平均胜率)
  Max_Drawdown_Protected   (组合层最大回撤，观察 Gap Guard / Regime Gate 拦截)

防未来函数泄漏：
  - 每折验证集与训练集时间物理隔离（train 仅用 fold 前数据）
  - regime 用截至 val_end 的行业数据判定（历史模拟）
  - T+1 Lock-Risk 用截至验证日的日线计算

# ── Changelog ──
# 2026-08-02 Claude: HERMES-20260802-001 审计需求补充——
#   每日交易明细输出：新增 daily_trades（验证期每交易日 gate 引擎 Top-N 的
#   win_prob/量能/regime/次日与5日收益），输出 *_daily_trades.csv + JSON。
#   新增 --since 参数（4年回测用 20200101；半年验证用 20260201）。
# 2026-08-02 Claude: HERMES-20260802-001 审计修复 4 Bug——
#   BUG-1: holdings append-only list → dict {ts_code: init_p} last-write-wins 去重，
#          杜绝同股重复条目导致旧概率漏判 Alpha Decay。
#   BUG-2: 组合层 Alpha Decay 真正执行——用上一折 init_p 与本折 P(Win) 对比，
#          触发砍仓的股票直接从 port 移出（alpha_cut = 组合层实际 cuts）。
#   BUG-3: MaxDD 拆分——MaxDrawdownOverPeriods(period级) + MaxDrawdownDaily(日度)。
#   BUG-4: Rank IC 透明报告——summary 加 rank_ic_boosted + Strategy_Rank_IC_Unboosted。
# 告警: 无——holdings dict 兼容跨折评估；rank_ic_unboosted 需 composite_score 列。
# 2026-08-01 Claude: HERMES-20260801-006 Fix-3 — Walk-Forward 接入「严格 T-1 量能异动」：
#   (1) 每折用 load_t1_daily（trade_date <= T-1）计算 vol_20d_spike_t1 = vol_{T-1}/MA20(vol)，
#       合并进 fold_scores；T-1 = 验证窗口前最后一个交易日，绝无盘中/分钟线数据。
#   (2) 调用 sfm.apply_volume_shock_adjustment 在引擎层施加量能确认
#       （动量/突破放量 +5~8分绝对加分，反转缩量 ×1.1）。
#   (3) route_portfolio 不再传 volume_shock_col，避免 composite 层二次叠加。
# 告警: 需 fold_scores 含 vol_20d_spike_t1；缺失列自动跳过量能确认（行为同 005）。
# 2026-08-01 Claude: HERMES-20260801-003 接入 Fix-2/Fix-3 后暴露的两处缺陷修复——
#   (1) fold_scores 未透传 mom5/value_score/earnings_yield/quality_score，
#       导致 momentum_boost 恒为 False、熊市防御评分只剩波动维度；
#       现从全量特征矩阵 merge 回这些列（防未来函数泄漏：只用 feat 中该折已有数据）。
#   (2) Fix-2 动量 top-3% 的局部变量 top_n 遮蔽了组合 Top-N 参数，更名 mom_top_n。
# 告警: 无——动量加成需 fold_scores 含 mom5 列，防御评分需 value/earnings 列。
# 2026-08-01 Claude: HERMES-20260801-004/005 V6 v2 两项改造——
#   (1) Volume Shock: fold_scores 透传 vol_ratio(T-1收盘量能比)，route_portfolio
#       传 volume_shock_col，放量突破>1.5 加分/缩量回踩<0.7 增强 Reversion。
#   (2) Alpha Decay: 跨折持仓跟踪，本折用 T-1 收盘 P(Win) 重估上折持仓，
#       P(Win)<初始×0.75 则砍仓（alpha_cut），提前截断暴跌。
# 告警: 无——vol_ratio/mom5 缺失时对应逻辑自动跳过；Alpha Decay 需 gate 引擎列存在。
# ─────────────

用法:
  python3 backtest/v6_walkforward_benchmark.py --smoke     # 快速验证管道（500只/10折）
  python3 backtest/v6_walkforward_benchmark.py --out backtest/v6_walkforward_report_20260801.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, COL_TS_CODE, COL_TRADE_DATE

from analysis import strategy_feature_masker as sfm
from analysis import winrate_engine as wre
from analysis import t1_gap_risk as t1
from analysis import dynamic_allocation as da
from analysis.alpha_decay_risk import load_t1_daily, compute_volume_shock_batch
from analysis.retrain_ml import load_stock_daily, load_industry_data
from analysis.ml_model import build_full_feature_matrix

logger = logging.getLogger("v6_wf")

FORWARD_DAYS = 20
GATE_THRESHOLD = 0.55  # P(Win) < 55% 一票否决（任务硬要求）


# ── 数据层 ─────────────────────────────────────────────────────
def load_t1_risk_upto(codes: list[str], upto_date: str,
                      lookback_days: int = 60) -> pd.DataFrame:
    """加载截至 upto_date（含）前 lookback_days 个交易日的日线，计算 T+1 Lock-Risk。

    与 t1.load_stock_daily 的区别：不取全局最新，而是回看截至验证日的数据，
    避免用未来数据计算历史折的 T+1 风险。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        recent = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM stock_daily "
            "WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
            conn, params=(upto_date, lookback_days + 5),
        )
        if recent.empty:
            return pd.DataFrame()
        cutoff = recent["trade_date"].min()
        if codes:
            ph = ",".join("?" for _ in codes)
            df = pd.read_sql_query(
                f"SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg "
                f"FROM stock_daily WHERE trade_date >= ? AND trade_date <= ? "
                f"AND ts_code IN ({ph}) ORDER BY ts_code, trade_date",
                conn, params=[cutoff, upto_date] + codes,
            )
        else:
            df = pd.read_sql_query(
                "SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg "
                "FROM stock_daily WHERE trade_date >= ? AND trade_date <= ? "
                "ORDER BY ts_code, trade_date",
                conn, params=[cutoff, upto_date],
            )
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame()
    return t1.compute_t1_lock_risk_batch(df)


def load_price_panel(codes: list[str], upto_date: str, fwd_days: int = 6) -> pd.DataFrame:
    """加载验证日当天与后续 fwd_days 天的价格，用于计算 T+1 / 5-Day 真实收益。

    未来收益仅用于【评估】（标签计算用），不会参与训练。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        future = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date > ? "
            "ORDER BY trade_date ASC LIMIT ?",
            conn, params=(upto_date, fwd_days + 1),
        )
        if future.empty:
            return pd.DataFrame()
        future_cutoff = future["trade_date"].max()
        ph = ",".join("?" for _ in codes)
        df = pd.read_sql_query(
            f"SELECT ts_code, trade_date, close FROM stock_daily "
            f"WHERE trade_date BETWEEN ? AND ? AND ts_code IN ({ph}) "
            f"ORDER BY ts_code, trade_date",
            conn, params=[upto_date, future_cutoff] + codes,
        )
    finally:
        conn.close()
    return df


def alloc_engine(regime: str) -> str:
    """按 regime 返回 gate 引擎（与 dynamic_allocation.ALLOCATION_RULES 一致）。"""
    if regime == "bull":
        return "momentum"
    return "reversion"  # range / bear 均以 reversion 为 gate 引擎


def _unboosted_ic(fold_scores: pd.DataFrame, regime: str, gate_engine: str) -> float:
    """BUG-4 (HERMES-20260802-001)：无 Alpha Boost 的 Rank IC 对照。

    用 gate 引擎的 P(Win) 列作 pred（alpha_boost=1.0），与 fwd_return 算 Spearman。
    若 gate 列缺失，退化为任一 win_prob 均值。返回 0.0 表示无可用数据。
    """
    if fold_scores is None or fold_scores.empty:
        return 0.0
    gate_col = f"win_prob_{gate_engine}"
    if gate_col not in fold_scores.columns:
        prob_cols = [c for c in fold_scores.columns if c.startswith("win_prob_")]
        if not prob_cols:
            return 0.0
        gate_col = prob_cols[0]
    pred = fold_scores[gate_col].astype(float)
    actual = fold_scores["fwd_return_20d"] if "fwd_return_20d" in fold_scores.columns else None
    if actual is None:
        return 0.0
    from scipy.stats import spearmanr
    valid = pred.notna() & actual.notna()
    if valid.sum() < 10:
        return 0.0
    r, _ = spearmanr(pred[valid], actual[valid])
    return round(float(r), 4) if not np.isnan(r) else 0.0


def regime_upto(l1_daily: pd.DataFrame, upto_date: str) -> str:
    """用截至 upto_date 的行业数据判定历史 regime（防未来泄漏）。"""
    if l1_daily is None or l1_daily.empty:
        return "range"
    sub = l1_daily[l1_daily[COL_TRADE_DATE] <= upto_date]
    if sub.empty:
        return "range"
    try:
        from analysis.market_regime import determine_regime
        rr = determine_regime(sub)
        return da.normalize_regime(rr.get("regime"))
    except Exception:
        return "range"


# ── 核心回测 ───────────────────────────────────────────────────
def run_walkforward(
    feat: pd.DataFrame,
    l1_daily: pd.DataFrame,
    n_folds: int = 98,
    initial_train_days: int = 120,
    val_days: int = 5,
    step_days: int = 5,
    top_n: int = 5,
    verbose: bool = False,
) -> dict:
    """98 折时间序列 Walk-Forward：三引擎独立训练 → 组合层风控路由 → 净值回测。"""
    t_start = time.time()
    feat = feat.dropna(subset=[COL_TRADE_DATE]).sort_values([COL_TS_CODE, COL_TRADE_DATE])
    dates = sorted(feat[COL_TRADE_DATE].unique())
    n_dates = len(dates)
    if n_dates < initial_train_days + val_days + FORWARD_DAYS:
        return {"error": f"交易日不足: {n_dates}"}

    # 三引擎掩码一次到位
    engines = {}
    for eng in sfm.ALL_ENGINES:
        masked = sfm.mask_feature_matrix(feat, eng, include_close=True)
        masked = wre.build_winrate_labels(masked, forward_days=FORWARD_DAYS)
        feats = [c for c in sfm.get_engine_features(eng) if c in masked.columns]
        engines[eng] = {"masked": masked, "feats": feats}

    # 价格面板（用于组合收益与 Max Drawdown）
    price_wide = feat[[COL_TS_CODE, COL_TRADE_DATE, "close", "atr_pct"]].copy()

    fold_records = []
    all_val_scores = []   # 汇总验证期 P(Win) 与真实收益（算 Rank IC / 胜率）
    portfolio_navs = []   # 组合净值序列（折尾记录）
    daily_navs = []       # 折内日度净值序列（BUG-3 修复，用于日度 MaxDD）
    daily_trades = []     # 每日交易明细（HERMES-20260802-001，供抽样验证）
    regime_counts = {"bull": 0, "range": 0, "bear": 0}
    holdings: dict[str, float] = {}   # 跨折持仓跟踪 {ts_code: init_prob} — V6 v2 Alpha Decay

    # 自适应步长：让可用验证区间（去头去尾）切出尽可能接近 n_folds 的折。
    # n_folds<=0（全部折）时用固定 5 天步长，产生密集折供每日明细抽样。
    available = n_dates - initial_train_days - FORWARD_DAYS
    if n_folds and n_folds > 0:
        eff_step = max(2, (available - val_days) // max(n_folds - 1, 1))
    else:
        eff_step = step_days

    train_end = initial_train_days
    fold_idx = 0
    fold_cap = n_folds if (n_folds and n_folds > 0) else 10 ** 6  # n_folds<=0 → 全部折
    while train_end + val_days <= n_dates - FORWARD_DAYS and fold_idx < fold_cap:
        fold_idx += 1
        val_start_idx = train_end
        val_end_idx = train_end + val_days - 1
        val_dates = dates[val_start_idx:val_end_idx + 1]
        val_start, val_end = dates[val_start_idx], dates[val_end_idx]

        # ── 逐引擎训练 + 预测验证期 P(Win) ──
        fold_scores = None
        for eng, info in engines.items():
            masked = info["masked"]
            feats = info["feats"]
            tr = masked[masked[COL_TRADE_DATE].isin(set(dates[:train_end]))].dropna(subset=["y_label"])
            va = masked[masked[COL_TRADE_DATE].isin(val_dates)].dropna(subset=["y_label"])
            if len(tr) < 300 or va.empty:
                continue
            model, _imp = wre.train_winrate_model(tr, feature_cols=feats, forward_days=FORWARD_DAYS)
            if model is None:
                continue
            va = va.copy()
            va[f"win_prob_{eng}"] = wre.predict_win_probability(model, va[feats].fillna(0))
            keep = [COL_TS_CODE, COL_TRADE_DATE, "close", "fwd_return_20d", f"win_prob_{eng}"]
            part = va[keep].dropna(subset=[f"win_prob_{eng}"])
            if fold_scores is None:
                fold_scores = part
            else:
                fold_scores = fold_scores.merge(
                    part[[COL_TS_CODE, COL_TRADE_DATE, f"win_prob_{eng}"]],
                    on=[COL_TS_CODE, COL_TRADE_DATE], how="inner")

        if fold_scores is None or fold_scores.empty:
            train_end += step_days
            continue

        # 补充 atr_pct 用于 Avg_Volatility_of_Top5 与熊市防御筛选
        fold_scores = fold_scores.merge(
            price_wide[[COL_TS_CODE, COL_TRADE_DATE, "atr_pct"]],
            on=[COL_TS_CODE, COL_TRADE_DATE], how="left")

        # ── Fix-2/Fix-3/V6v2: 透传动量/防御/量能维度列（引擎掩码裁剪后被丢弃）──
        _meta_avail = [c for c in ("mom5", "value_score", "earnings_yield", "quality_score",
                                   "vol_ratio")
                       if c in feat.columns]
        if _meta_avail:
            fold_scores = fold_scores.merge(
                feat[[COL_TS_CODE, COL_TRADE_DATE] + _meta_avail].drop_duplicates(
                    [COL_TS_CODE, COL_TRADE_DATE]),
                on=[COL_TS_CODE, COL_TRADE_DATE], how="left")

        # ── Fix-3 严格 T-1 量能异动 (HERMES-20260801-006)：只读 T-1 整日收盘 ──
        #    Vol_20d_Spike = vol_{T-1} / MA20(vol)。T-1 = 验证窗口前最后一个交易日。
        #    load_t1_daily 强制 trade_date <= asof（SQL 边界），compute_volume_shock_batch
        #    内部走 _assert_closing_snapshot_only 拒绝盘中/分钟线形态；绝不调用
        #    Tushare stk_mins（BANNED_INTRADAY_API）。
        _t1_codes = fold_scores[COL_TS_CODE].unique().tolist()
        _t1_before = [d for d in dates if d < val_start]
        if _t1_before:
            _t1_asof = _t1_before[-1]
            _t1_daily = load_t1_daily(_t1_codes, lookback_days=40, asof_date=_t1_asof)
            _vol_spike = compute_volume_shock_batch(_t1_daily, asof_date=_t1_asof)
            _vol_spike = _vol_spike[["ts_code", "vol_20d_spike_t1"]]
            fold_scores = fold_scores.merge(_vol_spike, on="ts_code", how="left")
            fold_scores["vol_20d_spike_t1"] = fold_scores["vol_20d_spike_t1"].fillna(1.0)
        else:
            fold_scores["vol_20d_spike_t1"] = 1.0

        # ── Fix-3 引擎级量能确认（动量/突破放量 +5~8分绝对加分，反转缩量 ×1.1）──
        _vol_conf_cols = [c for c in ("price_position_20d", "ma20_dev") if c in feat.columns]
        if _vol_conf_cols:
            fold_scores = fold_scores.merge(
                feat[[COL_TS_CODE, COL_TRADE_DATE] + _vol_conf_cols].drop_duplicates(
                    [COL_TS_CODE, COL_TRADE_DATE]),
                on=[COL_TS_CODE, COL_TRADE_DATE], how="left")
        fold_scores = sfm.apply_volume_shock_adjustment(
            fold_scores, vol_spike_col="vol_20d_spike_t1")

        # ── 历史 regime（截至 val_end，防泄漏）──
        regime = regime_upto(l1_daily, val_end)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # ── T+1 Gap-Risk（截至 val_end）──
        codes = fold_scores[COL_TS_CODE].unique().tolist()
        t1_risk = load_t1_risk_upto(codes, val_end)
        t1_summary = t1_risk[["ts_code", "score"]].rename(
            columns={"score": "t1_lock_risk"}).to_dict("records") if not t1_risk.empty else []

        # ── Fix-2 动量加成列：mom5 全市场前 3%（HERMES-20260801-003）──
        if "mom5" in fold_scores.columns and len(fold_scores) > 0:
            mom_top_n = max(int(np.ceil(len(fold_scores) * 0.03)), 1)
            fold_scores["momentum_boost"] = (
                fold_scores["mom5"].fillna(-1e9).rank(ascending=False, method="first") <= mom_top_n
            ).values
        else:
            fold_scores["momentum_boost"] = False

        # ── Fix-3 熊市防御池（截至 val_end，防未来泄漏）──
        defense_pool = da.build_defense_pool(upto_date=val_end) if regime == "bear" else None

        # ── Regime 动态路由（软权重 + T+1 + 动量加成 + 防御池）──
        #    006 Fix-3：量能确认已在 fold_scores 上按引擎施加（apply_volume_shock_adjustment），
        #    此处不再传 volume_shock_col，避免 composite 层二次叠加。
        routed = da.route_portfolio(
            fold_scores, regime,
            winrate_gate=GATE_THRESHOLD,
            t1_risk_scores=t1_risk,
            momentum_boost_col="momentum_boost",
            defense_pool=defense_pool,
        )
        port = routed.get("portfolio", pd.DataFrame())
        panel = routed.get("panel", {})
        gate = routed.get("gate_summary", {})

        # ── 每日交易明细记录（HERMES-20260802-001：供抽样验证系统效果）──
        #    对验证期内每个交易日，按 gate 引擎 P(Win) 排序取当日 Top-N，
        #    记录股票、分数、量能、regime 及次日/5日真实收益，供人工抽样核验。
        _gate_engine = alloc_engine(regime)
        _gate_col = f"win_prob_{_gate_engine}"
        _port_topn = alloc.get("top_n", top_n) if (alloc := routed.get("allocation")) else top_n
        if not fold_scores.empty and _gate_col in fold_scores.columns:
            for _day, _day_df in fold_scores.groupby(COL_TRADE_DATE):
                _day_top = _day_df.sort_values(_gate_col, ascending=False).head(_port_topn)
                for _, _r in _day_top.iterrows():
                    _px = load_price_panel([_r[COL_TS_CODE]], _day)
                    _sorted = _px.sort_values(COL_TRADE_DATE)
                    _entry = float(_sorted.iloc[0]["close"]) if not _sorted.empty else None
                    _t1px = float(_sorted.iloc[1]["close"]) if len(_sorted) >= 2 else None
                    _d5px = float(_sorted["close"].iloc[min(5, len(_sorted) - 1)]) if len(_sorted) >= 2 else None
                    daily_trades.append({
                        "trade_date": _day,
                        "ts_code": _r[COL_TS_CODE],
                        "regime": regime,
                        "gate_engine": _gate_engine,
                        "win_prob": round(float(_r[_gate_col]), 4),
                        "vol_20d_spike": round(float(_r["vol_20d_spike_t1"]), 4) if "vol_20d_spike_t1" in _r.index else None,
                        "mom5": round(float(_r["mom5"]), 4) if "mom5" in _r.index else None,
                        "entry_close": round(_entry, 2) if _entry else None,
                        "t1_close": round(_t1px, 2) if _t1px else None,
                        "d5_close": round(_d5px, 2) if _d5px else None,
                        "t1_ret_pct": round((_t1px / _entry - 1) * 100, 2) if _entry and _t1px else None,
                        "d5_ret_pct": round((_d5px / _entry - 1) * 100, 2) if _entry and _d5px else None,
                    })

        # ── 组合 T+1 / 5-Day 真实收益（仅评估用）──
        t1_win, t1_total, d5_win, d5_total = 0, 0, 0, 0
        alpha_cut = 0   # Alpha Decay 触发砍仓数（portfolio 层实际执行）

        # BUG-2 修复 (HERMES-20260802-001)：组合层真正执行 Alpha Decay cuts。
        # 用上一折持有的 init_p 与本折 fold_scores 的 P(Win) 对比，触发砍仓的
        # 股票直接从 port 移出（组合层生效，而非仅 cross-fold 计数）。
        # 首次折（无历史 holdings）不砍仓。
        if not port.empty and holdings:
            fold_prob_map = {}
            if not fold_scores.empty:
                g = fold_scores.groupby(COL_TS_CODE)[list(
                    c for c in fold_scores.columns if c.startswith("win_prob_"))]
                for code, grp in g:
                    for c in grp.columns:
                        fold_prob_map[(code, c)] = float(grp[c].dropna().mean())
            cut_codes = []
            for code, init_prob in holdings.items():
                cur_prob = fold_prob_map.get((code, "win_prob_" + alloc_engine(regime)))
                if cur_prob is None:
                    probs = [v for k, v in fold_prob_map.items() if k[0] == code]
                    cur_prob = float(np.mean(probs)) if probs else None
                if cur_prob is not None and da.alpha_decay_should_cut(init_prob, cur_prob):
                    cut_codes.append(code)
            if cut_codes:
                pre = len(port)
                port = port[~port[COL_TS_CODE].isin(cut_codes)].copy()
                alpha_cut = pre - len(port)
                logger.info("组合层 Alpha Decay 砍仓: 移出 %d 只 %s", alpha_cut, cut_codes[:5])

        if not port.empty:
            for _, row in port.iterrows():
                px = load_price_panel([row[COL_TS_CODE]], val_end)
                if px.empty or len(px) < 2:
                    continue
                px = px.sort_values(COL_TRADE_DATE)
                entry = px.iloc[0]["close"]
                t1_px = px.iloc[1]["close"] if len(px) >= 2 else None
                if t1_px is not None:
                    t1_total += 1
                    if t1_px > entry:
                        t1_win += 1
                d5_px = px.iloc[min(5, len(px) - 1)]["close"] if len(px) >= 2 else None
                if d5_px is not None:
                    d5_total += 1
                    if d5_px > entry:
                        d5_win += 1

        # ── V6 v2 Alpha Decay：跨折持仓评估（HERMES-20260801-004/005）──
        #    上一折建仓的组合，在本折验证期用 T-1 收盘 P(Win) 重估，
        #    P(Win) < 初始×0.75 则砍仓（提前离场，不持有至本折末）。
        #    本折新选出的 port 进入 holdings，供下一折评估。
        #    BUG-1 修复 (HERMES-20260802-001)：holdings 改为 dict {ts_code: init_p}
        #    last-write-wins + 去重，杜绝同股重复条目导致的旧概率漏判衰减。
        holdings_dict: dict[str, float] = {}
        if holdings:
            fold_prob_map = {}
            if not fold_scores.empty:
                g = fold_scores.groupby(COL_TS_CODE)[list(
                    c for c in fold_scores.columns if c.startswith("win_prob_"))]
                for code, grp in g:
                    for c in grp.columns:
                        fold_prob_map[(code, c)] = float(grp[c].dropna().mean())
            for code, init_prob in holdings.items():
                cur_prob = fold_prob_map.get((code, "win_prob_" + alloc_engine(regime)))
                if cur_prob is None:
                    # 引擎列名不匹配时退化为任一 win_prob 均值
                    probs = [v for k, v in fold_prob_map.items() if k[0] == code]
                    cur_prob = float(np.mean(probs)) if probs else None
                if cur_prob is not None and da.alpha_decay_should_cut(init_prob, cur_prob):
                    alpha_cut += 1
                    continue  # 砍仓：不续持
                holdings_dict[code] = init_prob
        # 本折新组合入持有跟踪（dict last-write-wins，同股覆盖为最新建仓概率）
        for _, row in port.iterrows():
            gc = alloc_engine(regime)
            pcol = f"win_prob_{gc}"
            init_p = float(row[pcol]) if pcol in row.index and pd.notna(row[pcol]) else 0.55
            holdings_dict[row[COL_TS_CODE]] = init_p
        holdings = holdings_dict

        fold_records.append({
            "fold": fold_idx,
            "val_start": val_start, "val_end": val_end,
            "regime": regime,
            "n_candidates": len(fold_scores),
            "gate_veto": gate.get("veto", 0),
            "gate_pass": gate.get("pass", 0),
            "portfolio_size": len(port),
            "rank_ic": panel.get("Rank IC", 0.0),
            "rank_ic_unboosted": _unboosted_ic(fold_scores, regime, alloc_engine(regime)),
            "win_rate_topn": panel.get("Actual Win Rate (>55%)", 0.0),
            "avg_vol_topn": panel.get("Avg_Volatility_of_Top5", 0.0),
            "t1_win": t1_win, "t1_total": t1_total,
            "d5_win": d5_win, "d5_total": d5_total,
            "t1_win_rate": round(t1_win / t1_total, 4) if t1_total else None,
            "d5_win_rate": round(d5_win / d5_total, 4) if d5_total else None,
            "t1_risk_high_count": sum(1 for r in t1_summary if r["t1_lock_risk"] >= 0.4),
            "regime_source": regime,
            "gate_blocked_pct": round(gate.get("veto", 0) / max(len(fold_scores), 1), 4),
            "alpha_cut": alpha_cut,
            "vol_breakout_confirmed": int(fold_scores["vol_breakout_confirmed"].sum()) if "vol_breakout_confirmed" in fold_scores.columns else 0,
            "vol_reversion_confirmed": int(fold_scores["vol_reversion_confirmed"].sum()) if "vol_reversion_confirmed" in fold_scores.columns else 0,
        })

        if verbose:
            logger.info("折%02d %s~%s regime=%s RankIC=%.3f t1=%.2f d5=%.2f veto=%d",
                        fold_idx, val_start, val_end, regime,
                        panel.get("Rank IC", 0.0),
                        (t1_win / t1_total if t1_total else 0),
                        (d5_win / d5_total if d5_total else 0),
                        gate.get("veto", 0))

        # ── 组合净值：验证期末尾一次性建仓，持有至下一折（简化等权）──
        #    BUG-3 修复 (HERMES-20260802-001)：同时记录 period 回报与折内日度净值，
        #    后者用于计算真实日度 MaxDD（而非仅 period-to-period）。
        if not port.empty and len(px_guard := load_price_panel(port[COL_TS_CODE].tolist(), val_end)) >= 2:
            entry_nav = float(px_guard.groupby(COL_TS_CODE).first()["close"].mean())
            exit_nav = float(px_guard.groupby(COL_TS_CODE)["close"].agg(lambda s: s.iloc[min(5, len(s)-1)]).mean())
            if entry_nav > 0:
                portfolio_navs.append(exit_nav / entry_nav)
                # 折内日度净值序列（每折的 T+1..T+5 日等权平均净值，相对建仓日）
                daily = px_guard.pivot(index=COL_TRADE_DATE, columns=COL_TS_CODE, values="close")
                if daily.shape[1] > 0 and not daily.empty:
                    daily_nav = daily.mean(axis=1) / entry_nav
                    daily_navs.extend(daily_nav.tolist())

        train_end += eff_step

    if not fold_records:
        return {"error": "无有效折"}

    df_folds = pd.DataFrame(fold_records)

    def _mean(key):
        vals = df_folds[key].dropna()
        return round(float(vals.mean()), 4) if len(vals) else None

    # ── 指标看板汇总 ──
    t1_rate = df_folds["t1_win_rate"].dropna()
    d5_rate = df_folds["d5_win_rate"].dropna()
    # BUG-3 修复 (HERMES-20260802-001)：period-to-period 与 日度 MaxDD 分别计算
    nav = np.cumprod(np.array(portfolio_navs))
    peak = np.maximum.accumulate(nav)
    max_dd_period = float((nav / peak - 1).min()) if len(nav) else 0.0
    daily_nav_arr = np.cumprod(np.maximum(np.array(daily_navs), 1e-9))
    daily_peak = np.maximum.accumulate(daily_nav_arr)
    max_dd_daily = float((daily_nav_arr / daily_peak - 1).min()) if len(daily_nav_arr) else 0.0

    summary = {
        "n_folds": len(fold_records),
        "target_folds": n_folds if n_folds and n_folds > 0 else len(fold_records),
        "date_range": f"{dates[0]} ~ {dates[-1]}",
        "n_dates": n_dates,
        "Strategy_Rank_IC": _mean("rank_ic"),
        "rank_ic_boosted": True,  # BUG-4 透明报告：当前 IC 含 Alpha ×2 boost
        "Strategy_Rank_IC_Unboosted": _mean("rank_ic_unboosted"),
        "rank_ic_positive_folds": round(float((df_folds["rank_ic"] > 0).mean()), 4),
        "Actual_Win_Rate_Verified_T1": round(float(t1_rate.mean()), 4) if len(t1_rate) else None,
        "Actual_Win_Rate_Verified_5D": round(float(d5_rate.mean()), 4) if len(d5_rate) else None,
        "Actual_Win_Rate_Verified_Combined": round(float(pd.concat([t1_rate, d5_rate]).mean()), 4) if len(t1_rate) else None,
        "WinRate_ge55_folds": round(float((df_folds["win_rate_topn"] >= 0.55).mean()), 4),
        "MaxDrawdownOverPeriods": round(max_dd_period, 4),
        "MaxDrawdownDaily": round(max_dd_daily, 4),
        "Portfolio_Avg_Volatility_Top5": _mean("avg_vol_topn"),
        "Avg_T1_Risk_High_Count": round(float(df_folds["t1_risk_high_count"].mean()), 4),
        "regime_distribution": regime_counts,
        "avg_gate_blocked_pct": round(float(df_folds["gate_blocked_pct"].mean()), 4),
        "Alpha_Decay_Cuts_Total": int(df_folds["alpha_cut"].sum()),
        "Alpha_Decay_Cuts_Per_Fold": round(float(df_folds["alpha_cut"].mean()), 4),
        "Vol_Breakout_Confirmed_Total": int(df_folds["vol_breakout_confirmed"].sum()) if "vol_breakout_confirmed" in df_folds.columns else 0,
        "Vol_Reversion_Confirmed_Total": int(df_folds["vol_reversion_confirmed"].sum()) if "vol_reversion_confirmed" in df_folds.columns else 0,
        "elapsed_sec": round(time.time() - t_start, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    daily_df = pd.DataFrame(daily_trades)
    summary["n_daily_trades"] = len(daily_df)
    return {"folds": df_folds, "summary": summary, "daily_trades": daily_df}


# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V6 Walk-Forward Backtest (HERMES-20260801-002)")
    parser.add_argument("--stocks", type=int, default=None, help="股票数（默认 None=全量池约4999）")
    parser.add_argument("--n-folds", type=int, default=98, help="目标折数（-1=全部折）")
    parser.add_argument("--smoke", action="store_true", help="快速冒烟（500只/10折）")
    parser.add_argument("--since", type=str, default="20240101",
                        help="数据起始日期 YYYYMMDD（4年回测用 20200101；半年验证用 20260201）")
    parser.add_argument("--initial-train", type=int, default=120,
                        help="初始训练交易日（半年回测用 60）")
    parser.add_argument("--out", type=str, default="", help="CSV/JSON 输出前缀")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.smoke:
        stocks, n_folds = 500, 10
    else:
        stocks, n_folds = args.stocks, args.n_folds

    t0 = time.time()
    stock_label = f"全量" if stocks is None else f"{stocks}"
    logger.info("① 加载 %s 至今股票日线 (%s 只)...", args.since, stock_label)
    # 全量池扩池(2026-08-05): universe 过滤防杂散 ST/退市行入训练
    from data.stock_industry_mapping import load_stock_universe
    universe = {u["ts_code"] for u in load_stock_universe()}
    stock = load_stock_daily(max_stocks=stocks, min_days=120, since_date=args.since, universe=universe)
    logger.info("   %d 行, %d 只", len(stock), stock[COL_TS_CODE].nunique())

    logger.info("② 加载行业数据...")
    l1_daily, l2_daily, l3_daily = load_industry_data()

    logger.info("③ 构建全量特征矩阵 (%s 至今)...", args.since)
    feat = build_full_feature_matrix(
        stock_daily_df=stock, l1_daily=l1_daily, l2_daily=l2_daily, l3_daily=l3_daily,
        stock_mapping={}, persistence_scores={"l1": {}, "l2": {}})
    logger.info("   %d 行 × %d 列", len(feat), len(feat.columns))

    logger.info("④ %s 折 Walk-Forward 回测...", n_folds)
    result = run_walkforward(
        feat, l1_daily,
        n_folds=n_folds,
        initial_train_days=args.initial_train, val_days=5, step_days=5,
        verbose=args.verbose,
    )
    if "error" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    folds, summary = result["folds"], result["summary"]
    daily_trades = result.get("daily_trades", pd.DataFrame())
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_prefix = args.out or f"backtest/v6_walkforward_report_{datetime.now().strftime('%Y%m%d')}"
    csv_path = f"{out_prefix}.csv"
    folds.to_csv(csv_path, index=False)
    json_path = f"{out_prefix}.json"
    daily_path = f"{out_prefix}_daily_trades.csv"
    if not daily_trades.empty:
        daily_trades.to_csv(daily_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "folds": folds.to_dict("records"),
            "daily_trades": daily_trades.to_dict("records") if not daily_trades.empty else [],
        }, f, ensure_ascii=False, indent=2)
    logger.info("报告已保存: %s / %s", csv_path, json_path)
    print(f"\n📊 报告: {csv_path}")
    if not daily_trades.empty:
        print(f"📊 每日交易明细: {daily_path} ({len(daily_trades)} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
