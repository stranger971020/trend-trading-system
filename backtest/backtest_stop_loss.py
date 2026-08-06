"""
评分止损 + 硬止损双保险 回测 (V4 ML 模型 · 修正版)
===================================================
=== 关键修正：避免前视偏差 ===
用 Walk-Forward 折叠模型逐段回测：
- 对每个时间窗口，只用该窗口之前的数据训练的模型做预测
- 滚动窗口: 训练 100→125→150→... 天，验证 ~25 天

四种方案对比:
  A. 评分止损 only (评分跌出全市场前3% → 卖出)
  B. 评分止损 + 硬止损 15%
  C. 评分止损 + 硬止损 10%
  D. 评分止损 + 硬止损 5%

买入: 每日 ML 评分 Top 5
持有: 每日检查评分排名位置 + 亏损幅度
卖出: 评分跌出前3% | 亏损 > 硬止损% | 持有满 20 日

# ── Changelog ──
# 2026-08-01 Claude: 修复 equity 计算 bug（隐式杠杆）
#               根因: day_ret = mean(活跃仓位收益) 作用于整个净值，
#                     空槽被当成已满仓 → 复利放大 → 5%止损策略 10年虚增至 +4.1M%
#               改动: day_ret = sum(收益)/TOP_N，空槽按现金(0)计入
#               改动: 新增 BACKTEST_WINDOW_DAYS=1000，仅验证最近 4 年
#               改动: 新增交易明细导出 backtest_trades_*.csv（每策略每笔持仓）
#               告警: 历史 CSV（backtest_wf_compare_*.csv）含旧 bug 数据，勿用
# ─────────────
"""

import logging
import os
import pickle
import sqlite3
import sys
import time
from dataclasses import dataclass
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, COL_TS_CODE, COL_TRADE_DATE, COL_CLOSE
from analysis.ml_model import (
    ALL_FEATURES, assign_group, _compute_amount_thresholds,
    compute_forward_labels, train_model_on_window, GROUP_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "lgb_models")
FEATURE_CACHE = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
DB_PATH = os.path.join(DATA_DIR, "sw_index_data.db")

# ── 策略参数 ──
TOP_N = 5
SCORE_STOP_PCT = 3       # 跌出前3%
MAX_HOLD_DAYS = 20
HARD_STOPS = [None, 15, 10, 5]

# ── Walk-Forward 窗口 ──
INITIAL_TRAIN_DAYS = 100
VAL_DAYS = 25
STEP_DAYS = 25

# ── 回测窗口限制（2026-08-01 新增）──
# 仅验证最近 BACKTEST_WINDOW_DAYS 个交易日；None = 全量。
# 用户确认报告回测盒用 1000 天（约4年）窗口。
# 可用环境变量覆盖：BACKTEST_WINDOW_DAYS=60 python3 backtest/backtest_stop_loss.py
BACKTEST_WINDOW_DAYS = int(os.environ.get("BACKTEST_WINDOW_DAYS", "1000") or "1000")

# ── Phase 2 大盘动能过滤扫描（2026-08-01 新增开关）──
# 31 个指标×阈值组合的扫描，4年窗口下约 3-4 小时。
# 报告回测盒只需要 Phase 1 四策略对比，默认关闭。
RUN_PHASE2 = False

# ── 交易成本（A股实际）──
COMMISSION_BUY = 0.00025    # 买入佣金 0.025%
COMMISSION_SELL = 0.00025   # 卖出佣金 0.025%
STAMP_DUTY = 0.0005         # 卖出印花税 0.05%
# 往返总成本: 0.1%


@dataclass
class Position:
    ts_code: str
    entry_date: str
    entry_price: float
    entry_idx: int


@dataclass
class TradeRecord:
    ts_code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    hold_days: int
    return_pct: float
    exit_reason: str
    entry_score_rank_pct: float
    avg_score_rank_pct: float
    max_drawdown_pct: float
    fold: int = 0


# ═══════════════════════════════════════════════════════════════
# 1. 数据准备 + Walk-Forward 折叠训练
# ═══════════════════════════════════════════════════════════════

def prepare_walk_forward():
    """加载数据、生成 Walk-Forward 折叠模型。

    Returns:
        folds_info: [(fold_id, val_start_date, val_end_date, model, feats), ...]
        score_df: 带评分的 DataFrame (跨折叠拼接)
        price_df: 价格矩阵
        all_dates: 所有交易日
    """
    logger.info("=" * 60)
    logger.info("数据准备 & Walk-Forward 训练")
    logger.info("=" * 60)

    # 加载特征矩阵
    feat_df = pd.read_parquet(FEATURE_CACHE)
    feat_df = feat_df.sort_values([COL_TS_CODE, COL_TRADE_DATE])
    logger.info("特征矩阵: %d 行 × %d 列, %d 股票, %d 交易日",
                len(feat_df), len(feat_df.columns),
                feat_df[COL_TS_CODE].nunique(), feat_df[COL_TRADE_DATE].nunique())

    # 加载 amount 用于三分组
    conn = sqlite3.connect(DB_PATH)
    amt_df = pd.read_sql("SELECT ts_code, trade_date, amount FROM stock_daily ORDER BY trade_date", conn)
    conn.close()
    amt_df[COL_TRADE_DATE] = amt_df[COL_TRADE_DATE].astype(str)
    avg_amount = amt_df.groupby(COL_TS_CODE)["amount"].mean().to_dict()
    thresholds = _compute_amount_thresholds(amt_df)

    # 准备价格矩阵
    price_df = feat_df.pivot_table(index=COL_TRADE_DATE, columns=COL_TS_CODE, values=COL_CLOSE, aggfunc="last")
    price_df = price_df.sort_index().ffill()

    all_dates = sorted(feat_df[COL_TRADE_DATE].unique())
    # 2026-08-01 新增: 回测窗口截断，仅验证最近 BACKTEST_WINDOW_DAYS 天
    # （保留 INITIAL_TRAIN_DAYS 天冷启动训练在窗口之前，验证期=目标窗口）
    if BACKTEST_WINDOW_DAYS and len(all_dates) > BACKTEST_WINDOW_DAYS + INITIAL_TRAIN_DAYS:
        all_dates = all_dates[-(BACKTEST_WINDOW_DAYS + INITIAL_TRAIN_DAYS):]
        logger.info("回测窗口截断: 仅验证最近 %d 个交易日（含 %d 天冷启动训练，从 %s 起）",
                    BACKTEST_WINDOW_DAYS, INITIAL_TRAIN_DAYS, all_dates[0])
    n = len(all_dates)
    logger.info("日期范围: %s ~ %s  (%d 天)", all_dates[0], all_dates[-1], n)

    # ── Walk-Forward 折叠 ──
    cfg_micro = GROUP_CONFIG["micro"]
    cfg_mid = GROUP_CONFIG["mid"]
    all_feats = [c for c in ALL_FEATURES if c in feat_df.columns]
    feat_list = list(all_feats)

    folds = []
    train_end = INITIAL_TRAIN_DAYS
    fold_id = 0

    # 缓存训练好的模型避免重复训练
    lgb_cached = {}

    def get_model(group, train_dates_set):
        """获取或训练指定分组和时间窗口的模型。"""
        key = (group, min(train_dates_set), max(train_dates_set))
        if key in lgb_cached:
            return lgb_cached[key]
        train_df = feat_df[feat_df[COL_TRADE_DATE].isin(train_dates_set)]
        if train_df.empty:
            return None

        # 计算前向收益标签
        train_labeled = compute_forward_labels(train_df, forward_days=20)
        train_labeled = train_labeled.dropna(subset=["fwd_return"])
        if train_labeled.empty:
            return None

        feats = [c for c in feat_list if c in train_labeled.columns]
        if len(feats) < 10:
            return None

        model, _ = train_model_on_window(train_labeled, group, feats)
        lgb_cached[key] = model
        return model

    all_score_records = []

    while train_end + VAL_DAYS <= n:
        fold_id += 1
        train_dates = all_dates[:train_end]
        val_dates = all_dates[train_end:train_end + VAL_DAYS]

        logger.info("折叠 %d: 训练 %s~%s (%d天) → 验证 %s~%s (%d天)",
                    fold_id, train_dates[0], train_dates[-1], len(train_dates),
                    val_dates[0], val_dates[-1], len(val_dates))

        # 训练 micro & mid 模型
        micro_model = get_model("micro", set(train_dates))
        mid_model = get_model("mid", set(train_dates))

        if micro_model is None and mid_model is None:
            logger.warning("  折叠 %d: 无可用模型", fold_id)
            train_end += STEP_DAYS
            continue

        # 对验证期每只股票/每日评分
        for date in val_dates:
            day_df = feat_df[feat_df[COL_TRADE_DATE] == date].copy()
            if day_df.empty:
                continue

            scores = {}
            for grp_name, model in [("micro", micro_model), ("mid", mid_model)]:
                if model is None:
                    continue
                grp_mask = day_df[COL_TS_CODE].apply(
                    lambda c: assign_group(avg_amount.get(c, 0), thresholds) == grp_name)
                grp_df = day_df[grp_mask]
                if grp_df.empty:
                    continue
                try:
                    model_feats = [c for c in model.booster_.feature_name() if c in grp_df.columns]
                except Exception:
                    model_feats = [c for c in feat_list if c in grp_df.columns]
                if not model_feats:
                    continue
                X = grp_df[model_feats].fillna(0)
                preds = model.predict(X)
                for j, code in enumerate(grp_df[COL_TS_CODE].values):
                    scores[code] = float(preds[j])

            if not scores:
                continue

            # 排名百分位
            sorted_items = sorted(scores.items(), key=lambda x: -x[1])
            total = len(sorted_items)
            for rank, (code, s) in enumerate(sorted_items):
                rank_pct = (rank + 1) / total * 100
                all_score_records.append({
                    COL_TRADE_DATE: date,
                    COL_TS_CODE: code,
                    "score": round(s, 4),
                    "score_rank_pct": round(rank_pct, 2),
                    "fold": fold_id,
                })

        logger.info("  折叠 %d 完成: 评分 %d 条", fold_id, len(all_score_records))
        train_end += STEP_DAYS

    if not all_score_records:
        logger.error("没有生成任何评分！")
        return None, None, None

    score_df = pd.DataFrame(all_score_records)
    logger.info("总评分矩阵: %d 行, %d 日, %d 股票, %d 折叠",
                len(score_df), score_df[COL_TRADE_DATE].nunique(),
                score_df[COL_TS_CODE].nunique(), score_df["fold"].nunique())

    # 对齐日期
    common = sorted(set(score_df[COL_TRADE_DATE].unique()) & set(price_df.index))
    logger.info("对齐后: %d 交易日", len(common))

    return score_df, price_df, common


# ═══════════════════════════════════════════════════════════════
# 2. 大盘动能指标（用于入场过滤）
# ═══════════════════════════════════════════════════════════════

def compute_market_indicators(db_path: str, dates: list) -> dict:
    """计算每个回测日期的大盘动能指标。

    Returns:
        {date: {"idx_5d": float, "idx_20d": float, "idx_60d": float,
                "above_ma200": bool, "breadth_pct": float}}
    """
    conn = sqlite3.connect(db_path)
    idx = pd.read_sql(
        "SELECT trade_date, AVG(close) as close FROM sw_index_daily GROUP BY trade_date ORDER BY trade_date",
        conn)
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx.set_index("trade_date")["close"]

    # 行业数据（用于广度计算）
    ind = pd.read_sql("SELECT trade_date, ts_code, close FROM sw_index_daily ORDER BY trade_date", conn)
    ind["trade_date"] = ind["trade_date"].astype(str)
    conn.close()

    result = {}
    for d in dates:
        if d not in idx.index:
            continue
        pos = idx.index.get_loc(d)
        info = {}

        # 多周期动量
        for lookback, key in [(5, "idx_5d"), (20, "idx_20d"), (60, "idx_60d")]:
            if pos >= lookback:
                info[key] = (idx.iloc[pos] / idx.iloc[pos - lookback] - 1) * 100

        # MA200 位置
        if pos >= 200:
            ma200 = idx.iloc[pos - 199:pos + 1].mean()
            info["above_ma200"] = idx.iloc[pos] > ma200

        # 行业广度（站上 MA20 的比例）
        day_ind = ind[ind["trade_date"] == d]
        if not day_ind.empty and pos >= 20:
            above = 0
            total = 0
            for c in day_ind["ts_code"].unique():
                c_data = ind[(ind["ts_code"] == c) & (ind["trade_date"] <= d)].sort_values("trade_date")
                if len(c_data) >= 20:
                    total += 1
                    if c_data["close"].iloc[-1] > c_data["close"].iloc[-20:].mean():
                        above += 1
            if total > 0:
                info["breadth_pct"] = above / total * 100

        result[d] = info

    return result


def build_market_filter(indicator: str, threshold: float, market_data: dict) -> dict:
    """从大盘指标构建买入过滤 dict。

    Args:
        indicator: "idx_5d" / "idx_20d" / "idx_60d" / "breadth_pct"
        threshold: 仅当指标 >= threshold 时允许买入
        market_data: compute_market_indicators() 的输出

    Returns:
        {date: bool} 作为 backtest() 的 market_filter 参数
    """
    result = {}
    for d, info in market_data.items():
        val = info.get(indicator)
        if val is not None:
            result[d] = val >= threshold
        else:
            result[d] = False  # 数据不足 → 不买
    return result


# ═══════════════════════════════════════════════════════════════
# 3. 单策略回测
# ═══════════════════════════════════════════════════════════════

def backtest(score_df, price_df, dates, hard_stop=None, label="策略", market_filter=None):
    """运行一个止损策略的回测。

    Args:
        market_filter: dict {date: bool}, 为 True 时允许买入，False 时空仓（已持仓不受影响）。
                       如果为 None，始终允许买入。
    """
    """运行一个止损策略的回测。"""
    positions: dict[str, Position] = {}
    trades: list[TradeRecord] = []

    nav = 1.0
    nav_list = [1.0]
    bench_nav = 1.0
    bench_list = [1.0]
    daily_rets_list = []

    all_codes = set(price_df.columns)

    for idx, date in enumerate(dates):
        day_scores = score_df[score_df[COL_TRADE_DATE] == date]
        if day_scores.empty:
            nav_list.append(nav)
            bench_list.append(bench_nav)
            daily_rets_list.append(0.0)
            continue

        # Top N
        top_n = day_scores.nlargest(TOP_N, "score")[COL_TS_CODE].tolist()

        today_prices = price_df.loc[date] if date in price_df.index else pd.Series(dtype=float)
        yesterday_prices = (price_df.loc[dates[idx-1]] if idx > 0 and dates[idx-1] in price_df.index
                            else pd.Series(dtype=float))

        # ── STEP 1: 检查退出 ──
        to_exit = []
        for code, pos in positions.items():
            # 评分退出
            score_row = day_scores[day_scores[COL_TS_CODE] == code]
            if score_row.empty:
                to_exit.append((code, "score_stop"))
                continue
            rank_pct = score_row.iloc[0]["score_rank_pct"]
            if rank_pct > SCORE_STOP_PCT:
                to_exit.append((code, "score_stop"))
                continue

            # 硬止损
            if hard_stop is not None and code in today_prices.index:
                cp = today_prices[code]
                if not np.isnan(cp) and pos.entry_price > 0:
                    loss = (cp / pos.entry_price - 1) * 100
                    if loss <= -hard_stop:
                        to_exit.append((code, "hard_stop"))
                        continue

            # 持有满 20 日
            hold_days = idx - pos.entry_idx
            if hold_days >= MAX_HOLD_DAYS:
                to_exit.append((code, "max_hold"))

        # 执行卖出
        for code, reason in to_exit:
            pos = positions.pop(code)
            exit_price = today_prices.get(code, np.nan)
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos.entry_price

            ret = (exit_price / pos.entry_price - 1) * 100
            hold_days = idx - pos.entry_idx

            # 最大回撤
            pos_slice = price_df.loc[pos.entry_date:date, code].dropna()
            mdd = (pos_slice / pos_slice.expanding().max() - 1).min() * 100 if len(pos_slice) > 1 else 0.0

            # 平均评分排名
            pos_scores = score_df[(score_df[COL_TS_CODE] == code) &
                                   (score_df[COL_TRADE_DATE] >= pos.entry_date) &
                                   (score_df[COL_TRADE_DATE] <= date)]
            avg_rank = pos_scores["score_rank_pct"].mean() if len(pos_scores) > 0 else 0

            trades.append(TradeRecord(
                ts_code=code, entry_date=pos.entry_date, exit_date=date,
                entry_price=pos.entry_price, exit_price=exit_price,
                hold_days=hold_days, return_pct=round(ret, 2),
                exit_reason=reason, entry_score_rank_pct=0,
                avg_score_rank_pct=round(avg_rank, 2), max_drawdown_pct=round(mdd, 2),
            ))

        # ── STEP 2: 开新仓 ──
        # 大盘动能过滤: 弱势日不买入（已持有个股仍按止损规则管理）
        allow_buy = True
        if market_filter is not None and not market_filter.get(date, True):
            allow_buy = False

        # 避免同日卖出+买回: hard_stop/max_hold 卖出的股票当天不买回
        exited_today = {code for code, reason in to_exit}
        held = set(positions.keys())

        if allow_buy:
            for code in top_n:
                if code in exited_today:
                    continue
                if code in held or code not in all_codes:
                    continue
                price = today_prices.get(code, np.nan)
                if np.isnan(price) or price <= 0:
                    continue
                positions[code] = Position(ts_code=code, entry_date=date, entry_price=price, entry_idx=idx)

        # ── STEP 3: 计算当日组合收益 ──
        day_ret = 0.0
        # 当日交易成本（买入佣金 + 卖出佣金+印花税）
        cost_today = 0.0
        # 卖出成本（每只卖出的股票占 1/N 权重）
        if to_exit and len(positions) + len(to_exit) > 0:
            cost_today += len(to_exit) * (COMMISSION_SELL + STAMP_DUTY) / TOP_N
        # 买入成本（新开仓的股票）
        cost_today += sum(1 for code in top_n if code not in exited_today and code not in held and code in all_codes) * COMMISSION_BUY / TOP_N
        active = list(positions.keys())
        if active and idx > 0:
            rets = []
            for code in active:
                pos = positions[code]
                if pos.entry_date == date:  # 当天新买，不贡献收益
                    continue
                if code in today_prices.index and code in yesterday_prices.index:
                    tc, yc = today_prices[code], yesterday_prices[code]
                    if not np.isnan(tc) and not np.isnan(yc) and yc > 0:
                        rets.append(tc / yc - 1)
            if rets:
                # 2026-08-01 修复: 原 np.mean(rets) 把空槽当已满仓(隐式杠杆)。
                # 正确: 每槽 1/TOP_N 权重，空槽按现金(0收益)计入组合。
                day_ret = float(sum(rets)) / TOP_N

        # 基准
        bench_ret = 0.0
        if idx > 0:
            common = today_prices.index.intersection(yesterday_prices.index)
            if len(common) > 0:
                bench_ret = float((today_prices[common] / yesterday_prices[common] - 1).mean())

        nav *= (1 + day_ret - cost_today)
        bench_nav *= (1 + bench_ret)
        nav_list.append(nav)
        bench_list.append(bench_nav)
        daily_rets_list.append(day_ret)

    # 最后平仓
    last_date = dates[-1]
    last_prices = price_df.loc[last_date] if last_date in price_df.index else pd.Series(dtype=float)
    for code in list(positions.keys()):
        pos = positions.pop(code)
        exit_price = last_prices.get(code, np.nan)
        if np.isnan(exit_price):
            exit_price = pos.entry_price
        ret = (exit_price / pos.entry_price - 1) * 100
        hold_days = len(dates) - 1 - pos.entry_idx
        trades.append(TradeRecord(
            ts_code=code, entry_date=pos.entry_date, exit_date=last_date,
            entry_price=pos.entry_price, exit_price=exit_price,
            hold_days=hold_days, return_pct=round(ret, 2),
            exit_reason="end", entry_score_rank_pct=0,
            avg_score_rank_pct=0, max_drawdown_pct=0,
        ))

    dt_idx = pd.to_datetime([dates[0]] + dates)
    nav_s = pd.Series(nav_list, index=dt_idx)
    bench_s = pd.Series(bench_list, index=dt_idx)
    daily_s = pd.Series(daily_rets_list, index=pd.to_datetime(dates))

    metrics = compute_metrics(nav_s, bench_s, daily_s, hard_stop)
    return nav_s, trades, metrics


# ═══════════════════════════════════════════════════════════════
# 3. 指标 & 分析
# ═══════════════════════════════════════════════════════════════

def compute_metrics(nav, bench_nav, daily_returns, hard_stop):
    rets = daily_returns.dropna()
    if len(rets) < 5:
        return {}
    total_ret = float(nav.iloc[-1] - 1)
    bench_total = float(bench_nav.iloc[-1] - 1)
    n_years = len(rets) / 252
    ann_ret = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0
    ann_vol = float(rets.std() * np.sqrt(252))
    excess_r = rets - 0.02 / 252
    sharpe = float(excess_r.mean() / excess_r.std() * np.sqrt(252)) if excess_r.std() > 0 else 0
    peak = nav.expanding().max()
    dd = (nav - peak) / peak
    max_dd = float(dd.min()) * 100
    calmar = ann_ret / abs(max_dd / 100) if abs(max_dd) > 0.001 else 0
    win_rate_d = float((rets > 0).mean())
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rate_m = float((monthly > 0).mean()) if len(monthly) > 0 else 0
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pl_ratio = float(wins.mean() / abs(losses.mean())) if len(losses) > 0 and losses.mean() != 0 else 0
    return {
        "strategy": f"评分止损{'only' if hard_stop is None else f'+硬{hard_stop:.0f}%'}",
        "hard_stop": str(hard_stop) if hard_stop else "none",
        "total_return_pct": round(total_ret * 100, 2),
        "benchmark_return_pct": round(bench_total * 100, 2),
        "excess_return_pct": round((total_ret - bench_total) * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "annual_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_daily_pct": round(win_rate_d * 100, 1),
        "win_rate_monthly_pct": round(win_rate_m * 100, 1),
        "profit_loss_ratio": round(pl_ratio, 2),
        "n_trading_days": len(rets),
    }


def analyze_trades(trades):
    if not trades:
        return {}
    df = pd.DataFrame([t.__dict__ for t in trades])
    total = len(df)
    win = df[df["return_pct"] > 0]
    loss = df[df["return_pct"] <= 0]
    wr = len(win) / total * 100
    avg_ret = df["return_pct"].mean()
    avg_w = win["return_pct"].mean() if len(win) > 0 else 0
    avg_l = loss["return_pct"].mean() if len(loss) > 0 else 0
    pl_ratio_val = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
    reason_dist = df["exit_reason"].value_counts().to_dict()
    reason_perf = df.groupby("exit_reason")["return_pct"].agg(["mean", "count"]).to_dict("index")
    return {
        "total_trades": total,
        "win_rate_pct": round(wr, 1),
        "avg_return_pct": round(avg_ret, 2),
        "avg_win_pct": round(avg_w, 2),
        "avg_loss_pct": round(avg_l, 2),
        "profit_loss_ratio": round(pl_ratio_val, 2),
        "avg_hold_days": round(df["hold_days"].mean(), 1),
        "max_hold_days": round(df["hold_days"].max(), 1),
        "exit_reasons": reason_dist,
        "exit_performance": reason_perf,
    }


# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Walk-Forward 数据准备
    score_df, price_df, common_dates = prepare_walk_forward()
    if score_df is None:
        logger.error("数据准备失败")
        return

    score_df = score_df[score_df[COL_TRADE_DATE].isin(common_dates)]

    # ═══════════════════════════════════════════════════════════
    # Phase 1: 基准回测（4种止损方案）
    # ═══════════════════════════════════════════════════════════
    results = []
    all_navs = {}
    all_trades = {}

    for hs in HARD_STOPS:
        label = f"评分止损{'only' if hs is None else f'+硬{hs:.0f}%'}"
        logger.info("=" * 60)
        logger.info("▶ 策略: %s", label)
        logger.info("=" * 60)

        nav, trades, m = backtest(score_df, price_df, common_dates, hard_stop=hs, label=label)
        if m:
            results.append(m)
            all_navs[label] = nav
            all_trades[label] = trades
            logger.info("  总收益 %+.2f%% | 年化 %+.2f%% | Sharpe %.3f | MaxDD %.2f%% | 交易 %d",
                        m["total_return_pct"], m["annual_return_pct"],
                        m["sharpe_ratio"], m["max_drawdown_pct"], len(trades))
        else:
            logger.error("  失败!")

    print("\n" + "=" * 82)
    print("  评分止损 + 硬止损 · Walk-Forward 回测对比（无前视偏差）")
    print("=" * 82)
    print(f"  回测区间: {common_dates[0]} ~ {common_dates[-1]}  ({len(common_dates)} 交易日)")
    print(f"  训练窗口: 初始 {INITIAL_TRAIN_DAYS}天 → +{STEP_DAYS}天滑动")
    print(f"  买入: Top {TOP_N} | 评分止损: 前{SCORE_STOP_PCT}% | 最长持有: {MAX_HOLD_DAYS}日")
    print("=" * 82)
    print(f"  {'策略':<22} {'总收益%':>9} {'年化%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'日胜率%':>7} {'盈亏比':>7}")
    print("-" * 82)

    for r in sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True):
        print(f"  {r['strategy']:<22} {r['total_return_pct']:>+9.2f} {r['annual_return_pct']:>+7.2f} "
              f"{r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>7.2f}% "
              f"{r['win_rate_daily_pct']:>6.1f}% {r['profit_loss_ratio']:>6.2f}")

    print("-" * 82)
    if results:
        print(f"  {'基准(等权)':<22} {results[0]['benchmark_return_pct']:>+9.2f}%")
    print("=" * 82)

    best_s = max(results, key=lambda x: x["sharpe_ratio"]) if results else None
    if best_s:
        print(f"\n  ★ Sharpe 最优: {best_s['strategy']}  "
              f"Sharpe={best_s['sharpe_ratio']:.3f}  总收益={best_s['total_return_pct']:+.2f}%  最大回撤={best_s['max_drawdown_pct']:.2f}%")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 大盘动能过滤回测（仅用最优止损方案的硬5%）
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 2: 大盘动能入场过滤回测")
    logger.info("=" * 60)

    if RUN_PHASE2:
        market_data = compute_market_indicators(DB_PATH, common_dates)
        logger.info("大盘指标计算完成: %d 个日期", len(market_data))
    else:
        logger.info("Phase 2 已跳过（RUN_PHASE2=False，仅保留 Phase 1 四策略对比）")
        market_data = {}

    HARD_STOP = 5  # 最优硬止损

    # 指标 → 阈值候选列表
    SWEEP_CONFIG = {
        "idx_5d":    list(range(-5, 6, 1)),     # -5% ~ +5%
        "idx_20d":   list(range(-10, 11, 2)),   # -10% ~ +10%
        "idx_60d":   list(range(-15, 16, 5)),   # -15% ~ +15%
        "breadth_pct": list(range(20, 81, 10)),  # 20% ~ 80%
    }

    sweep_results = []
    if RUN_PHASE2:
        for indicator, thresholds in SWEEP_CONFIG.items():
            for th in thresholds:
                mf = build_market_filter(indicator, th, market_data)
                nav, trades, m = backtest(
                    score_df, price_df, common_dates,
                    hard_stop=HARD_STOP,
                    label=f"{indicator}>={th}",
                    market_filter=mf,
                )
                if not m:
                    continue
                # 计算持仓天数占比
                buy_days = sum(1 for v in mf.values() if v)
                pct_invested = buy_days / len(mf) * 100 if mf else 100
                sweep_results.append({
                    "indicator": indicator,
                    "threshold": th,
                    "total_return_pct": m["total_return_pct"],
                    "annual_return_pct": m["annual_return_pct"],
                    "sharpe_ratio": m["sharpe_ratio"],
                    "max_drawdown_pct": m["max_drawdown_pct"],
                    "n_trades": len(trades),
                    "buy_days": buy_days,
                    "total_days": len(mf) if mf else 0,
                    "pct_invested": round(pct_invested, 1),
                })

    # ── 输出 ──
    sweep_df = pd.DataFrame(sweep_results)
    if not sweep_df.empty:
        # 按总收益排序
        sweep_df = sweep_df.sort_values("total_return_pct", ascending=False)

        print("\n" + "=" * 90)
        print("  大盘动能入场过滤 · 回测对比（硬5%止损 + Walk-Forward）")
        print("=" * 90)
        print(f"  仅当指标 >= 阈值时买入 Top 5，否则空仓等权现金")
        print("-" * 90)
        print(f"  {'指标':<12} {'阈值':>6} {'总收益%':>8} {'年化%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'交易':>5} {'持仓%':>7}")
        print("-" * 90)

        # 展示每个指标的最优 + Top 3
        shown = set()
        for indicator in SWEEP_CONFIG:
            subset = sweep_df[sweep_df["indicator"] == indicator]
            if subset.empty:
                continue
            best_row = subset.iloc[0]
            shown.add((indicator, best_row["threshold"]))
            print(f"  {indicator:<12} {best_row['threshold']:>6.0f} {best_row['total_return_pct']:>+8.2f} "
                  f"{best_row['annual_return_pct']:>+7.2f} {best_row['sharpe_ratio']:>7.3f} "
                  f"{best_row['max_drawdown_pct']:>6.2f}% {best_row['n_trades']:>5} "
                  f"{best_row['pct_invested']:>6.1f}%"
                  + ("  ← 最优" if best_row["indicator"] == indicator else ""))

        # 全局 Top 5
        print("-" * 90)
        print(f"\n  全局 Top 10（按总收益排序）:")
        print(f"  {'指标':<12} {'阈值':>6} {'总收益%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'持仓%':>7}")
        print(f"  {'-'*60}")
        for _, row in sweep_df.head(10).iterrows():
            print(f"  {row['indicator']:<12} {int(row['threshold']):>6} {row['total_return_pct']:>+8.2f} "
                  f"{row['sharpe_ratio']:>7.3f} {row['max_drawdown_pct']:>6.2f}% "
                  f"{row['pct_invested']:>6.1f}%")

        # 对比基准
        base = [r for r in results if r["hard_stop"] == "5"]
        if base:
            print(f"\n  基准（始终买入）:  总收益 {base[0]['total_return_pct']:+.2f}%  "
                  f"Sharpe {base[0]['sharpe_ratio']:.3f}  MaxDD {base[0]['max_drawdown_pct']:.2f}%  "
                  f"持仓 100%")

        print("=" * 90)

        # 保存
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        combo_df = pd.DataFrame(results)
        combo_df.to_csv(f"backtest_compare_{timestamp}.csv", index=False)
        sweep_df.to_csv(f"backtest_market_filter_sweep_{timestamp}.csv", index=False)
        logger.info("结果已保存")

    # ═══════════════════════════════════════════════════════════
    # 交易分析（略，仅打印基准4方案）
    # ═══════════════════════════════════════════════════════════

    # 交易分析
    print("\n" + "-" * 82)
    print("  交易分析")
    print("-" * 82)
    for hs in HARD_STOPS:
        label = f"评分止损{'only' if hs is None else f'+硬{hs:.0f}%'}"
        trades = all_trades.get(label, [])
        a = analyze_trades(trades)
        if not a:
            continue
        print(f"\n  【{label}】")
        print(f"    交易: {a['total_trades']} 次 | 胜率: {a['win_rate_pct']}% | "
              f"平均收益: {a['avg_return_pct']:+.2f}%")
        print(f"    盈利: {a['avg_win_pct']:+.2f}% / 亏损: {a['avg_loss_pct']:+.2f}% | "
              f"盈亏比: {a['profit_loss_ratio']} | 持有: {a['avg_hold_days']}天")
        for r, c in sorted(a['exit_reasons'].items(), key=lambda x: -x[1]):
            perc = a['exit_performance'].get(r, {})
            print(f"      ├ {r}: {int(c):>3}次, 平均 {perc.get('mean', 0):+.2f}%")

    # ── 导出每笔持仓明细（2026-08-01 新增）──
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    for label, trades in all_trades.items():
        if not trades:
            continue
        safe = label.replace(" ", "_").replace("+", "p").replace("%", "pct")
        rows = [{
            "strategy": label,
            "ts_code": t.ts_code,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "hold_days": t.hold_days,
            "return_pct": t.return_pct,
            "exit_reason": t.exit_reason,
            "entry_score_rank_pct": t.entry_score_rank_pct,
            "avg_score_rank_pct": t.avg_score_rank_pct,
            "max_drawdown_pct": t.max_drawdown_pct,
            "fold": getattr(t, "fold", 0),
        } for t in trades]
        pd.DataFrame(rows).to_csv(f"backtest_trades_{timestamp}_{safe}.csv", index=False)
        logger.info("交易明细已导出: backtest_trades_%s_%s.csv (%d 笔)", timestamp, safe, len(rows))

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.1f} 秒")

    pd.DataFrame(results).to_csv(f"backtest_wf_compare_{timestamp}.csv", index=False)
    logger.info("结果已保存")

    return results, all_navs, all_trades


if __name__ == "__main__":
    main()
