#!/usr/bin/env python3
from __future__ import annotations
"""
ML 特征数据集 + 模型重训练 V4 — 增量更新

=== 特征数据集（主产物）===
全量 118 个特征，8 个数据源，按交易日增量更新。
特征矩阵持久化为 feature_matrix_v4.parquet，每次只追加新日期。

=== 用法 ===
  # 特征数据集管理
  python3 analysis/retrain_ml.py --rebuild              # 全量重建（首次或强制）
  python3 analysis/retrain_ml.py --update               # 增量更新（追加新日期）
  python3 analysis/retrain_ml.py --status               # 查看数据集状态

  # 训练（基于当前特征数据集）
  python3 analysis/retrain_ml.py --train                # 用当前数据集训练
  python3 analysis/retrain_ml.py --train --show-data    # 训练 + 特征分布

  # 快速测试
  python3 analysis/retrain_ml.py --stocks 500 --days 120 --rebuild

── Changelog ──
# 2026-08-05 Claude: 全量池扩池 — load_stock_daily max_stocks=None=全量(移除LIMIT)
#               build_feature_dataset 按 universe(全量映射) 过滤, 防杂散 ST/退市行入训练
#               --stocks 默认 None
─────────────
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, DATA_DIR, COL_TS_CODE, COL_TRADE_DATE
from data.stock_industry_mapping import load_stock_industry_mapping
from analysis.ml_model import (
    build_full_feature_matrix,
    train_model,
    walk_forward_train,
    save_walk_forward_report,
    save_model,
    load_model,
    GROUP_CONFIG,
    MODEL_PATHS,
    ALL_FEATURES,
    _compute_amount_thresholds,
)
from analysis.module2_persistence import compute_persistence

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("retrain_ml")

CACHE_PATH = os.path.join(DATA_DIR, "feature_matrix_v4.parquet")
LOOKBACK_BUFFER = 60  # 滚动特征需要的历史缓冲天数


# ═══════════════════════════════════════════════════════════════
# 个股模型新鲜度（每日增量训练核心，2026-07-31 新增）
# ═══════════════════════════════════════════════════════════════

def _compute_group_column(feature_df: pd.DataFrame) -> pd.DataFrame:
    """按全市场成交额三分位给特征矩阵打 group 标签（micro/mid/large）。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        all_amounts = pd.read_sql_query(
            "SELECT ts_code, AVG(amount) as avg_amt FROM stock_daily "
            "WHERE amount > 0 GROUP BY ts_code", conn
        )
    finally:
        conn.close()
    amounts = all_amounts.set_index(COL_TS_CODE)["avg_amt"]
    thresh_lo, thresh_hi = amounts.quantile(1 / 3), amounts.quantile(2 / 3)
    group_map = {
        code: ("micro" if amt <= thresh_lo else ("mid" if amt <= thresh_hi else "large"))
        for code, amt in amounts.items()
    }
    feature_df = feature_df.copy()
    feature_df["group"] = feature_df[COL_TS_CODE].map(group_map)
    return feature_df


def load_feature_dataset(cache_path: str = CACHE_PATH) -> pd.DataFrame | None:
    """加载特征矩阵缓存（全量，约 6.5M 行）。"""
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        logger.info("加载特征数据集: %d 行 × %d 列", len(df), len(df.columns))
        return df
    return None


def get_stock_model_training_date() -> str | None:
    """个股模型训练日期 — 优先 walk_forward_report.json 的 timestamp；
    报告缺失时回退到 lgb_micro.pkl 文件 mtime（仅日期，近似值）。"""
    report_path = os.path.join(DATA_DIR, "walk_forward_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                report = json.load(f)
            ts = report.get("timestamp", "")
            if ts:
                return str(ts)[:10]
        except Exception:
            pass
    pkl_path = MODEL_PATHS.get("micro")
    if pkl_path and os.path.exists(pkl_path):
        return pd.Timestamp.fromtimestamp(os.path.getmtime(pkl_path)).strftime("%Y-%m-%d")
    return None


def ensure_stock_model_fresh(
    feature_df: pd.DataFrame | None = None,
    max_stale_days: int = 3,
    force: bool = False,
) -> bool:
    """检查个股模型新鲜度，落后最新数据超过阈值则快速重训最终模型。

    每日增量训练路径：只拟合最终模型（skip_validation=True，约 2-5 分钟），
    98 折验证指标不变——由每周六 `--walk-forward` 全量刷新。
    返回是否实际重训练。

    注意：调用前须确保特征矩阵已通过 `--update` 刷新到最新交易日，
    否则重训基于过期特征（新鲜度链依赖特征矩阵先刷新）。
    """
    from analysis.ml_common import get_latest_stock_daily_date
    latest = get_latest_stock_daily_date()

    if not force and latest is not None:
        train_date_str = get_stock_model_training_date()
        if train_date_str:
            train_date = pd.to_datetime(train_date_str)
            days_stale = (latest - train_date).days
            if days_stale <= max_stale_days:
                logger.info("个股模型新鲜（差 %d 天），跳过重训练", days_stale)
                return False

    # ── 过期（或强制）→ 快速重训最终模型 ──
    if feature_df is None:
        feature_df = load_feature_dataset()
    if feature_df is None or feature_df.empty:
        logger.error("个股模型重训练: 特征矩阵不可用")
        return False

    if "group" not in feature_df.columns:
        feature_df = _compute_group_column(feature_df)

    retrained_any = False
    for grp in ["micro", "mid"]:
        result = walk_forward_train(
            feature_df, group=grp,
            initial_train_days=150, val_days=25, step_days=25,
            skip_validation=True,
        )
        if "error" in result:
            logger.warning("个股 %s 快速重训练失败: %s", grp, result["error"])
            continue
        if result.get("final_model"):
            save_model(result["final_model"], grp)
            logger.info("个股 %s 快速重训练完成", grp)
            retrained_any = True

    if retrained_any:
        logger.info("个股模型快速重训练完成（98 折验证指标保留上次全量结果）")
    return retrained_any


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_stock_daily(max_stocks: int | None = None, min_days: int = 120,
                      since_date=None, universe: set | None = None) -> pd.DataFrame:
    """从 DB 加载个股日线。

    Args:
        max_stocks: 最多加载股票数；None=全量（无 LIMIT，全量池扩池 2026-08-05）
        since_date: 若指定，只加载该日期之后的数据（增量模式）
        universe: 若指定，只保留该股票集合内的股票（防杂散 ST/退市行入训练）
    """
    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT ts_code, COUNT(*) as cnt FROM stock_daily GROUP BY ts_code HAVING cnt >= ?"
    params: list = [min_days]
    if max_stocks is not None:
        sql += " ORDER BY cnt DESC LIMIT ?"
        params.append(max_stocks)
    cur = conn.execute(sql, params)
    top_codes = [r[0] for r in cur.fetchall()]
    if universe:
        top_codes = [c for c in top_codes if c in universe]
    if not top_codes:
        conn.close()
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in top_codes)

    if since_date:
        query = (
            f"SELECT * FROM stock_daily WHERE ts_code IN ({placeholders}) "
            f"AND trade_date >= ? ORDER BY ts_code, trade_date"
        )
        params = top_codes + [since_date]
    else:
        query = (
            f"SELECT * FROM stock_daily WHERE ts_code IN ({placeholders}) "
            f"ORDER BY ts_code, trade_date"
        )
        params = top_codes

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    logger.info("stock_daily: %d 只股票, %d 行%s",
                len(top_codes), len(df),
                f" (>= {since_date})" if since_date else "")
    return df


def load_industry_data() -> tuple:
    conn = sqlite3.connect(DB_PATH)
    l1 = pd.read_sql_query("SELECT * FROM sw_index_daily ORDER BY ts_code, trade_date", conn)
    l2 = pd.read_sql_query("SELECT * FROM sw_l2_index_daily ORDER BY ts_code, trade_date", conn)
    l3 = pd.DataFrame()
    try:
        l3 = pd.read_sql_query(
            "SELECT * FROM sw_l3_index_daily WHERE trade_date >= '20250101' "
            "ORDER BY ts_code, trade_date", conn
        )
    except Exception as e:
        logger.warning("L3 数据加载失败: %s", e)
    conn.close()
    logger.info("行业数据: L1=%d, L2=%d, L3=%d", len(l1), len(l2), len(l3))
    return l1, l2, l3


def load_margin_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM margin_cache WHERE ts_code NOT LIKE '51%' "
            "AND ts_code NOT LIKE '52%' AND ts_code NOT LIKE '56%' "
            "AND ts_code NOT LIKE '58%' AND ts_code NOT LIKE '92%' "
            "ORDER BY ts_code, trade_date", conn
        )
        logger.info("融资融券: %d 行, %d 只个股",
                     len(df), df[COL_TS_CODE].nunique() if not df.empty else 0)
    except Exception as e:
        logger.warning("融资融券加载失败: %s", e)
        df = pd.DataFrame()
    conn.close()
    return df


def load_moneyflow_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM moneyflow_cache ORDER BY ts_code, trade_date", conn
        )
        logger.info("资金流向: %d 行, %d 只", len(df), df[COL_TS_CODE].nunique() if not df.empty else 0)
    except Exception as e:
        logger.warning("资金流向加载失败: %s", e)
        df = pd.DataFrame()
    conn.close()
    return df


def load_fundamental_data() -> tuple:
    conn = sqlite3.connect(DB_PATH)
    fund, fq = pd.DataFrame(), pd.DataFrame()
    try:
        fund = pd.read_sql_query(
            "SELECT * FROM fundamental_cache ORDER BY ts_code, trade_date", conn
        )
        logger.info("基本面: %d 行, %d 只", len(fund), fund[COL_TS_CODE].nunique() if not fund.empty else 0)
    except Exception as e:
        logger.warning("基本面加载失败: %s", e)
    try:
        fq = pd.read_sql_query(
            "SELECT * FROM financial_quality_cache ORDER BY ts_code, end_date", conn
        )
        logger.info("财务质量: %d 行, %d 只", len(fq), fq[COL_TS_CODE].nunique() if not fq.empty else 0)
    except Exception as e:
        logger.warning("财务质量加载失败: %s", e)
    conn.close()
    return fund, fq


def load_market_index() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM sw_index_daily WHERE ts_code = '000300.SH' "
            "ORDER BY trade_date", conn
        )
        if df.empty:
            df = pd.read_sql_query(
                "SELECT * FROM sw_index_daily ORDER BY trade_date LIMIT 1", conn
            )
        logger.info("市场基准: %d 行", len(df))
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


# ═══════════════════════════════════════════════════════════════
# 特征数据集构建（全量）
# ═══════════════════════════════════════════════════════════════

def build_feature_dataset(
    max_stocks: int | None = None,
    min_days: int = 120,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建/增量更新特征数据集。

    Args:
        max_stocks: 最多股票数；None=全量（全量池扩池 2026-08-05）
        existing_df: 已有缓存数据集。若提供，只追加新日期。
    """
    mapping = load_stock_industry_mapping()
    universe = set(mapping.keys())
    logger.info("行业映射: %d 条（全量池 universe 过滤）", len(universe))

    # ── 确定增量范围 ──
    if existing_df is not None and not existing_df.empty:
        last_date = existing_df[COL_TRADE_DATE].max()
        logger.info("当前数据集最后日期: %s，增量追加 > %s", last_date, last_date)

        # 滚动特征需要历史缓冲，多取 LOOKBACK_BUFFER 天
        since = _date_add(last_date, -LOOKBACK_BUFFER)
        stock_df = load_stock_daily(max_stocks, min_days, since_date=since, universe=universe)
    else:
        stock_df = load_stock_daily(max_stocks, min_days, universe=universe)

    # ── 加载行业、外部数据 ──
    l1_daily, l2_daily, l3_daily = load_industry_data()
    margin_df = load_margin_data()
    moneyflow_df = load_moneyflow_data()
    fund_df, fq_df = load_fundamental_data()
    market_df = load_market_index()

    # 持续性评分（缓存无关，全量计算）
    logger.info("计算行业持续性评分...")
    l1_p = compute_persistence(l1_daily) if not l1_daily.empty else pd.DataFrame()
    l2_p = compute_persistence(l2_daily) if not l2_daily.empty else pd.DataFrame()
    persist = {
        "l1": dict(zip(l1_p[COL_TS_CODE], l1_p["persistence_score"])) if not l1_p.empty else {},
        "l2": dict(zip(l2_p[COL_TS_CODE], l2_p["persistence_score"])) if not l2_p.empty else {},
    }

    # ── 全量特征流水线 ──
    logger.info("=" * 60)
    logger.info("构建特征矩阵 V4")
    logger.info("=" * 60)
    t0 = time.time()

    new_features = build_full_feature_matrix(
        stock_daily_df=stock_df,
        l1_daily=l1_daily,
        l2_daily=l2_daily,
        l3_daily=l3_daily,
        stock_mapping=mapping,
        persistence_scores=persist,
        market_index_df=market_df,
        margin_df=margin_df,
        moneyflow_df=moneyflow_df,
        fundamental_df=fund_df,
        financial_quality_df=fq_df,
    )

    t1 = time.time()
    logger.info("特征矩阵: %d 行 × %d 列 (%.1f 秒)", len(new_features), len(new_features.columns), t1 - t0)

    # ── 增量合并 ──
    if existing_df is not None and not existing_df.empty:
        last_date = existing_df[COL_TRADE_DATE].max()

        # 只保留新日期（过滤缓冲前置天数）
        new_dates = new_features[new_features[COL_TRADE_DATE] > last_date]
        if new_dates.empty:
            logger.info("无新数据，数据集已是最新 (%s)", last_date)
            return existing_df

        logger.info("新特征: %d 行 (%s ~ %s)",
                     len(new_dates), new_dates[COL_TRADE_DATE].min(), new_dates[COL_TRADE_DATE].max())

        # 合并：保留旧数据 + 新日期
        old_data = existing_df[existing_df[COL_TRADE_DATE] <= last_date]
        feature_df = pd.concat([old_data, new_dates], ignore_index=True)
        feature_df = feature_df.sort_values([COL_TS_CODE, COL_TRADE_DATE]).reset_index(drop=True)
    else:
        feature_df = new_features

    # ── 保存缓存 ──
    try:
        feature_df.to_parquet(CACHE_PATH, index=False)
        size_mb = os.path.getsize(CACHE_PATH) / 1e6
        logger.info("数据集已保存: %s (%.1f MB, %d 行, %s ~ %s)",
                     CACHE_PATH, size_mb, len(feature_df),
                     feature_df[COL_TRADE_DATE].min(), feature_df[COL_TRADE_DATE].max())
    except Exception as e:
        logger.warning("保存失败: %s", e)

    return feature_df


# ═══════════════════════════════════════════════════════════════
# 训练（基于特征数据集）
# ═══════════════════════════════════════════════════════════════

def train_from_dataset(feature_df: pd.DataFrame, show_data: bool = False):
    """基于特征数据集训练三分组模型。"""
    if feature_df is None or feature_df.empty:
        logger.error("特征数据集为空，无法训练")
        return

    # ── 分组 ──
    logger.info("=" * 60)
    logger.info("三分组")
    logger.info("=" * 60)

    # 从原始 DB 获取成交额阈值
    conn = sqlite3.connect(DB_PATH)
    all_amounts = pd.read_sql_query(
        "SELECT ts_code, AVG(amount) as avg_amt FROM stock_daily "
        "WHERE amount > 0 GROUP BY ts_code", conn
    )
    conn.close()
    amounts = all_amounts.set_index(COL_TS_CODE)["avg_amt"]
    thresh_lo, thresh_hi = amounts.quantile(1/3), amounts.quantile(2/3)
    group_map = {}
    for code, amt in amounts.items():
        if amt <= thresh_lo:
            group_map[code] = "micro"
        elif amt <= thresh_hi:
            group_map[code] = "mid"
        else:
            group_map[code] = "large"
    feature_df["group"] = feature_df[COL_TS_CODE].map(group_map)
    logger.info("分组: micro=%d, mid=%d, large=%d",
                (feature_df["group"] == "micro").sum(),
                (feature_df["group"] == "mid").sum(),
                (feature_df["group"] == "large").sum())

    # ── 特征分布 ──
    if show_data:
        logger.info("\n特征统计 (分组合并):")
        for c in ALL_FEATURES:
            if c in feature_df.columns:
                s = feature_df[c]
                nonzero = (s.fillna(0) != 0).sum()
                logger.info("  %-30s nonzero=%5.1f%%  mean=%8.4f  std=%8.4f",
                            c, nonzero / len(s) * 100, s.mean(), s.std())

    # ── 训练 ──
    logger.info("=" * 60)
    logger.info("三分组训练")
    logger.info("=" * 60)

    models = {}
    for grp in ["micro", "mid"]:
        gdf = feature_df[feature_df["group"] == grp].copy()
        if gdf.empty:
            logger.warning("分组 %s 无数据", grp)
            continue

        logger.info("\n训练 %s (%d 行)...", grp, len(gdf))
        t0 = time.time()
        model, imp = train_model(gdf, grp)
        if model is None:
            logger.warning("分组 %s 训练失败", grp)
            continue
        logger.info("耗时: %.1f 秒", time.time() - t0)

        if imp is not None:
            logger.info("  Top-10 特征重要性:")
            for _, r in imp.head(10).iterrows():
                logger.info("    %-30s %4d", r["feature"], int(r["importance"]))

        save_model(model, grp)
        models[grp] = model

    # ── 验证 ──
    dates = sorted(feature_df[COL_TRADE_DATE].unique())
    if len(dates) < 20:
        logger.warning("数据天数不足，跳过验证")
        return

    split = int(len(dates) * 0.8)
    train_d, test_d = set(dates[:split]), set(dates[split:])
    logger.info("\n验证起始: %s, 训练 %d 天, 验证 %d 天", dates[split], len(train_d), len(test_d))

    for grp, nm in [("micro", "微盘"), ("mid", "中小盘")]:
        model = load_model(grp)
        if model is None:
            continue
        gdf = feature_df[feature_df["group"] == grp]
        test = gdf[gdf[COL_TRADE_DATE].isin(test_d)]
        if test.empty:
            logger.warning("分组 %s: 验证集为空，跳过", nm)
            continue
        feats = [c for c in GROUP_CONFIG[grp]["features"] if c in test.columns]
        X_te = test[feats].fillna(0)
        if X_te.empty:
            logger.warning("分组 %s: 验证特征为空，跳过", nm)
            continue
        pred = model.predict(X_te)

        from scipy.stats import spearmanr
        if "close" in test.columns:
            r, _ = spearmanr(pred, test["close"].values)
            logger.info("分组 %s: Spearman r=%.4f (验证 %d 样本)", nm, r, len(test))

        t = test.copy()
        t["pred"] = pred
        if "close" in t.columns:
            top20 = t.nlargest(20, "pred")["close"].median()
            bot20 = t.nsmallest(20, "pred")["close"].median()
            spread = _safe_div(top20 - bot20, bot20) * 100 if bot20 > 0 else 0
            logger.info("  Top20 med=%.2f, Bottom20 med=%.2f, spread=%.2f%%", top20, bot20, spread)

    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info("  微盘模型: %s", MODEL_PATHS.get("micro", "N/A"))
    logger.info("  中小盘模型: %s", MODEL_PATHS.get("mid", "N/A"))
    logger.info("  大盘股: 线性评分（ML 不适用）")
    logger.info("  特征数: %d", len(ALL_FEATURES))
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _safe_div(a, b):
    return a / b if abs(b) > 1e-10 else 0.0


def _date_add(date_str: str, offset: int) -> str:
    """日期加减（YYYYMMDD 格式）。"""
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, "%Y%m%d")
    d += timedelta(days=offset)
    return d.strftime("%Y%m%d")


def _fmt_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def show_status():
    """显示特征数据集状态。"""
    if not os.path.exists(CACHE_PATH):
        logger.info("特征数据集不存在 (%s)", CACHE_PATH)
        return

    df = pd.read_parquet(CACHE_PATH)
    cols = [c for c in df.columns if c not in (COL_TS_CODE, COL_TRADE_DATE)]
    non_zero = sum(1 for c in cols if df[c].fillna(0).abs().sum() > 0)

    logger.info("=" * 60)
    logger.info("特征数据集状态")
    logger.info("=" * 60)
    logger.info("  路径: %s", CACHE_PATH)
    logger.info("  大小: %s", _fmt_size(os.path.getsize(CACHE_PATH)))
    logger.info("  行数: %d", len(df))
    logger.info("  列数: %d", len(df.columns))
    logger.info("  特征数: %d (非零特征: %d)", len(cols), non_zero)
    logger.info("  日期范围: %s ~ %s", df[COL_TRADE_DATE].min(), df[COL_TRADE_DATE].max())
    logger.info("  交易日数: %d", df[COL_TRADE_DATE].nunique())
    logger.info("  股票数: %d", df[COL_TS_CODE].nunique())
    logger.info("  每日均行: %.0f", len(df) / df[COL_TRADE_DATE].nunique())
    logger.info("  最后更新: %s", time.strftime("%Y-%m-%d %H:%M:%S",
                 time.localtime(os.path.getmtime(CACHE_PATH))))


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ML 特征数据集 + 模型训练 V4")
    parser.add_argument("--rebuild", action="store_true", help="全量重建特征数据集")
    parser.add_argument("--update", action="store_true", help="增量更新特征数据集")
    parser.add_argument("--train", action="store_true", help="基于当前数据集训练模型")
    parser.add_argument("--walk-forward", action="store_true", help="Walk-Forward 时间序列滚动训练（全量98折，周六用）")
    parser.add_argument("--refresh-models", action="store_true",
                        help="个股模型快速刷新（仅最终模型，每日用）")
    parser.add_argument("--initial-train-days", type=int, default=150, help="初始训练窗口（交易日）")
    parser.add_argument("--val-days", type=int, default=25, help="验证窗口大小")
    parser.add_argument("--step-days", type=int, default=25, help="滚动步长")
    parser.add_argument("--forward-days", type=int, default=20, help="前向收益预测天数")
    parser.add_argument("--status", action="store_true", help="显示特征数据集状态")
    parser.add_argument("--show-data", action="store_true", help="训练时显示特征分布")
    parser.add_argument("--stocks", type=int, default=None, help="最大股票数；默认 None=全量池(约4999)")
    parser.add_argument("--days", type=int, default=120, help="最小交易日数")
    args = parser.parse_args()

    # ── 状态查询 ──
    if args.status:
        show_status()
        return

    # ── 个股模型快速刷新（每日增量训练）──
    # 内部自行判断新鲜度：模型新鲜则快速跳过，过期才加载特征矩阵重训。
    # 注意：特征矩阵须先由 --update 刷新到最新交易日。
    if args.refresh_models:
        ensure_stock_model_fresh()
        return

    # ── 特征数据集管理 ──
    feature_df = None

    if args.update:
        # 增量更新：加载现有缓存，追加新日期
        if os.path.exists(CACHE_PATH):
            logger.info("增量更新模式")
            existing = pd.read_parquet(CACHE_PATH)
            feature_df = build_feature_dataset(
                max_stocks=args.stocks, min_days=args.days, existing_df=existing
            )
        else:
            logger.warning("缓存不存在 (%s)，转为全量重建", CACHE_PATH)
            feature_df = build_feature_dataset(max_stocks=args.stocks, min_days=args.days)

    elif args.rebuild or not os.path.exists(CACHE_PATH):
        logger.info("全量重建模式")
        feature_df = build_feature_dataset(max_stocks=args.stocks, min_days=args.days)

    else:
        # 默认：加载已有缓存
        if os.path.exists(CACHE_PATH):
            logger.info("加载已有特征数据集: %s", CACHE_PATH)
            feature_df = pd.read_parquet(CACHE_PATH)
            logger.info("  %d 行 × %d 列", len(feature_df), len(feature_df.columns))

    # ── Walk-Forward 训练 ──
    if args.walk_forward and feature_df is not None and not feature_df.empty:
        logger.info("=" * 60)
        logger.info("Walk-Forward 时间序列滚动训练")
        logger.info("  初始训练: %d 天, 验证: %d 天, 步长: %d 天, 前向: %d 天",
                    args.initial_train_days, args.val_days, args.step_days, args.forward_days)
        logger.info("=" * 60)

        # 分组标签（复用 _compute_group_column，与 ensure_stock_model_fresh 一致）
        feature_df = _compute_group_column(feature_df)

        log_path = os.path.join(DATA_DIR, "walk_forward_report.json")
        all_results = {}
        best_model = None

        for grp in ["micro", "mid"]:
            result = walk_forward_train(
                feature_df, group=grp,
                initial_train_days=args.initial_train_days,
                val_days=args.val_days,
                step_days=args.step_days,
                forward_days=args.forward_days,
            )
            all_results[grp] = result

            if "error" not in result:
                # 保存最终模型
                if result.get("final_model"):
                    save_model(result["final_model"], grp)
                    logger.info("Walk-Forward 最终模型已保存: %s", grp)

                # 取最好的折的模型作为备用
                if result.get("folds") and result.get("fold_models"):
                    best_fold = max(result["folds"], key=lambda x: x["spearman_r"])
                    logger.info("  最佳折 #%d: r=%.4f (%s~%s)",
                                best_fold["fold"], best_fold["spearman_r"],
                                best_fold.get("val_start", "?"), best_fold.get("val_end", "?"))

        # 保存报告
        save_walk_forward_report(
            {"micro": all_results.get("micro", {}),
             "mid": all_results.get("mid", {})},
            log_path
        )

        # 总体输出
        logger.info("=" * 60)
        logger.info("Walk-Forward 训练完成")
        for grp in ["micro", "mid"]:
            r = all_results.get(grp, {})
            if "error" in r:
                logger.info("  %s: %s", grp, r["error"])
            else:
                o = r.get("overall", {})
                logger.info("  %s: %.1f折 | mean_r=%+.4f | pos=%.0f%% | spread=%.2f%%",
                            grp, o.get("n_folds", 0), o.get("mean_spearman_r", 0),
                            o.get("positive_ratio", 0) * 100, o.get("mean_spread", 0))
        logger.info("  报告: %s", log_path)
        logger.info("=" * 60)

    # ── 常规训练 ──
    elif args.train and feature_df is not None and not feature_df.empty:
        train_from_dataset(feature_df, show_data=args.show_data)
    elif args.train:
        logger.error("特征数据集不可用，无法训练")


if __name__ == "__main__":
    main()
