#!/usr/bin/env python3
"""
v6_daily_pipeline.py — V6 日报生产管道

一站式流程: 检查→刷新→温度→衰减→仓位→HTML→Telegram→部署

用法:
  python3 backtest/v6_daily_pipeline.py --refresh-only    # 仅刷新数据
  python3 backtest/v6_daily_pipeline.py --report-only     # 仅生成报告
  python3 backtest/v6_daily_pipeline.py --telegram --deploy  # 完整流程+推送

── Changelog ──
# 2026-08-03 Claude: 报告生成时统一自愈机制（覆盖完整数据依赖链）
#   --report-only 新增: 数据滞后 → 自动补拉 → 重查 → 出报告
#     链: stock_daily(fetch_all_stocks) → V4(retrain_ml --update) → V5(integrate_v5 --update)
#     注意: V5 依赖 V4 基底, 只刷 V5 会漏掉新日期——V4 滞后必须先补 V4
#   自愈仍失败 → 传 --stale-days 给 v6_daily_report.py 打"数据滞后 N 天"横幅(消费端自证)
#   修复 check_freshness 滞后误判: 旧启发式跨周末算错(周五→周一报滞后3天, 假警报)
#   改用 Tushare trade_cal 精确日历判定(含节假日), 避免每周一误触发 25 分钟空拉
# 2026-08-02 Claude: 初版, 检查→刷新→温度→衰减→仓位→HTML→Telegram→部署
─────────────
"""

import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

FEAT_PATH = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
V4_PATH = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v4.parquet")
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data_storage", "lgb_models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "v6_pipeline.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("v6_pipeline")


# ═══════════════════════════════════════════════════════════════
# 新鲜度检查
# ═══════════════════════════════════════════════════════════════

def _stock_expected_and_lag(db_max, now=None):
    """返回 (expected, lag): 期望最新入库交易日 + 实际滞后交易日数。

    用 Tushare trade_cal 精确计算（含节假日），避免旧启发式跨周末误判。
    北京时判定:
      - 交易日 22:30 前（早间/盘中/晚间拉取前）: 期望 = 上一交易日
      - 22:30 后 或 非交易日: 期望 = 最新交易日
    日历查询失败时返回 (None, None)，由调用方退化为粗略估算。
    """
    import tushare as ts
    from config import TUSHARE_TOKEN
    now = now or datetime.now()
    today = now.strftime("%Y%m%d")
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        start = (now - timedelta(days=20)).strftime("%Y%m%d")
        df = pro.trade_cal(exchange="SSE", start_date=start, end_date=today)
        open_days = sorted(df[df["is_open"] == 1]["cal_date"].astype(str).tolist())
        if not open_days:
            return None, None
        latest = open_days[-1]
        now_min = now.hour * 60 + now.minute
        if latest == today and now_min < 22 * 60 + 30:
            expected = open_days[-2] if len(open_days) >= 2 else latest
        else:
            expected = latest
        lag = len([d for d in open_days if d > str(db_max) and d <= expected])
        return expected, lag
    except Exception as e:
        logger.warning("交易日历查询失败（退化粗略判定）: %s", e)
        return None, None


def check_freshness():
    """检查所有数据源新鲜度，返回 (status, warnings)"""
    warnings = []
    status = {}

    # 1. stock_daily（精确交易日历判定）
    db = sqlite3.connect(STOCK_DB)
    db_max = pd.read_sql("SELECT MAX(trade_date) as d FROM stock_daily", db)["d"][0]
    db.close()
    expected, db_lag = _stock_expected_and_lag(db_max)
    if db_lag is None:
        db_lag = _trading_days_between(db_max, datetime.now().strftime("%Y%m%d"))
        expected = None
    status["stock_daily"] = {"date": db_max, "lag": db_lag, "expected": expected}
    if db_lag > 0:
        exp_txt = f"（期望 {expected}）" if expected else ""
        warnings.append(f"⚠️ stock_daily 滞后 {db_lag} 个交易日 ({db_max}{exp_txt})")

    # 2. feature_matrix（同用精确日历）
    if os.path.exists(FEAT_PATH):
        feat = pd.read_parquet(FEAT_PATH)
        feat_max = str(feat["trade_date"].max())
        _, feat_lag = _stock_expected_and_lag(feat_max)
        if feat_lag is None:
            feat_lag = _trading_days_between(feat_max, datetime.now().strftime("%Y%m%d"))
        status["feature_matrix"] = {"date": feat_max, "lag": feat_lag}
        if feat_lag > 0:
            warnings.append(f"⚠️ 特征矩阵滞后 {feat_lag} 个交易日 ({feat_max})")
    else:
        status["feature_matrix"] = {"date": None, "lag": 999}
        warnings.append("❌ 特征矩阵不存在!")

    # 2.5 V4 基底（V5 依赖 V4，滞后会卡住 V5 产出）
    if os.path.exists(V4_PATH):
        v4_max = str(pd.read_parquet(V4_PATH, columns=["trade_date"])["trade_date"].max())
        _, v4_lag = _stock_expected_and_lag(v4_max)
        if v4_lag is None:
            v4_lag = _trading_days_between(v4_max, datetime.now().strftime("%Y%m%d"))
        status["v4_matrix"] = {"date": v4_max, "lag": v4_lag}
        if v4_lag > 0:
            warnings.append(f"⚠️ V4 特征矩阵滞后 {v4_lag} 个交易日 ({v4_max})")
    else:
        status["v4_matrix"] = {"date": None, "lag": 999}
        warnings.append("❌ V4 特征矩阵不存在!")

    # 3. models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("v6_momentum_")]
    if model_files:
        latest_model = max(f.split("_")[-1].replace(".pkl", "") for f in model_files)
        model_dt = datetime.strptime(latest_model, "%Y%m%d")
        model_age = (datetime.now() - model_dt).days
        status["models"] = {"date": latest_model, "age_days": model_age}
        if model_age > 7:
            warnings.append(f"⚠️ 模型已 {model_age} 天未更新 ({latest_model})")
    else:
        status["models"] = {"date": None, "age_days": 999}
        warnings.append("❌ 无模型文件!")

    return status, warnings


def _trading_days_between(date1, date2):
    """估算两个日期间交易日数（简化：排除周末）"""
    if not date1 or not date2:
        return 999
    d1 = datetime.strptime(str(date1), "%Y%m%d")
    d2 = datetime.strptime(str(date2), "%Y%m%d")
    days = (d2 - d1).days
    # 粗略减去周末
    weekends = (days // 7) * 2
    return max(0, days - weekends)


# ═══════════════════════════════════════════════════════════════
# 数据刷新
# ═══════════════════════════════════════════════════════════════

def _v4_freshness():
    """返回 V4 特征矩阵是否新鲜（最新交易日 == 期望交易日）。

    V5 依赖 V4 作为基底（integrate_v5 从 V4 join 特征），V4 滞后时
    V5 无法产出新日期 —— 自愈必须覆盖完整依赖链 stock_daily → V4 → V5。
    """
    if not os.path.exists(V4_PATH):
        return False
    try:
        v4_max = str(pd.read_parquet(V4_PATH, columns=["trade_date"])["trade_date"].max())
        _, lag = _stock_expected_and_lag(v4_max)
        return lag is not None and lag <= 0
    except Exception as e:
        logger.warning("V4 新鲜度检查失败: %s", e)
        return False


def refresh_features():
    """增量刷新 V5 特征矩阵（V4 滞后时先补 V4，保证完整依赖链）。"""
    # 1) V4 基底
    if not _v4_freshness():
        logger.info("V4 特征矩阵滞后，先补 V4 (retrain_ml --update)...")
        v4_script = os.path.join(PROJECT_ROOT, "analysis", "retrain_ml.py")
        r = subprocess.run(
            [sys.executable, v4_script, "--update"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800
        )
        if r.returncode != 0:
            logger.error("V4 刷新失败: %s", r.stderr[-500:])
        else:
            for line in r.stderr.split("\n"):
                if "新特征" in line or "数据集已保存" in line:
                    logger.info(line.strip())
    # 2) V5
    logger.info("刷新 V5 特征矩阵...")
    script = os.path.join(PROJECT_ROOT, "integrate_v5_features.py")
    result = subprocess.run(
        [sys.executable, script, "--update", "--skip-tushare"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        logger.error("V5 刷新失败: %s", result.stderr[-500:])
        return False
    # 提取日志中的关键信息
    for line in result.stderr.split("\n"):
        if "增量追加" in line or "已是最新" in line or "跳过" in line:
            logger.info(line.strip())
    return True


def refresh_stock_daily():
    """增量刷新个股日线（统一自愈：报告前若滞后自动补拉）。

    复用 fetch_all_stocks 逐股循环 + 限流，增量约 20-25 分钟；
    报告可能推迟到 ~08:55（可接受，晚但正确好过准时但错误）。
    """
    logger.info("刷新个股日线数据（自愈）...")
    try:
        from data.stock_industry_mapping import load_stock_industry_mapping, load_stock_universe
        from data.stock_daily_updater import fetch_all_stocks
        mapping = load_stock_industry_mapping()
        codes = sorted(mapping.keys())
        list_dates = {u["ts_code"]: u["list_date"] for u in load_stock_universe()}
        summary = fetch_all_stocks(STOCK_DB, codes, list_dates=list_dates)
        logger.info("个股刷新完成: %s", summary)
        return True
    except Exception as e:
        logger.error("个股刷新失败: %s", e)
        return False


def refresh_models():
    """检查是否需要周度重训练"""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("v6_momentum_")]
    if not model_files:
        logger.info("无模型，执行初始训练...")
        return _run_retrain()
    latest = max(f.split("_")[-1].replace(".pkl", "") for f in model_files)
    age = (datetime.now() - datetime.strptime(latest, "%Y%m%d")).days
    if age >= 7:
        logger.info("模型已 %d 天，执行周度重训练...", age)
        return _run_retrain()
    logger.info("模型新鲜 (%d 天), 跳过重训练", age)
    return True


def _run_retrain():
    script = os.path.join(SCRIPT_DIR, "v6_weekly_retrain.py")
    result = subprocess.run(
        [sys.executable, script, "--weeks", "12"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600
    )
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V6 日报生产管道")
    parser.add_argument("--refresh-only", action="store_true", help="仅刷新数据")
    parser.add_argument("--report-only", action="store_true", help="仅生成报告")
    parser.add_argument("--telegram", action="store_true", help="推送 Telegram")
    parser.add_argument("--deploy", action="store_true", help="部署到 GitHub Pages")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("═" * 50)
    logger.info("V6 日报管道 — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("═" * 50)

    # ── Step 1: 新鲜度 ──
    status, warnings = check_freshness()
    logger.info("新鲜度: stock_daily=%s feat=%s models=%s",
                status["stock_daily"]["date"], status["feature_matrix"]["date"], status["models"]["date"])
    for w in warnings:
        logger.warning(w)

    # ── Step 2: 刷新 ──
    if not args.report_only:
        logger.info("── 数据刷新 ──")
        if status["feature_matrix"]["lag"] > 1:
            if not refresh_features():
                logger.error("特征刷新失败，中止")
                sys.exit(1)
        if not refresh_models():
            logger.error("模型刷新失败")
            # 不中止——用旧模型也能跑
    else:
        # --report-only: 统一自愈 —— 报告生成前若数据滞后，自动补拉再出报告
        # 完整依赖链: stock_daily → V4 → V5（refresh_features 内部按链补 V4）
        lag_s = status["stock_daily"]["lag"] or 0
        lag_f = status["feature_matrix"]["lag"] or 0
        lag_v4 = status["v4_matrix"]["lag"] or 0
        if lag_s > 0 or lag_f > 0 or lag_v4 > 0:
            logger.info("── 数据自愈 ── (stock_daily 滞后 %d, V4 滞后 %d, V5 滞后 %d)",
                        lag_s, lag_v4, lag_f)
            if lag_s > 0:
                refresh_stock_daily()
            if lag_f > 0 or lag_v4 > 0:
                refresh_features()
            status, warnings = check_freshness()
            logger.info("自愈后新鲜度: stock_daily=%s V4=%s feat=%s",
                        status["stock_daily"]["date"], status["v4_matrix"]["date"],
                        status["feature_matrix"]["date"])

    if args.refresh_only:
        status2, _ = check_freshness()
        logger.info("刷新后新鲜度: feat=%s", status2["feature_matrix"]["date"])
        logger.info("管道完成 (仅刷新) — %.0f 秒", time.time() - t0)
        return

    # ── Step 3: 日报 ──
    logger.info("── 日报生成 ──")
    script = os.path.join(SCRIPT_DIR, "v6_daily_report.py")
    db = sqlite3.connect(STOCK_DB)
    db_max = pd.read_sql("SELECT MAX(trade_date) as d FROM stock_daily", db)["d"][0]
    db.close()

    cmd = [sys.executable, script, "--asof", str(db_max)]
    # 自愈后仍滞后 → 报告带"数据滞后 N 天"横幅（消费端自证，杜绝静默错数据）
    stale_days = max(status["stock_daily"]["lag"] or 0,
                     status["feature_matrix"]["lag"] or 0,
                     status["v4_matrix"]["lag"] or 0)
    if stale_days > 0:
        cmd += ["--stale-days", str(stale_days)]
        logger.warning("⚠️ 自愈后仍滞后 %d 个交易日，报告将带数据滞后横幅", stale_days)
    if args.telegram:
        cmd.append("--telegram")
    if args.deploy:
        cmd.append("--deploy")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error("日报生成失败: %s", result.stderr[-500:])
        sys.exit(1)

    for line in result.stderr.split("\n"):
        if "✅" in line or "温度" in line:
            logger.info(line.strip())

    logger.info("═" * 50)
    logger.info("管道完成 — %.0f 秒", time.time() - t0)

    # 失败时退出码非零（供 cron 检测）
    if warnings:
        logger.warning("有 %d 条警告", len(warnings))


if __name__ == "__main__":
    main()
