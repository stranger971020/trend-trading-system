#!/usr/bin/env python3
"""
A股趋势交易系统 - 主入口
用法:
    python3 run_analysis.py                # 正常执行（检查交易日，晚间完整报告）
    python3 run_analysis.py --morning      # 早间模式（仅更新行业数据，重新选股）
    python3 run_analysis.py --force        # 强制执行（跳过交易日检查）
    python3 run_analysis.py --dry-run      # 干跑（不发送 Telegram）
    python3 run_analysis.py --html-only    # 仅生成 HTML 报告（不推送）

输出:
    - reports/report_YYYYMMDD_HHMM.html  每日 HTML 报告（早晚各一份）
    - reports/latest_morning.html         最新早间报告
    - reports/latest.html                 最新晚间报告
    - Telegram 推送（除非 --dry-run 或 --html-only）
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import tushare as ts

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    TUSHARE_TOKEN,
    DB_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOGS_DIR,
    BEIJING_TZ_OFFSET,
    COL_TS_CODE,
)
from data.industry_mapping import load_industry_mapping
from data.industry_daily_updater import (
    init_db,
    fetch_and_store_incremental,
    get_db_connection,
    load_daily_data,
)
from data.stock_industry_mapping import (
    load_stock_industry_mapping,
    get_stocks_by_industry,
)
from data.stock_daily_updater import (
    update_stocks_for_industries,
    fetch_all_stocks,
    load_stock_daily,
)
from data.l3_industry_updater import (
    load_l3_mapping,
    fetch_and_store_l3,
    load_l3_daily,
)
from analysis.module1_sentiment import analyze_sentiment
from analysis.module2_persistence import (
    analyze_persistence,
    analyze_l3_persistence,
)
from analysis.module3_stock_mining import analyze_stocks, analyze_stocks_l3, analyze_stocks_l2
from analysis.module0_l3_leading import analyze_l3_leading
from analysis.module0_l2_leading import analyze_l2_leading
from data.l2_industry_updater import load_l2_mapping, fetch_and_store_l2, load_l2_daily
from analysis.market_regime import determine_regime
from analysis.weekly_filter import daily_to_weekly, compute_weekly_momentum, apply_weekly_filter
from analysis.crowding_warning import detect_crowding
from analysis.atr_stop_loss import compute_stop_loss
from analysis.moneyflow_filter import fetch_and_cache_moneyflow, apply_moneyflow_filter
from analysis.fundamental_factors import compute_fundamental_score
from analysis.ml_model import load_model, rerank_with_ml, build_feature_matrix
from analysis.margin_warning import fetch_today_margin, detect_margin_divergence
from analysis.virtual_portfolio import update_portfolio, get_portfolio_summary
from analysis.anomaly_detector import detect_anomalies
from analysis.module_stock_derived_industry import analyze_stock_derived_industry
from analysis.stock_picks import generate_picks_text, generate_picks_html
from report.report_generator import generate_report
from report.html_report_generator import generate_html_report
from notify.telegram_sender import send_report
from scheduler import is_trading_day

# ============================================================
# 日志初始化
# ============================================================
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"run_analysis_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("run_analysis")


# ============================================================
# 主流程
# ============================================================

def main(force: bool = False, dry_run: bool = False, morning: bool = False) -> bool:
    """执行完整分析流程。

    Args:
        force: 跳过交易日检查
        dry_run: 跳过 Telegram 推送
        morning: 早间模式（跳过个股数据下载）

    Returns:
        True 如果整体执行成功
    """
    start_time = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    beijing_now_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("A股趋势交易系统 v0.1.0 (MVP)")
    logger.info("启动时间: %s CST", beijing_now_str)
    logger.info("=" * 60)

    # ---- 模块状态追踪 ----
    module_status = {
        "module0": "pending",
        "module1": "pending",
        "module2": "pending",
        "module2_l3": "pending",
        "module3": "pending",
        "regime": "pending",
        "weekly_filter": "pending",
        "crowding": "pending",
        "atr": "pending",
        "data_update": "pending",
    }
    data_summary = {
        "latest_date": "N/A",
        "l3_latest_date": "N/A",
        "stock_latest_date": "N/A",
        "moneyflow_latest_date": "N/A",
        "margin_latest_date": "N/A",
        "industries_updated": 0,
        "total_rows": 0,
        "new_rows": 0,
    }

    # ========================================================
    # 0. 交易日检查
    # ========================================================
    if not force:
        try:
            if not is_trading_day():
                logger.info("今日非交易日，分析跳过。使用 --force 可强制执行。")
                return True  # 非交易日不算失败
        except Exception as e:
            logger.warning("交易日检查异常: %s，继续执行...", e)
    else:
        logger.info("--force: 跳过交易日检查")

    # ========================================================
    # 1. 初始化 Tushare
    # ========================================================
    logger.info("初始化 Tushare Pro 连接...")
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        logger.info("Tushare 连接成功")
    except Exception as e:
        logger.critical("Tushare 连接失败: %s", e)
        return False

    # ========================================================
    # 2. 加载行业映射
    # ========================================================
    logger.info("加载 SW L1 行业映射...")
    try:
        mapping = load_industry_mapping()
        logger.info("已加载 %d 个行业映射", len(mapping))
    except Exception as e:
        logger.critical("行业映射加载失败: %s", e)
        return False

    # ========================================================
    # 3. 数据库初始化 & 数据更新
    # ========================================================
    logger.info("初始化数据库...")
    try:
        conn = init_db()
    except Exception as e:
        logger.critical("数据库初始化失败: %s", e)
        return False

    total_new_rows = 0
    try:
        summary = fetch_and_store_incremental(conn, mapping)
        total_new_rows = sum(summary.values())
        industries_updated = sum(1 for v in summary.values() if v > 0)
        data_summary["new_rows"] = total_new_rows
        data_summary["industries_updated"] = len(mapping)
        module_status["data_update"] = "success"

        # 获取最新日期
        cur = conn.execute("SELECT MAX(trade_date) FROM sw_index_daily")
        row = cur.fetchone()
        if row and row[0]:
            data_summary["latest_date"] = str(row[0])

        cur = conn.execute("SELECT COUNT(*) FROM sw_index_daily")
        row = cur.fetchone()
        if row:
            data_summary["total_rows"] = row[0]

        logger.info(
            "数据更新完成: %d 个行业有新数据，%d 条新记录",
            industries_updated, total_new_rows,
        )
    except Exception as e:
        logger.error("数据更新失败: %s", e, exc_info=True)
        module_status["data_update"] = "failed"
        # 继续使用已有数据

    # ========================================================
    # 4. 加载日线数据
    # ========================================================
    logger.info("从数据库加载日线数据...")
    try:
        all_codes = sorted(mapping.keys())
        daily_df = load_daily_data(conn, all_codes)
        if daily_df.empty:
            logger.warning("数据库为空，无法继续分析。请先确保数据获取成功。")
            conn.close()
            # 仍然生成报告（显示数据缺失）
            daily_df = None  # type: ignore
    except Exception as e:
        logger.error("数据加载失败: %s", e)
        daily_df = None  # type: ignore

    # ========================================================
    # 5. 模块1: 市场情绪
    # ========================================================
    logger.info("=" * 40)
    logger.info("执行模块1: 市场情绪与择时")
    try:
        sentiment_result = analyze_sentiment(daily_df, mapping)
        module_status["module1"] = sentiment_result.get("status", "failed")
    except Exception as e:
        logger.error("模块1异常: %s", e, exc_info=True)
        sentiment_result = {"status": "failed", "error": str(e)}
        module_status["module1"] = "failed"

    # ========================================================
    # 6. 模块2: 板块持续性
    # ========================================================
    logger.info("=" * 40)
    logger.info("执行模块2: 板块持续性评分")
    try:
        persistence_result = analyze_persistence(daily_df, mapping)
        module_status["module2"] = persistence_result.get("status", "failed")
    except Exception as e:
        logger.error("模块2异常: %s", e, exc_info=True)
        persistence_result = {"status": "failed", "error": str(e)}
        module_status["module2"] = "failed"

    # ========================================================
    # 6b. L3 三级行业数据
    # ========================================================
    logger.info("=" * 40)
    logger.info("加载三级行业数据...")
    l3_daily_df = None
    l3_leading_result = None
    l3_persistence_result = None
    l3_stock_result = None

    try:
        l3_mapping = load_l3_mapping()
        logger.info("L3 行业映射: %d 个代码", len(l3_mapping))

        l3_summary = fetch_and_store_l3(DB_PATH, l3_mapping)
        data_summary["l3_total"] = l3_summary["total"]
        data_summary["l3_active"] = l3_summary["active"]
        data_summary["l3_new_rows"] = l3_summary["new_rows"]
        logger.info(
            "L3 数据: %d 代码, %d 活跃, +%d 条",
            l3_summary["total"], l3_summary["active"], l3_summary["new_rows"],
        )

        l3_daily_df = load_l3_daily(DB_PATH)
        logger.info("加载 %d 条 L3 日线数据", len(l3_daily_df))
        if not l3_daily_df.empty:
            data_summary["l3_latest_date"] = str(l3_daily_df["trade_date"].max())

        # ---- 模块0: 三级领先信号 ----
        logger.info("执行模块0: 三级行业领先信号...")
        l3_leading_result = analyze_l3_leading(l3_daily_df, daily_df)
        module_status["module0"] = l3_leading_result.get("status", "failed")
        logger.info("模块0: %s", module_status["module0"])

        # ---- 模块2 L3: 三级持续性 ----
        logger.info("执行模块2 L3: 三级行业持续性评分...")
        l3_persistence_result = analyze_l3_persistence(l3_daily_df)
        module_status["module2_l3"] = l3_persistence_result.get("status", "failed")
        logger.info("模块2 L3: %s", module_status["module2_l3"])

    except Exception as e:
        logger.error("L3 管线失败: %s", e, exc_info=True)
        module_status["module0"] = "failed"
        module_status["module2_l3"] = "failed"

    # ---- L2 二级行业（投资方向主引擎） ----
    l2_daily_df = None
    l2_leading_result = None
    l2_persistence_result = None
    try:
        l2_mapping = load_l2_mapping()
        logger.info("L2 行业映射: %d 个代码", len(l2_mapping))
        l2_summary = fetch_and_store_l2(DB_PATH, l2_mapping)
        data_summary["l2_active"] = l2_summary["active"]
        logger.info("L2 数据: %d 活跃, +%d 条", l2_summary["active"], l2_summary["new_rows"])
        l2_daily_df = load_l2_daily(DB_PATH)
        logger.info("加载 %d 条 L2 日线数据", len(l2_daily_df))
        # L2 领先信号
        l2_leading_result = analyze_l2_leading(l2_daily_df, daily_df)
        module_status["module0_l2"] = l2_leading_result.get("status", "failed")
        # L2 持续性
        from analysis.module2_persistence import compute_persistence
        l2_persistence_df = compute_persistence(l2_daily_df)
        l2_persistence_result = {"status":"success","df":l2_persistence_df}
        if not l2_persistence_df.empty:
            l2_persistence_result["high_persistence"] = l2_persistence_df[l2_persistence_df["label"]=="🔥高持续性"]["name"].tolist()
            l2_persistence_result["medium_persistence"] = l2_persistence_df[l2_persistence_df["label"]=="⚡中等持续性"]["name"].tolist()
            l2_persistence_result["low_persistence"] = l2_persistence_df[l2_persistence_df["label"]=="⚠️低持续性"]["name"].tolist()
        module_status["module2_l2"] = "success"
        logger.info("L2 持续性: %d 行业评分完成", len(l2_persistence_df))
    except Exception as e:
        logger.error("L2 管线失败: %s", e, exc_info=True)
        module_status["module0_l2"] = "failed"
        module_status["module2_l2"] = "failed"

    # ========================================================
    # 7. 模块3: 个股挖掘（L2 选股）
    # ========================================================
    logger.info("=" * 40)
    logger.info("执行模块3: 个股挖掘")

    stock_mapping = None
    stock_daily_df = None
    stock_summary = {"fetched": 0, "updated": 0, "new_rows": 0}

    try:
        # 7a. 加载个股→行业映射
        logger.info("加载个股行业映射...")
        stock_mapping = load_stock_industry_mapping()
        logger.info("已加载 %d 只个股的行业映射", len(stock_mapping))

        # 7b. 全量拉取所有个股日线数据（早间模式跳过，晚间首次~18分钟，后续增量~30秒）
        all_stock_codes = sorted(stock_mapping.keys())
        if morning:
            logger.info("早间模式：跳过个股数据下载，使用已有数据")
            stock_summary = {"total_stocks": len(all_stock_codes), "updated": 0, "new_rows": 0}
        else:
            stock_summary = fetch_all_stocks(DB_PATH, all_stock_codes)
        data_summary["stocks_fetched"] = stock_summary.get("total_stocks", 0)
        data_summary["stocks_updated"] = stock_summary.get("updated", 0)
        data_summary["stocks_new_rows"] = stock_summary.get("new_rows", 0)

        # 7c. 加载全量个股数据
        stock_daily_df = load_stock_daily(DB_PATH, all_stock_codes, min_rows=20)
        logger.info("加载 %d 只个股日线", len(stock_daily_df[COL_TS_CODE].unique()) if not stock_daily_df.empty else 0)

        # 7d. 执行 L2 选股（优先 L2，回退 L1）
        if l2_daily_df is not None and not l2_daily_df.empty and \
           l2_persistence_result is not None and l2_persistence_result.get("status") == "success":
            stock_result = analyze_stocks_l2(
                stock_daily_df=stock_daily_df,
                l2_daily_df=l2_daily_df,
                l2_persistence_result=l2_persistence_result,
                stock_mapping=stock_mapping,
            )
            logger.info("模块3: L2 选股模式")
        else:
            stock_result = analyze_stocks(
                stock_daily_df=stock_daily_df,
                industry_daily_df=daily_df,
                persistence_result=persistence_result,
                stock_mapping=stock_mapping,
                industry_mapping=mapping,
            )
            logger.info("模块3: L1 选股模式")

        module_status["module3"] = stock_result.get("status", "skipped")

    except Exception as e:
        logger.error("模块3异常: %s", e, exc_info=True)
        stock_result = {"status": "failed", "error": str(e)}
        module_status["module3"] = "failed"

    # ========================================================
    # 7g. 风控增强（第四阶段）
    # ========================================================
    logger.info("=" * 40)
    logger.info("执行风控模块...")

    regime_result = None
    crowding_result = None

    try:
        # 宏观状态机
        regime_result = determine_regime(daily_df)
        module_status["regime"] = "success"
        logger.info("Regime: %s (ADX=%.1f)", regime_result["regime"], regime_result["adx"])
    except Exception as e:
        logger.error("Regime 失败: %s", e)
        module_status["regime"] = "failed"

    try:
        # 周线过滤 → 调整持续性评分
        weekly_df = daily_to_weekly(daily_df)
        weekly_scores = compute_weekly_momentum(weekly_df)
        if persistence_result.get("status") == "success" and not weekly_scores.empty:
            persistence_df_adjusted = apply_weekly_filter(
                persistence_result["df"], weekly_scores
            )
            persistence_result["df"] = persistence_df_adjusted
            persistence_result["weekly_filter_applied"] = True
        # 保存持续性历史趋势（周线过滤后的最终分数，与排名表一致）
        if persistence_result.get("status") == "success" and persistence_result.get("df") is not None:
            _save_persistence_history(persistence_result["df"], morning)
        module_status["weekly_filter"] = "success"
    except Exception as e:
        logger.error("周线过滤失败: %s", e)
        module_status["weekly_filter"] = "failed"

    try:
        # 拥挤度预警
        crowding_result = detect_crowding(daily_df)
        module_status["crowding"] = "success"
    except Exception as e:
        logger.error("拥挤度检测失败: %s", e)
        module_status["crowding"] = "failed"

    try:
        # ATR 止损
        if stock_result.get("status") == "success" and stock_daily_df is not None:
            stock_result["stocks"] = compute_stop_loss(
                stock_daily_df, stock_result["stocks"]
            )
        module_status["atr"] = "success"
    except Exception as e:
        logger.error("ATR 止损失败: %s", e)
        module_status["atr"] = "failed"

    data_summary["crowding_count"] = len(crowding_result.get("crowded_industries", [])) if crowding_result else 0

    # ---- 个股推算行业指标 ----
    stock_derived_industry_result = None
    try:
        if stock_daily_df is not None and not stock_daily_df.empty and stock_mapping:
            stock_derived_industry_result = analyze_stock_derived_industry(
                stock_daily_df=stock_daily_df,
                stock_mapping=stock_mapping,
            )
            module_status["stock_derived"] = stock_derived_industry_result.get("status", "failed")
            logger.info(
                "个股推算行业指标: %s, %d 个 L2 行业",
                stock_derived_industry_result.get("status"),
                stock_derived_industry_result.get("l2_count", 0),
            )
        else:
            module_status["stock_derived"] = "skipped"
    except Exception as e:
        logger.error("个股推算行业指标失败: %s", e)
        module_status["stock_derived"] = "failed"
        stock_derived_industry_result = {"status": "failed", "error": str(e)}

    # ---- 统一提取各数据源最新日期 ----
    try:
        import sqlite3 as _sql
        _c = _sql.connect(DB_PATH)
        for _tbl, _key in [("stock_daily", "stock_latest_date"),
                            ("moneyflow_cache", "moneyflow_latest_date"),
                            ("margin_cache", "margin_latest_date")]:
            _r = _c.execute(f"SELECT MAX(trade_date) FROM {_tbl}").fetchone()
            if _r and _r[0] and data_summary.get(_key) in (None, "N/A"):
                data_summary[_key] = str(_r[0])
        _c.close()
    except Exception:
        pass

    # ---- 数据新鲜度自检 ----
    data_freshness = _check_data_freshness(data_summary, logger)
    data_summary["freshness"] = data_freshness
    if data_freshness.get("stale_sources"):
        logger.warning("⚠️ 数据新鲜度警告: %s", ", ".join(data_freshness["stale_sources"]))
    else:
        logger.info("✅ 数据新鲜度正常")

    # ========================================================
    # 7h. 增强数据源（第五阶段）
    # ========================================================
    logger.info("=" * 40)
    logger.info("执行增强数据模块...")

    moneyflow_result = {}
    margin_divergences = []
    portfolio_result = None
    anomaly_result = None

    try:
        if stock_result.get("status") == "success" and stock_result.get("stocks"):
            stock_codes = [s["ts_code"] for s in stock_result["stocks"]]
            moneyflow_result = fetch_and_cache_moneyflow(stock_codes)
            stock_result["stocks"] = apply_moneyflow_filter(
                stock_result["stocks"], moneyflow_result
            )
        module_status["moneyflow"] = "success"
        # 提取资金流最新日期
        try:
            import sqlite3
            mf_conn = sqlite3.connect(DB_PATH)
            cur = mf_conn.execute("SELECT MAX(trade_date) FROM moneyflow_cache")
            row = cur.fetchone()
            if row and row[0]:
                data_summary["moneyflow_latest_date"] = str(row[0])
            mf_conn.close()
        except Exception:
            pass
    except Exception as e:
        logger.error("资金流失败: %s", e)
        module_status["moneyflow"] = "failed"

    try:
        if stock_result.get("status") == "success" and stock_result.get("stocks"):
            stock_result["stocks"] = compute_fundamental_score(
                stock_result["stocks"], stock_mapping
            )
        module_status["fundamental"] = "success"
    except Exception as e:
        logger.error("基本面因子失败: %s", e)
        module_status["fundamental"] = "failed"

    # ML 模型重排
    try:
        ml_model = load_model()
        if ml_model is not None and stock_result.get("stocks"):
            # 构建今日特征
            today_features = build_feature_matrix(
                stock_daily_df, daily_df, stock_mapping,
                persistence_scores={
                    "l1": {r["ts_code"]: r["persistence_score"] for _, r in persistence_result.get("df", pd.DataFrame()).iterrows()}
                    if persistence_result.get("df") is not None else {},
                },
            )
            stock_result["stocks"] = rerank_with_ml(
                stock_result["stocks"], today_features, ml_model
            )
            module_status["ml_rerank"] = "success"
        elif ml_model is None:
            module_status["ml_rerank"] = "skipped"
        else:
            module_status["ml_rerank"] = "skipped"
    except Exception as e:
        logger.warning("ML 重排失败: %s (回退到线性评分)", e)
        module_status["ml_rerank"] = "failed"

    try:
        fetch_today_margin()
        margin_divergences = detect_margin_divergence(daily_df, stock_mapping)
        module_status["margin"] = "success"
        try:
            import sqlite3
            mg_conn = sqlite3.connect(DB_PATH)
            cur = mg_conn.execute("SELECT MAX(trade_date) FROM margin_cache")
            row = cur.fetchone()
            if row and row[0]:
                data_summary["margin_latest_date"] = str(row[0])
            mg_conn.close()
        except Exception:
            pass
    except Exception as e:
        logger.error("融资背离失败: %s", e)
        module_status["margin"] = "failed"

    try:
        if stock_result.get("status") == "success":
            portfolio_result = update_portfolio(stock_result["stocks"], stock_daily_df)
        else:
            portfolio_result = get_portfolio_summary()
        module_status["portfolio"] = "success"
    except Exception as e:
        logger.error("虚拟持仓失败: %s", e)
        module_status["portfolio"] = "failed"

    try:
        persistence_df = persistence_result.get("df") if persistence_result else None
        anomaly_result = detect_anomalies(sentiment_result, persistence_df, stock_result)
        module_status["anomaly"] = "success"
    except Exception as e:
        logger.error("异常检测失败: %s", e)
        module_status["anomaly"] = "failed"

    # ========================================================
    # 8. 生成报告
    # ========================================================
    logger.info("=" * 40)
    logger.info("生成报告...")

    # 8a. 个股推荐（含止损与概率）
    stock_picks_text = None
    stock_picks_html = None
    try:
        latest_stock_date = data_summary.get("stock_latest_date", "latest")
        stock_picks_text = generate_picks_text(date=latest_stock_date, top_n=5)
        stock_picks_html = generate_picks_html(date=latest_stock_date, top_n=5)
        logger.info("个股推荐生成完成")
    except Exception as e:
        logger.warning("个股推荐生成失败（可选模块，不影响主流程）: %s", e)

    # 8b. 文字版报告（Telegram）
    try:
        report_text = generate_report(
            sentiment_result=sentiment_result,
            persistence_result=persistence_result,
            stock_result=stock_result,
            module_status=module_status,
            data_summary=data_summary,
            stock_derived_industry_result=stock_derived_industry_result,
            stock_picks_text=stock_picks_text,
            regime_result=regime_result,
        )
        logger.info("文字报告生成完成 (%d 字符)", len(report_text))
    except Exception as e:
        logger.critical("文字报告生成失败: %s", e, exc_info=True)
        report_text = f"<b>A股趋势交易系统 - 报告生成失败</b>\n\n错误: {e}"

    # 8b. HTML 报告
    html_path = None
    try:
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        now = datetime.now()
        time_slot = "morning" if morning else "evening"
        report_date = now.strftime("%Y%m%d")
        html_filename = f"report_{report_date}_{time_slot}.html"
        html_path = os.path.join(reports_dir, html_filename)

        # 持续性趋势数据
        trend_codes = []
        if persistence_result.get("df") is not None:
            trend_codes = persistence_result["df"].head(10)["ts_code"].tolist()
        persistence_trend = _load_persistence_trend(trend_codes) if trend_codes else {}

        html_content = generate_html_report(
            sentiment_result=sentiment_result,
            persistence_result=persistence_result,
            stock_result=stock_result,
            module_status=module_status,
            data_summary=data_summary,
            l3_leading_result=l3_leading_result,
            l3_persistence_result=l3_persistence_result,
            l2_leading_result=l2_leading_result,
            l2_persistence_result=l2_persistence_result,
            regime_result=regime_result,
            crowding_result=crowding_result,
            portfolio_result=portfolio_result,
            anomaly_result=anomaly_result,
            time_slot=time_slot,
            persistence_trend=persistence_trend,
            stock_derived_industry_result=stock_derived_industry_result,
            stock_picks_html=stock_picks_html,
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # latest 链接
        if morning:
            latest_path = os.path.join(reports_dir, "latest_morning.html")
        else:
            latest_path = os.path.join(reports_dir, "latest.html")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("HTML 报告已保存: %s (%d 字节)", html_path, len(html_content))
    except Exception as e:
        logger.error("HTML 报告生成失败: %s", e, exc_info=True)

    # 打印报告到控制台
    print("\n" + "=" * 60)
    print(report_text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))
    print("=" * 60)
    if html_path:
        print(f"📄 HTML 报告: {html_path}")

        # 自动推送报告到 GitHub
        if not dry_run:
            try:
                _git_push_reports(report_date)
                logger.info("报告已推送到 GitHub")
            except Exception as e:
                logger.warning("GitHub 推送失败（不影响报告生成）: %s", e)

        # 同步报告到 Obsidian Vault
        if not dry_run:
            try:
                _sync_report_to_obsidian(html_path, report_date, time_slot)
                logger.info("报告已同步到 Obsidian Vault")
            except Exception as e:
                logger.warning("Obsidian 同步失败（不影响报告生成）: %s", e)

    print()

    # ========================================================
    # 9. Telegram 推送
    # ========================================================
    if not dry_run:
        logger.info("=" * 40)
        logger.info("Telegram 推送...")
        # 附加 GitHub 报告链接
        report_filename = f"report_{report_date}_{time_slot}.html"
        github_url = f"https://stranger971020.github.io/trend-trading-system/reports/{report_filename}"
        report_text_with_link = report_text + f"\n\n📄 <a href='{github_url}'>GitHub 完整报告</a>"

        try:
            push_result = send_report(report_text_with_link)
            logger.info(
                "推送结果: %d/%d 发送成功, %d 失败",
                push_result["sent"],
                push_result["total"],
                push_result["failed"],
            )
        except Exception as e:
            logger.error("推送异常: %s", e, exc_info=True)
    else:
        logger.info("--dry-run: 跳过 Telegram 推送")

    # ========================================================
    # 10. 清理与总结
    # ========================================================
    conn.close()

    end_time = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    elapsed = (end_time - start_time).total_seconds()

    # 判断整体成功
    critical_failures = [
        module_status["module1"],
        module_status["module2"],
    ]
    all_failed = all(s == "failed" for s in critical_failures)

    logger.info("=" * 60)
    logger.info("执行完成 | 耗时: %.1f 秒", elapsed)
    logger.info(
        "模块状态: M0=%s M1=%s M2=%s M2L3=%s M3=%s Data=%s",
        module_status.get("module0", "N/A"),
        module_status["module1"],
        module_status["module2"],
        module_status.get("module2_l3", "N/A"),
        module_status["module3"],
        module_status["data_update"],
    )
    if all_failed:
        logger.warning("所有核心模块均失败，请检查日志")
    else:
        logger.info("分析流水线执行完毕 ✓")

    return not all_failed


# ============================================================
# CLI 入口
# ============================================================

def _save_persistence_history(df, morning: bool) -> None:
    """保存持续性得分历史（用于趋势追踪）。"""
    import json
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_storage", "persistence_history.json")
    today = datetime.now().strftime("%Y%m%d")
    slot = "morning" if morning else "evening"

    records = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            records = json.load(f)

    for _, row in df.iterrows():
        records.append({
            "date": today, "slot": slot,
            "ts_code": row["ts_code"], "name": row["name"],
            "score": row["persistence_score"],
        })

    # 保留最近 60 天
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
    records = [r for r in records if r["date"] >= cutoff]

    with open(history_path, "w") as f:
        json.dump(records, f, ensure_ascii=False)

    logger.debug("持续性历史已保存: %d 条记录", len(records))


def _load_persistence_trend(top_codes: list[str], days: int = 10) -> dict:
    """加载指定行业的持续性得分趋势。"""
    import json
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_storage", "persistence_history.json")
    if not os.path.exists(history_path):
        return {}
    with open(history_path, "r") as f:
        records = json.load(f)

    trend = {}
    for code in top_codes:
        items = [(r["date"], r["score"], r.get("name", code)) for r in records if r["ts_code"] == code]
        items.sort(key=lambda x: x[0])
        trend[code] = items[-days:] if len(items) >= 2 else items
    return trend


def _git_push_reports(report_date: str) -> None:
    """自动推送 reports/ 目录到 GitHub。"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    try:
        # stage reports
        subprocess.run(
            ["git", "add", "reports/"],
            cwd=project_root, capture_output=True, timeout=15,
        )

        # check if there are changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=project_root, capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return  # no changes

        # commit
        subprocess.run(
            ["git", "commit", "-m", f"📊 日报更新 {report_date}"],
            cwd=project_root, capture_output=True, timeout=15,
        )

        # push
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=project_root, capture_output=True, timeout=30,
        )
        logger.info("✅ GitHub 推送成功")

    except Exception as e:
        logger.warning("Git push 异常: %s", e)


def _sync_report_to_obsidian(html_path: str, report_date: str, time_slot: str) -> None:
    """将生成的 HTML 报告同步到 Obsidian Vault。

    目标目录: ~/Documents/Obsidian Vault/raw/trend-trading-reports/
    同时更新 latest.html / latest_morning.html / latest_evening.html 副本。
    """
    import shutil

    vault_dir = os.path.expanduser("~/Documents/Obsidian Vault/raw/trend-trading-reports")
    os.makedirs(vault_dir, exist_ok=True)

    src = html_path
    if not os.path.exists(src):
        logger.warning("Obsidian 同步: 源文件不存在 %s", src)
        return

    # 复制带日期的报告
    dst_dated = os.path.join(vault_dir, os.path.basename(src))
    shutil.copy2(src, dst_dated)

    # 更新 latest 副本（Obsidian 不跟随 symlink，用文件复制）
    latest_name = "latest.html" if time_slot == "evening" else f"latest_{time_slot}.html"
    dst_latest = os.path.join(vault_dir, latest_name)
    shutil.copy2(src, dst_latest)

    logger.info("✅ Obsidian Vault 同步: %s", dst_dated)


def _check_data_freshness(data_summary: dict, logger) -> dict:
    """检查各数据源的新鲜度，返回标记 dict。

    按**交易日**滞后天数判断（周末/节假日不计入滞后）：
    - 行情数据 (stock_daily): 滞后 >1 个交易日 → STALE
    - 行业指数 (sw_index): 滞后 >1 个交易日 → STALE
    - 资金流 (moneyflow): 滞后 >2 个交易日 → STALE（T+1 是正常的）
    - 融资融券 (margin): 滞后 >1 个交易日 → STALE

    Returns:
        {"status": "fresh"|"stale", "stale_sources": [...], "warnings": [...]}
    """
    from datetime import datetime, timezone, timedelta

    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    today_str = beijing_now.strftime("%Y%m%d")

    freshness = {
        "status": "fresh",
        "stale_sources": [],
        "warnings": [],
        "check_time": today_str,
    }

    # (数据源 key, data_summary key, 最大交易日后滞, 标签)
    checks = [
        ("stock_daily", "stock_latest_date", 1, "个股行情"),
        ("sw_index_daily", "latest_date", 1, "行业指数"),
        ("moneyflow_cache", "moneyflow_latest_date", 2, "资金流"),
        ("margin_cache", "margin_latest_date", 1, "融资融券"),
    ]

    for source, key, max_trading_lag, label in checks:
        date_str = data_summary.get(key)
        if date_str in (None, "N/A", ""):
            freshness["stale_sources"].append(label)
            freshness["warnings"].append(f"❌ {label}: 无数据")
            freshness["status"] = "stale"
            continue

        try:
            trading_lag = _count_trading_days_between(str(date_str)[:8], today_str)
            if trading_lag > max_trading_lag:
                freshness["stale_sources"].append(label)
                freshness["warnings"].append(
                    f"⚠️ {label}: 最新 {date_str}，滞后 {trading_lag} 个交易日（阈值 {max_trading_lag}）"
                )
                freshness["status"] = "stale"
        except Exception:
            pass

    return freshness


def _count_trading_days_between(from_date: str, to_date: str) -> int:
    """计算两个日期之间（不含起始日）的交易日数。

    用简单规则：排除周六、周日。
    Args:
        from_date: YYYYMMDD 格式的起始日期
        to_date: YYYYMMDD 格式的结束日期
    Returns:
        交易日天数
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(from_date[:8], "%Y%m%d")
    end = datetime.strptime(to_date[:8], "%Y%m%d")

    count = 0
    current = start + timedelta(days=1)  # 从次日开始计
    while current <= end:
        if current.weekday() < 5:  # 周一至周五
            count += 1
        current += timedelta(days=1)
    return count


if __name__ == "__main__":
    force = "--force" in sys.argv
    dry_run = "--dry-run" in sys.argv
    html_only = "--html-only" in sys.argv
    morning = "--morning" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    # --html-only 隐含 --dry-run
    if html_only:
        dry_run = True

    success = main(force=force, dry_run=dry_run, morning=morning)
    sys.exit(0 if success else 1)
