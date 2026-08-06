#!/usr/bin/env python3
"""
v6_daily_report.py — V6 每日交易日报

整合 P1 Gate温度 + P2 Alpha Decay衰减榜 + P3 仓位规则，生成:
  1. Telegram 消息（晨间推送，简洁卡片）
  2. GitHub Pages HTML（完整版，手机适配）

用法:
  python3 backtest/v6_daily_report.py                           # 最新数据，只生成 HTML
  python3 backtest/v6_daily_report.py --telegram                # 同时推送 Telegram
  python3 backtest/v6_daily_report.py --deploy                  # 生成 HTML + git push
  python3 backtest/v6_daily_report.py --telegram --deploy       # 全流程

输出路径: reports/v6_daily/v6_daily_YYYYMMDD.html

── Changelog ──
# 2026-08-02 Claude: 初版, HTML+Telegram 双通道, P1温度+P2衰减+P3仓位
# 2026-08-03 Claude: 顶部接入脆弱度 danger 警示条（复用 analysis/risk_assessment.compute_from_db）
#               仅展示不联动: 非 danger/失败时红条为空, HTML/TG 与旧版一致
# 2026-08-03 Claude: 新增 --stale-days 数据滞后横幅（统一自愈失败时管道传入）
#               位置: header 之后、danger 红条之前; 橙黄警示样式, 消费端自证防静默错数据
#               下游: v6_daily_pipeline.py --report-only 自愈失败时传参
#               健康检查 FILE_GROUPS 已补 v6_daily_report.py
# 2026-08-05 Claude: 衰减榜标题下插入
# 2026-08-06 Claude: 行业衰减榜改 L2（无L2回退L1），改为 衰减TOP10/上升TOP10 双榜
# 2026-08-06 Claude: Telegram 消息加 PWin 摘要（最衰减/最上升 股票 PWin前→今）
# 2026-08-06 Claude: 榜单/搜索表加回 PWin(前)/PWin(今) 列（用户要求，与分位/变化并列展示）
## 2026-08-06 Claude: compute_decay 衰减度量改方案A — 全市场PWin百分位变化(pp)替代百分比
#               消除低基数放大(0.008→0.079显示+867%失真); 搜索表显示分位前/今+变化(pp)
# 2026-08-06 Claude: compute_decay 改同模型对比(方案2) — 今/前均用同一最新模型预测
#               修复全量池重训后衰减榜被模型校准污染(99.9%股票"衰减", PWin尺度差5x)
# 2026-08-06 Claude: 衰减榜变化单元格颜色按数值正负着色(绿≥0/橙-10~0/红<-10)
#               修复逆势上升榜负值被写死标绿的问题; 表头改名「排名(变化)」+ 排名口径图例
## 2026-08-06 Claude: POSITION_RULES 按全量池重训审计更新 (v6_audit_full_20260806, bear低温WR 74.1%)
#"V6 Alpha Decay 指标使用方法"说明卡
#               (HERMES-20260805-001) — 修正派发模板的 mojibake/残缺 <b> 标签
#               并校准 Gate 阻塞率表述（回测口径: range≈93%/bear≈34%，非报告固定 gate_proxy=25%）
#               同步修正 compute_temperature 过时 docstring（"PWin 从不低于0.75"已不成立）
─────────────
"""

import argparse
import json
import os
import pickle
import sqlite3
import sys
import time as time_module
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "data_storage", "lgb_models")
FEAT_PARQUET = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
STOCK_DB = os.path.join(PROJECT_ROOT, "data_storage", "sw_index_data.db")
INDUSTRY_CSV = os.path.join(PROJECT_ROOT, "data_storage", "stock_industry_mapping.csv")
FINAL_JSON = os.path.join(SCRIPT_DIR, "v6_final_20260801.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "v6_daily")

ENGINES = ["momentum", "reversion", "breakdown"]  # 注意: breakout 在模型中名为 breakout

# ── P3 仓位规则 ──
POSITION_RULES = {
    # 全量池扩池(2026-08-06)重训后更新: v6_audit_full_20260806.json 98 folds 三维分位
    ("bull", "低温"): ("5只 (满仓)", 0.619, 0.704),
    ("bull", "中温"): ("3只 (半仓)", 0.500, 0.533),
    ("bull", "高温"): ("1只或空仓", 0.333, 0.417),
    ("range", "低温"): ("2只 (轻仓)", 0.364, 0.424),
    ("range", "中温"): ("3只 (半仓)", 0.492, 0.483),
    ("range", "高温"): ("3只 (半仓)", 0.579, 0.444),
    ("bear", "低温"): ("5只 (满仓)", 0.741, 0.593),
}

TEMP_ADVICE = {
    "低温": "市场温度极低，gate 极度严格，大部分候选被模型否决。建议严格控制仓位，等待温度回升。",
    "中温": "市场温度适中，可适度参与。选股时关注 P(Win) > 0.70 的高置信度标的。",
    "高温": "市场温度偏高，gate 相对宽松。但注意历史上 Bull+高温反而胜率下降（过热风险），建议分散行业。",
}

GITHUB_BASE = "https://stranger971020.github.io/trend-trading-system/reports/v6_daily"


def fmt_pct(v):
    return f"{v*100:.1f}%"


def fmt_num(v, d=2):
    return f"{v:.{d}f}"


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_data(final_json: str | None = None):
    """加载所有数据

    Args:
        final_json: 审计 JSON 路径（全量池扩池后重训产物）；默认用模块 FINAL_JSON
    """
    final_json = final_json or FINAL_JSON
    feat = pd.read_parquet(FEAT_PARQUET)
    feat["trade_date"] = feat["trade_date"].astype(str)

    models = {}
    for eng in ["momentum", "reversion", "breakout"]:
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(f"v6_{eng}_")])
        if files:
            with open(os.path.join(MODEL_DIR, files[-1]), "rb") as f:
                models[eng] = (pickle.load(f), files[-1].replace(".pkl", ""))

    with open(final_json) as f:
        folds = json.load(f).get("folds", [])

    industry_df = pd.read_csv(INDUSTRY_CSV) if os.path.exists(INDUSTRY_CSV) else None

    db = sqlite3.connect(STOCK_DB)
    db_max = pd.read_sql("SELECT MAX(trade_date) as d FROM stock_daily", db)["d"][0]
    next_dates = pd.read_sql(
        f"SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date > '{db_max}' ORDER BY trade_date LIMIT 5", db
    )
    db.close()

    return feat, models, folds, industry_df, db_max, next_dates


def predict_pwin(model, feat, date_str, stocks):
    """对指定日期+股票推断 P(Win)"""
    mf = model.feature_name_
    day_data = feat[(feat["trade_date"] == date_str) & (feat["ts_code"].isin(stocks))].copy()
    if len(day_data) == 0:
        return None
    avail = [c for c in mf if c in day_data.columns]
    missing = [c for c in mf if c not in day_data.columns]
    X = day_data[avail].copy()
    for m in missing:
        X[m] = 0.0
    X = X[mf].fillna(0)
    proba = model.predict_proba(X)
    pwin = proba[:, 2] if proba.ndim == 2 and proba.shape[1] >= 3 else proba[:, 1]
    return pd.DataFrame({"ts_code": day_data["ts_code"].values, "pwin": pwin}).set_index("ts_code")


# ═══════════════════════════════════════════════════════════════
# P1: 温度
# ═══════════════════════════════════════════════════════════════

def compute_temperature(feat, models, folds, asof_date):
    """计算当日温度。

    使用 PWin 百分位替代绝对阈值，避免模型概率校准偏差。
    模型 PWin 分布右偏且未校准（IC≈0.11，绝对概率无可比性），
    用百分位排名消除此偏差——只看相对排序，不看绝对概率。
    """
    stocks = sorted(feat[feat["trade_date"] == asof_date]["ts_code"].unique())
    if len(stocks) < 100:
        return {"temp": 50, "level": "中温", "regime": "range", "gate_proxy": 0.5, "wp_median": 0.5, "model_ver": "N/A", "n_stocks": len(stocks)}

    # 逐个引擎推断 PWin
    engine_pwins = {}
    for eng in ["momentum", "reversion", "breakout"]:
        if eng not in models:
            continue
        result = predict_pwin(models[eng][0], feat, asof_date, stocks)
        if result is not None:
            engine_pwins[eng] = result["pwin"]

    if not engine_pwins:
        return {"temp": 50, "level": "中温", "regime": "range", "gate_proxy": 0.5, "wp_median": 0.5, "model_ver": "N/A", "n_stocks": len(stocks)}

    # PWin 百分位排名（每引擎内独立排名，消除校准偏差）
    pwin_ranks = pd.DataFrame(index=pd.Index(stocks, name="ts_code"))
    for eng, pw in engine_pwins.items():
        pwin_ranks[f"{eng}_rank"] = pw.rank(pct=True)

    # 平均百分位排名 → 综合得分
    avg_rank = pwin_ranks.mean(axis=1)
    wp_median_rank = avg_rank.median()

    # gate_proxy: 排名后 25% 的视为"被 gate 挡"（相对概念，不受绝对概率偏差影响）
    gate_proxy = 0.25  # 固定：任何时刻都是后 25% 被挡

    # 平均 PWin（用于展示，不是用于温度计算）
    avg_pwin = pd.DataFrame(engine_pwins).mean(axis=1)
    wp_median = avg_pwin.median()

    # 推断 regime
    regime = folds[-1].get("regime", "range") if folds else "range"

    # 温度: 用排名中位数映射。高排名中位数 → 模型整体信心高 → 高温
    # wp_median_rank 范围 ~0-1，映射到 20-80
    temp = wp_median_rank * 60 + 20

    if temp < 35:
        level = "低温"
    elif temp < 55:
        level = "中温"
    else:
        level = "高温"

    model_ver = models.get("momentum", ("", ""))[1].split("_")[-1]

    return {"temp": temp, "level": level, "regime": regime, "gate_proxy": gate_proxy,
            "wp_median": wp_median, "wp_median_rank": wp_median_rank,
            "model_ver": model_ver, "n_stocks": len(stocks)}


# ═══════════════════════════════════════════════════════════════
# P2: 衰减
# ═══════════════════════════════════════════════════════════════

def compute_decay(feat, asof_date, lookback=4):
    """计算全市场衰减排名"""
    all_dates = sorted(feat["trade_date"].unique())
    target = (pd.to_datetime(asof_date) - pd.DateOffset(weeks=lookback)).strftime("%Y%m%d")
    past_dates = [d for d in all_dates if d <= target]
    if len(past_dates) < 2:
        return None
    past_date = past_dates[-1]

    common = sorted(set(feat[feat["trade_date"] == asof_date]["ts_code"]) &
                    set(feat[feat["trade_date"] == past_date]["ts_code"]))

    # 推断函数
    # 同模型对比(2026-08-06 方案2): 用同一最新模型预测"今日"和"4周前"，隔离特征驱动衰减。
    # 原实现 model≤日期 会跨模型版本(全量池重训后新旧PWin校准差异~5x)，使衰减榜被模型重训练污染。
    def predict(eng, date_str):
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(f"v6_{eng}_")])
        if not files:
            return None
        with open(os.path.join(MODEL_DIR, files[-1]), "rb") as f:
            model = pickle.load(f)
        return predict_pwin(model, feat, date_str, common)

    today_pwins = {}
    past_pwins = {}
    for eng in ["momentum", "reversion", "breakout"]:
        t = predict(eng, asof_date)
        p = predict(eng, past_date)
        if t is not None:
            today_pwins[eng] = t["pwin"]
        if p is not None:
            past_pwins[eng] = p["pwin"]

    if not today_pwins or not past_pwins:
        return None

    result = pd.DataFrame(index=pd.Index(common, name="ts_code"))
    for eng in ["momentum", "reversion", "breakout"]:
        if eng in today_pwins and eng in past_pwins:
            result[f"pwin_today"] = result.get("pwin_today", 0) + today_pwins[eng]
            result[f"pwin_past"] = result.get("pwin_past", 0) + past_pwins[eng]
            result["_count"] = result.get("_count", 0) + 1

    result["pwin_today"] /= result["_count"]
    result["pwin_past"] /= result["_count"]
    # 方案A(2026-08-06): 百分位排名变化替代百分比——消除低基数放大(0.008→0.079显示+867%失真)+免疫模型校准
    # decay = 全市场 PWin 百分位(今) − 百分位(前)，单位百分位点(pp)，与 Hermes"看全市场排位"解读一致
    result["rank_pct_past"] = result["pwin_past"].rank(pct=True) * 100
    result["rank_pct_today"] = result["pwin_today"].rank(pct=True) * 100
    result["decay_pct"] = result["rank_pct_today"] - result["rank_pct_past"]
    result = result.sort_values("decay_pct")
    result = result.drop(columns=["_count"])

    # 行业映射
    ind_df = pd.read_csv(INDUSTRY_CSV) if os.path.exists(INDUSTRY_CSV) else None
    if ind_df is not None:
        mapping = ind_df.set_index("ts_code")["l1_name"].to_dict()
        l2mapping = ind_df.set_index("ts_code")["l2_name"].to_dict()
        result["l1_name"] = result.index.map(mapping).fillna("-")
        # L2 行业（无 L2 的股票回退到 L1，无缝切换；2026-08-06 用户要求行业榜用 L2）
        l2m = result.index.map(l2mapping).fillna("")
        result["l2_name"] = l2m.where(l2m != "", result["l1_name"])
    else:
        result["l1_name"] = "-"
        result["l2_name"] = "-"

    # L2 行业统计（Top10 衰减 / Bottom10 上升）
    ind_stats = result.groupby("l2_name")["decay_pct"].agg(["mean", "count"]).sort_values("mean")

    # 股票名称
    db = sqlite3.connect(STOCK_DB)
    names = pd.read_sql("SELECT DISTINCT ts_code FROM stock_daily", db)
    db.close()
    # 从 industry CSV 取名称
    if ind_df is not None and "stock_name" in ind_df.columns:
        name_map = dict(zip(ind_df["ts_code"], ind_df["stock_name"]))
        result["name"] = result.index.map(name_map).fillna("")
    else:
        result["name"] = ""

    return {"ranking": result, "ind_stats": ind_stats, "today": asof_date, "past": past_date, "n": len(common)}


# ═══════════════════════════════════════════════════════════════
# HTML 生成
# ═══════════════════════════════════════════════════════════════

def temp_color(temp):
    """温度 → 颜色"""
    if temp < 35:
        return "#ef4444"
    elif temp < 55:
        return "#f59e0b"
    else:
        return "#16a34a"


def temp_bar(temp):
    """温度条 CSS"""
    pct = min(max(temp, 0), 100)
    return (
        f'<div style="background:#e2e8f0;border-radius:6px;height:12px;margin:10px 0">'
        f'<div style="background:linear-gradient(90deg,#ef4444,#f59e0b,#16a34a);'
        f'border-radius:6px;height:12px;width:{pct}%"></div></div>'
    )


def compute_fragility_info():
    """计算市场脆弱度 danger 信号；非 danger 或失败返回 None（不破坏日报）。

    复用 analysis/risk_assessment.compute_from_db() —— 与 run_analysis.py 晨间流程同源，
    读 sw_index_data.db 行业广度 + Tushare，完全自包含。
    已知限制: danger 按当前时点计算（compute_from_db 内部用 datetime.now()），
    手动用历史 --asof 生成旧报告时红条反映当下而非 asof 日；生产 cron（T+1 08:30）两者一致。
    """
    try:
        from analysis.risk_assessment import compute_from_db
        ra = compute_from_db()  # 默认 DB_PATH = data_storage/sw_index_data.db
        if not ra or ra.get("alert_level") != "danger":
            return None
        return {
            "alert_label": ra.get("alert_label", ""),
            "pos_cap": ra.get("pos_cap", 15),
            "down_pct": ra.get("down_pct"),
            "p1_strict": ra.get("p1_strict", False),
        }
    except Exception as e:
        print(f"⚠️ 脆弱度计算失败（忽略，不影响日报）: {e}", file=sys.stderr)
        return None


def generate_html(report_date, next_date, feat_date, price_date, model_ver, temp_info, decay_info, current_regime, danger_info=None, stale_days=0):
    """生成完整 HTML"""
    temp = temp_info["temp"]
    level = temp_info["level"]
    regime = temp_info.get("regime", current_regime)
    size, hist_t1, hist_d5 = POSITION_RULES.get((regime, level), ("3只 (半仓)", 0.45, 0.45))

    dt = datetime.strptime(report_date, "%Y%m%d")
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]
    next_dt = datetime.strptime(next_date, "%Y%m%d")
    next_weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][next_dt.weekday()]

    # ── 温度卡片 ──
    temp_html = f"""
    <div class="section">
      <div class="section-title">🌡️ 市场温度计</div>
      <div style="text-align:center;padding:10px 0">
        <div style="font-size:3rem;font-weight:800;color:{temp_color(temp)};line-height:1.2">{fmt_num(temp, 1)}</div>
        <div style="font-size:1.2rem;color:{temp_color(temp)};font-weight:600">{level}</div>
      </div>
      {temp_bar(temp)}
      <div style="display:flex;justify-content:space-between;font-size:.75rem;color:#94a3b8">
        <span>0 极寒</span><span>50 适中</span><span>100 偏热</span>
      </div>
      <div class="dashboard-row" style="margin-top:16px">
        <span>Gate 阻塞率: <b>{fmt_pct(temp_info['gate_proxy'])}</b></span>
        <span>P(Win) 中位数: <b>{fmt_num(temp_info['wp_median'], 4)}</b></span>
        <span>覆盖: <b>{temp_info['n_stocks']}</b> 只</span>
      </div>
      <div class="dashboard-row">
        <span>推断 Regime: <b>{regime}</b></span>
        <span>模型版本: <b>{model_ver}</b></span>
        <span>特征日期: <b>{feat_date}</b></span>
      </div>
      <div style="margin-top:14px;padding:12px;background:#f8fafc;border-radius:8px;font-size:.88rem;color:#475569">
        > {TEMP_ADVICE.get(level, '')}
      </div>
    </div>"""

    # ── 核心结论 ──
    conclusion_color = "#ef4444" if temp < 35 else ("#f59e0b" if temp < 55 else "#16a34a")
    conclusion_html = f"""
    <div class="conclusion" style="border-left-color:{conclusion_color}">
      <div class="conclusion-title" style="color:{conclusion_color}">🎯 温度 {fmt_num(temp, 1)}/100 · {level} · {regime}</div>
      <div class="conclusion-sub">建议仓位: <b>{size}</b> · 行业覆盖 ≥ 4 · 该配置历史 T+1 胜率 {fmt_pct(hist_t1)} · D+5 胜率 {fmt_pct(hist_d5)}</div>
    </div>"""

    # ── 数据滞后横幅（统一自愈仍失败时显示，消费端自证，杜绝静默错数据）──
    stale_html = ""
    if stale_days > 0:
        stale_html = f"""
    <div style="margin:16px 0 20px;padding:14px 16px;border:2px solid #f59e0b;border-left:6px solid #f59e0b;border-radius:10px;background:#fffbeb;color:#92400e">
      <div style="font-weight:800;font-size:1.05rem">⚠️ 数据滞后 {stale_days} 个交易日</div>
      <div style="margin-top:6px;font-size:.82rem;line-height:1.6">
        自动补拉数据失败，本报告基于 <b>{price_date}</b> 及更早行情。最新交易日数据未就绪，以下全部指标仅供参考，谨慎交易。
      </div>
    </div>"""

    # ── 脆弱度 danger 警示条（仅 danger 时显示，其他状态与旧版逐字一致）──
    danger_html = ""
    if danger_info:
        down_txt = fmt_pct(danger_info["down_pct"] / 100) if danger_info.get("down_pct") is not None else "N/A"
        strict_txt = "（广度严格模式）" if danger_info.get("p1_strict") else ""
        danger_html = f"""
    <div style="margin:16px 0 20px;padding:14px 16px;border:2px solid #dc2626;border-left:6px solid #dc2626;border-radius:10px;background:#fef2f2;color:#991b1b">
      <div style="font-weight:800;font-size:1.05rem">⚠️ 市场脆弱度 · {danger_info['alert_label']}</div>
      <div style="margin-top:6px;font-size:.85rem;line-height:1.6">
        行业广度崩塌{strict_txt} · 当日行业下跌比例 <b>{down_txt}</b> · 历史精确率 84.6%
      </div>
      <div style="margin-top:8px;padding:8px 10px;background:#fff;border-radius:6px;font-size:.8rem;color:#7f1d1d;line-height:1.6">
        脆弱度规则建议仓位 <b>≤{danger_info['pos_cap']}%</b>（优先于 P3）｜下方 P3 仓位规则为 98-fold 历史查表，两者独立、不联动；danger 期间请以 ≤15% 为纪律底线
      </div>
    </div>"""

    # ── 衰减榜 ──
    if decay_info:
        ranking = decay_info["ranking"]
        ind_stats = decay_info["ind_stats"]

        top_rows = ""
        for i, (idx, row) in enumerate(ranking.head(10).iterrows(), 1):
            dc = row['decay_pct']
            color = '#dc2626' if dc < -10 else ('#f59e0b' if dc < 0 else '#16a34a')
            top_rows += f"<tr><td>{i}</td><td style='font-size:.8rem'>{idx}</td><td style='font-size:.78rem;color:#64748b'>{row.get('name','')}</td><td style='font-size:.78rem;color:#64748b'>{row['l1_name']}</td><td style='font-size:.76rem'>{row['pwin_past']:.3f}</td><td style='font-size:.76rem'>{row['pwin_today']:.3f}</td><td style='color:{color};font-weight:600'>{dc:+.1f}pp</td></tr>"

        rise_rows = ""
        for i, (idx, row) in enumerate(ranking.tail(10)[::-1].iterrows(), 1):
            dc = row['decay_pct']
            color = '#dc2626' if dc < -10 else ('#f59e0b' if dc < 0 else '#16a34a')
            rise_rows += f"<tr><td>{i}</td><td style='font-size:.8rem'>{idx}</td><td style='font-size:.78rem;color:#64748b'>{row.get('name','')}</td><td style='font-size:.78rem;color:#64748b'>{row['l1_name']}</td><td style='font-size:.76rem'>{row['pwin_past']:.3f}</td><td style='font-size:.76rem'>{row['pwin_today']:.3f}</td><td style='color:{color};font-weight:600'>{dc:+.1f}pp</td></tr>"

        # L2 行业双榜: 衰减 TOP10 + 逆势上升 TOP10（2026-08-06 用户要求）
        ind_down_rows = ""
        for idx, row in ind_stats.head(10).iterrows():
            color = "#dc2626" if row["mean"] < -10 else ("#f59e0b" if row["mean"] < 0 else "#16a34a")
            ind_down_rows += f"<tr><td style='font-size:.78rem'>{idx}</td><td>{int(row['count'])}</td><td style='color:{color};font-weight:600'>{row['mean']:+.1f}pp</td></tr>"
        ind_up_rows = ""
        for idx, row in ind_stats.tail(10)[::-1].iterrows():
            color = "#dc2626" if row["mean"] < -10 else ("#f59e0b" if row["mean"] < 0 else "#16a34a")
            ind_up_rows += f"<tr><td style='font-size:.78rem'>{idx}</td><td>{int(row['count'])}</td><td style='color:{color};font-weight:600'>{row['mean']:+.1f}pp</td></tr>"

        # 全量数据 JSON（供前端搜索，~4951条×8字段≈400KB，可接受）
        rank_cols = ["ts_code", "name", "l1_name", "pwin_past", "pwin_today",
                     "rank_pct_past", "rank_pct_today", "decay_pct"]
        rank_json = ranking.reset_index()[rank_cols].to_json(orient="records", force_ascii=False)
        ind_json = ind_stats.reset_index().to_json(orient="records", force_ascii=False)

        decay_html = f"""
    <div class="section">
      <div class="section-title">📉 Alpha Decay 全市场衰减榜</div>
      <div style="font-size:.82rem;color:#94a3b8;margin-bottom:16px">
        对比: {decay_info['past']} → {decay_info['today']}（4周间隔）· 覆盖 {decay_info['n']} 只股票
      </div>
      <div class="section">
        <div class="section-title">📘 V6 Alpha Decay 指标使用方法</div>
        <div style="font-size:.82rem;color:#475569;line-height:1.6;padding:10px;background:#f0fdf4;border-radius:8px;">
          <b>核心逻辑：相对价值</b><br />
          Alpha Decay 是三个趋势引擎的加权值，非概率校准。绝对数（如 &lt;0.5 或 &gt;0.9）无独立意义；<b>全市场排名变化</b>才是有效信号。<br />
          <br>
          <b>✅ 有效用法：</b><br />
          - <b>选股过滤</b>：看相对排名（前25% vs 后75%），回避跌入后75%的股票<br />
          - <b>板块轮动</b>：行业衰减水平横向比较（资金流向）<br />
          - <b>市场温度</b>：高温（temp&gt;55）可加仓、低温（temp&lt;35）轻仓；回测口径 Gate 阻塞率仅作 regime 参考（range 市≈93%、bear 市≈34%）<br />
          <br>
          <b>❌ 无效用法：</b><br />
          - <b>连板/事件驱动股</b>（V6 无封单量/龙虎榜特征，PWin 暴跌是模型"失去特征预测能力"而非"必跌"，需独立接力框架）<br />
          - 依赖 <b>PWin 绝对值</b> 做决策（IC 仅 0.11，无独立预测能力）
        </div>
      </div>
      <!-- 搜索框 -->
      <div style="margin-bottom:14px">
        <input type="text" id="decaySearch" placeholder="🔍 输入股票代码/名称/行业 搜索衰减排名..."
          style="width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:.9rem;outline:none"
          oninput="filterDecay()">
        <div id="decayResult" style="margin-top:8px;font-size:.82rem;color:#475569"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <div style="font-weight:700;color:#dc2626;margin-bottom:8px">🔴 衰减 TOP 10</div>
          <table class="data-table">
            <tr><th>#</th><th>代码</th><th>名称</th><th>行业</th><th>PWin(前)</th><th>PWin(今)</th><th>变化(pp)</th></tr>
            {top_rows}
          </table>
        </div>
        <div>
          <div style="font-weight:700;color:#16a34a;margin-bottom:8px">🟢 逆势上升 TOP 10</div>
          <table class="data-table">
            <tr><th>#</th><th>代码</th><th>名称</th><th>行业</th><th>PWin(前)</th><th>PWin(今)</th><th>变化(pp)</th></tr>
            {rise_rows}
          </table>
        </div>
      </div>
      <div style="margin-top:20px">
        <div style="font-weight:700;margin-bottom:8px">行业衰减排名（L2）</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <div>
            <div style="font-weight:700;color:#dc2626;margin-bottom:8px">🔴 行业衰减 TOP 10（L2）</div>
            <table class="data-table">
              <tr><th>行业</th><th>股票数</th><th>平均变化(pp)</th></tr>
              {ind_down_rows}
            </table>
          </div>
          <div>
            <div style="font-weight:700;color:#16a34a;margin-bottom:8px">🟢 行业逆势上升 TOP 10（L2）</div>
            <table class="data-table">
              <tr><th>行业</th><th>股票数</th><th>平均变化(pp)</th></tr>
              {ind_up_rows}
            </table>
          </div>
        </div>
      </div>
      <div style="margin-top:14px;padding:12px;background:#f8fafc;border-radius:8px;font-size:.88rem;color:#475569">
        > 使用方式: ① 搜索你的持仓看排名 · ② 从 🔴/🟢 两端各挑 2-3 只加自选观察 · ③ 避开衰减最严重的行业<br>
        > <b>📌 排名口径</b>: 全市场排名 = 按 <b>PWin 百分位变化(pp)</b> 的升序排名。<br>
        > &nbsp;&nbsp;变化 = <b>全市场 PWin 分位(今) − 分位(前)</b>，单位百分位点(pp)，非百分比、非 PWin 绝对值。<br>
        > &nbsp;&nbsp;排名 <b>1 = 衰减最严重（最该回避）</b> → 排名 N = 逆势上升最猛。数字越小衰减越重。
      </div>
    </div>
<script>
var decayData = {rank_json};
var indData = {ind_json};
function filterDecay() {{
  var q = document.getElementById('decaySearch').value.toLowerCase();
  var result = document.getElementById('decayResult');
  if (!q || q.length < 1) {{ result.innerHTML = ''; return; }}
  var matches = decayData.filter(function(r) {{
    return r.ts_code.toLowerCase().indexOf(q) >= 0
        || (r.name || '').toLowerCase().indexOf(q) >= 0
        || (r.l1_name || '').toLowerCase().indexOf(q) >= 0;
  }});
  if (matches.length === 0) {{
    result.innerHTML = '<span style=color:#dc2626>未找到匹配股票</span>';
  }} else {{
    var html = '<table class=data-table style=margin-top:4px><tr><th>代码</th><th>名称</th><th>行业</th><th>PWin(前)</th><th>PWin(今)</th><th>分位(前)</th><th>分位(今)</th><th>变化(pp)</th><th>排名(变化)<sup title="按全市场PWin百分位变化升序: 1=最衰减/最该回避, N=逆势上升最猛">?</sup></th></tr>';
    for (var i = 0; i < Math.min(matches.length, 30); i++) {{
      var r = matches[i];
      var rank = decayData.indexOf(r) + 1;
      var color = r.decay_pct < -10 ? '#dc2626' : r.decay_pct < 0 ? '#f59e0b' : '#16a34a';
      html += '<tr><td>' + r.ts_code + '</td><td>' + (r.name||'') + '</td><td>' + r.l1_name + '</td><td>' + r.pwin_past.toFixed(4) + '</td><td>' + r.pwin_today.toFixed(4) + '</td><td>' + r.rank_pct_past.toFixed(0) + '%</td><td>' + r.rank_pct_today.toFixed(0) + '%</td><td style=color:' + color + ';font-weight:600>' + (r.decay_pct>0?'+':'') + r.decay_pct.toFixed(1) + 'pp</td><td>' + rank + '/' + decayData.length + '</td></tr>';
    }}
    html += '</table>';
    if (matches.length > 30) html += '<div style=color:#94a3b8;font-size:.75rem>显示前30条，共' + matches.length + '条匹配</div>';
    result.innerHTML = html;
  }}
}}
function filterIndustry() {{
  var q = document.getElementById('indSearch').value.toLowerCase();
  var rows = document.querySelectorAll('#indTable tr');
  for (var i = 1; i < rows.length; i++) {{
    var cell = rows[i].cells[0];
    if (cell) {{
      var txt = cell.textContent.toLowerCase();
      rows[i].style.display = txt.indexOf(q) >= 0 ? '' : 'none';
    }}
  }}
}}
</script>"""
    else:
        decay_html = '<div class="section"><div class="section-title">📉 Alpha Decay 衰减榜</div><p>⚠️ 数据不足，无法计算</p></div>'

    # ── 仓位规则 ──
    rule_rows = ""
    for (r, tq), (sz, wr1, wr5) in POSITION_RULES.items():
        is_current = (r == regime and tq == level)
        bg = ' style="background:#eef2ff"' if is_current else ""
        marker = " ← 当前" if is_current else ""
        rule_rows += f"<tr{bg}><td>{r}</td><td>{tq}</td><td><b>{sz}{marker}</b></td><td>{fmt_pct(wr1)}</td><td>{fmt_pct(wr5)}</td></tr>"

    rules_html = f"""
    <div class="section">
      <div class="section-title">📊 仓位配置规则</div>
      <div style="font-size:.82rem;color:#94a3b8;margin-bottom:16px">基于 98 folds · 2.5 年历史 · 三维分位回归</div>
      <table class="data-table">
        <tr><th>Regime</th><th>温度</th><th>建议仓位</th><th>历史 T+1 WR</th><th>历史 D+5 WR</th></tr>
        {rule_rows}
      </table>
      <div style="margin-top:14px;padding:12px;background:#fef2f2;border-radius:8px;font-size:.85rem;color:#dc2626">
        ⚠️ Bull+高温是陷阱！历史 D+5 仅 32.3%。过度分散反而降胜率：4-6 个行业最优 (55.9% WR)。
      </div>
    </div>"""

    # ── 策略建议 ──
    strategies = []
    if temp > 55:
        strategies.append(f"✅ 温度偏高 ({fmt_num(temp, 1)}) — 可参与，但注意 Regime={regime} 下过热风险")
        if regime == "bull":
            strategies.append("⚠️ Bull+高温历史 D+5 仅 32.3% — 缩短持仓周期，偏向 T+1 快进快出")
    elif temp > 35:
        strategies.append(f"🟡 温度适中 ({fmt_num(temp, 1)}) — 适度参与，仓位 {size}")
    else:
        strategies.append(f"🔴 温度偏低 ({fmt_num(temp, 1)}) — 防御为主，建议 {size}")

    if decay_info:
        top_ind = decay_info["ind_stats"].index[0]
        strategies.append(f"🚫 规避行业: {top_ind}（衰减最严重）")

    strat_items = "".join(f"<div class='plan-item'>{s}</div>" for s in strategies)

    strategy_html = f"""
    <div class="section">
      <div class="section-title">📋 明日交易策略</div>
      {strat_items}
      <div class="plan-item" style="margin-top:10px">
        📊 仓位: <b>{size}</b> · 行业覆盖 ≥ 4 · 单票 ≤ 25%
      </div>
      <div class="plan-item">
        📖 使用流程: 温度→仓位→衰减榜→选股 (PWin > 0.70) → 确保 ≥ 4 行业 → 建仓
      </div>
    </div>"""

    # ── 完整页面 ──
    report_dt = datetime.strptime(report_date, "%Y%m%d")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>V6日报 {report_date}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Helvetica Neue',system-ui,sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.6;padding:16px;max-width:900px;margin:0 auto;-webkit-font-smoothing:antialiased}}

.header{{background:linear-gradient(135deg,#1e293b,#334155);color:#fff;padding:24px 28px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:1.3rem;font-weight:700;letter-spacing:-0.3px}}
.header .time{{font-size:.82rem;color:#94a3b8;margin-top:4px}}
.header .meta{{font-size:.72rem;color:#64748b;margin-top:6px}}

.conclusion{{background:#fff;border-left:4px solid #ef4444;border-radius:10px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.conclusion-title{{font-size:1.05rem;font-weight:700}}
.conclusion-sub{{font-size:.88rem;color:#64748b;margin-top:4px}}

.section{{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.section-title{{font-size:1.05rem;font-weight:700;padding-bottom:12px;margin-bottom:16px;border-bottom:2px solid #e2e8f0}}

.dashboard-row{{font-size:.84rem;line-height:1.8;padding:2px 0}}
.dashboard-row span{{margin-right:16px}}

.data-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
.data-table th{{text-align:left;padding:6px 8px;border-bottom:2px solid #e2e8f0;color:#64748b;font-weight:600}}
.data-table td{{padding:6px 8px;border-bottom:1px solid #f1f5f9}}

.plan-item{{font-size:.88rem;padding:4px 0;color:#475569}}

.footer{{text-align:center;padding:20px;font-size:.75rem;color:#94a3b8;line-height:1.8}}
.footer a{{color:#6366f1;text-decoration:none}}

@media(max-width:600px){{
  body{{padding:10px}}
  .header{{padding:16px 20px}}
  .section{{padding:14px 16px}}
  .data-table{{font-size:.72rem}}
}}
</style>
</head>
<body>

<div class="header">
  <h1>📊 V6 每日交易日报</h1>
  <div class="time">分析基准: {report_date} {weekday}（A股收盘后）</div>
  <div class="time">下一交易日: {next_date} {next_weekday}</div>
  <div class="meta">特征: {feat_date} · 行情: {price_date} · 模型: {model_ver}</div>
</div>

{stale_html}
{danger_html}
{conclusion_html}
{temp_html}
{decay_html}
{rules_html}
{strategy_html}

<div class="footer">
  V6 Gate Engine + Alpha Decay + 仓位配置规则<br>
  98 folds · 2.5 年历史回测 · 不构成投资建议<br>
  模型: momentum / reversion / breakout @ {model_ver}
</div>

</body>
</html>"""

    return html


def generate_telegram_msg(report_date, next_date, temp_info, decay_info, danger_info=None, stale_days=0):
    """生成 Telegram 推送消息"""
    temp = temp_info["temp"]
    level = temp_info["level"]
    regime = temp_info["regime"]
    size, hist_t1, hist_d5 = POSITION_RULES.get((regime, level), ("3只(半仓)", 0.45, 0.45))

    next_dt = datetime.strptime(next_date, "%Y%m%d")
    next_wd = ["周一", "周二", "周三", "周四", "周五"][next_dt.weekday()]

    report_dt = datetime.strptime(report_date, "%Y%m%d")
    report_wd = ["周一", "周二", "周三", "周四", "周五"][report_dt.weekday()]

    lines = [
        f"📊 <b>V6日报 {report_date} {report_wd} → {next_date} {next_wd}</b>",
        "",
        f"🌡️ 温度 <b>{fmt_num(temp, 1)}/100</b> {level} | 推断 Regime: <b>{regime}</b>",
        f"📈 建议仓位: <b>{size}</b> | 历史 T+1胜率 {fmt_pct(hist_t1)}",
    ]

    # 警告前缀置顶：数据滞后 优先于 脆弱度 danger（非触发不显示，保持原格式）
    warn_prefix = []
    if stale_days > 0:
        warn_prefix.append(f"⚠️ <b>数据滞后 {stale_days} 个交易日</b> · 基于 {report_date}，谨慎交易")
    if danger_info:
        warn_prefix.append(f"⚠️ <b>市场脆弱度 DANGER</b> · {danger_info['alert_label']} · 建议 ≤{danger_info['pos_cap']}% 仓位（优先于 P3）")
    if warn_prefix:
        lines[0:0] = warn_prefix + [""]

    if decay_info:
        ranking = decay_info["ranking"]
        ind_stats = decay_info["ind_stats"]
        worst = list(ind_stats.head(3).index)
        best = list(ind_stats.tail(3).index)
        lines.append(f"📉 衰减板块: {' '.join(worst)}")
        lines.append(f"🟢 逆势板块: {' '.join(best)}")
        # PWin 摘要（2026-08-06: 用户要求 Telegram 显示 P(Win) 值）
        if ranking is not None and not ranking.empty:
            top_d = ranking.head(1).iloc[0]
            top_r = ranking.tail(1).iloc[0]
            d_name = top_d.get("name", "") or top_d.name
            r_name = top_r.get("name", "") or top_r.name
            lines.append(f"🔻 最衰减 {d_name}: PWin {top_d['pwin_past']:.3f}→{top_d['pwin_today']:.3f} ({top_d['decay_pct']:+.0f}pp)")
            lines.append(f"🚀 最上升 {r_name}: PWin {top_r['pwin_past']:.3f}→{top_r['pwin_today']:.3f} ({top_r['decay_pct']:+.0f}pp)")

    # 警示
    if temp < 35:
        lines.append("⚠️ 温度极低，防御为主")
    elif regime == "bull" and level == "高温":
        lines.append("⚠️ Bull+高温=危险信号，历史D+5仅32.3%")

    url = f"{GITHUB_BASE}/v6_daily_{report_date}.html"
    lines.append("")
    lines.append(f'📄 <a href="{url}">完整报告</a>')

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V6 每日交易日报")
    parser.add_argument("--asof", type=str, default="", help="分析基准日 YYYYMMDD")
    parser.add_argument("--telegram", action="store_true", help="推送 Telegram")
    parser.add_argument("--deploy", action="store_true", help="git push 到 GitHub Pages")
    parser.add_argument("--stale-days", type=int, default=0,
                        help="数据滞后交易日数(>0 时报告顶部显示数据滞后横幅，由管道自愈失败时传入)")
    parser.add_argument("--final-json", type=str, default="",
                        help="审计 JSON 路径（全量池重训产物，默认模块 FINAL_JSON）")
    args = parser.parse_args()

    # ── 加载数据 ──
    feat, models, folds, industry_df, db_max, next_dates_data = load_data(final_json=args.final_json or None)
    all_dates = sorted(feat["trade_date"].unique())

    # 确定分析日期
    if args.asof:
        trading_date = args.asof
    else:
        trading_date = db_max  # stock_daily 最新日期
    # 特征日期
    feat_candidates = [d for d in all_dates if d <= trading_date]
    feat_date = feat_candidates[-1] if feat_candidates else all_dates[-1]

    # 下一个交易日
    if len(next_dates_data) > 0:
        next_date = next_dates_data["trade_date"].iloc[0]
    else:
        d = datetime.strptime(trading_date, "%Y%m%d") + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        next_date = d.strftime("%Y%m%d")

    # 模型版本
    model_ver = models.get("momentum", ("", ""))[1].split("_")[-1]
    price_date = db_max

    print(f"分析日期: {trading_date} (特征: {feat_date}, 行情: {price_date})", file=sys.stderr)
    print(f"下一交易日: {next_date}", file=sys.stderr)

    # ── P1: 温度 ──
    temp_info = compute_temperature(feat, models, folds, feat_date)
    print(f"温度: {fmt_num(temp_info['temp'], 1)}/100 {temp_info['level']} Regime={temp_info['regime']}", file=sys.stderr)

    # ── P2: 衰减 ──
    decay_info = compute_decay(feat, feat_date)
    if decay_info:
        print(f"衰减: {decay_info['n']} 只, 周期 {decay_info['past']}→{decay_info['today']}", file=sys.stderr)

    # ── 脆弱度 danger（仅展示，与 P3 仓位规则独立不联动）──
    danger_info = compute_fragility_info()
    if danger_info:
        print(f"⚠️ 脆弱度 danger 触发: {danger_info['alert_label']} (≤{danger_info['pos_cap']}%)", file=sys.stderr)
    else:
        print("ℹ️ 脆弱度: normal / 非 danger，不显示警示条", file=sys.stderr)

    # ── 数据滞后（管道统一自愈失败时传入）──
    stale_days = getattr(args, "stale_days", 0) or 0
    if stale_days > 0:
        print(f"⚠️ 数据滞后 {stale_days} 个交易日，报告将带滞后横幅", file=sys.stderr)

    # ── 生成 HTML ──
    html = generate_html(trading_date, next_date, feat_date, price_date, model_ver,
                         temp_info, decay_info, temp_info["regime"], danger_info, stale_days)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    html_path = os.path.join(REPORTS_DIR, f"v6_daily_{trading_date}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML: {html_path}", file=sys.stderr)

    # ── Telegram ──
    if args.telegram:
        try:
            from notify.telegram_sender import send_single_message
            msg = generate_telegram_msg(trading_date, next_date, temp_info, decay_info, danger_info, stale_days)
            ok = send_single_message(msg, parse_mode="HTML")
            if ok:
                print("✅ Telegram 已推送", file=sys.stderr)
            else:
                print("❌ Telegram 推送失败", file=sys.stderr)
        except Exception as e:
            print(f"❌ Telegram 异常: {e}", file=sys.stderr)

    # ── Deploy ──
    if args.deploy:
        import subprocess
        # 添加新的 HTML 文件
        subprocess.run(["git", "add", f"reports/v6_daily/"], cwd=PROJECT_ROOT, capture_output=True)
        # 检查是否有变更
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
        if status.returncode != 0:
            subprocess.run(["git", "commit", "-m", f"📊 V6日报更新 {trading_date}"], cwd=PROJECT_ROOT, capture_output=True)
            result = subprocess.run(["git", "push"], cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ 已部署到 GitHub Pages: {GITHUB_BASE}/v6_daily_{trading_date}.html", file=sys.stderr)
            else:
                print(f"❌ git push 失败: {result.stderr}", file=sys.stderr)
        else:
            print("ℹ️ 无变更，跳过 git push", file=sys.stderr)


if __name__ == "__main__":
    main()
