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
# 2026-09-25 Hermes: 接入波段决策单 → 报告"择优合并"(非追加/非替换)
#   · 去重: 「推断Regime」→ ①阶段(业内标准口径 技术性牛熊±20% + 年线/半年线 + 实证超额)
#           「🎯主线(选池)」→ ②行业(阶段自适应臂)
#   · 下注对象: analysis/decision_card.py (REGIME_ARM 阶段自适应行业臂) + compute_card() 单次计算
#   · 契约: generate_html/generate_telegram_msg 末尾追加 card_info=None (向后兼容)
#   · 出口: HTML reports/v6_daily 与 TG 推送同源; decision_card 失败静默回退旧版式不阻断推送
#   · 择依据: backtest/report_signal_compare.py 实测 —
#           主线单独超额+0.39% / 全行业+2.46% / 震荡期「主线∩热度带」+2.88% / 过热期「主线」+2.77%
#   · 新增: ③个股候选(top8, 市值100-500亿, 含P≧20%) / ④出场 / ⑤纪律
#   · 保留: 温度/大盘风险/盘中预案/主升浪/情绪/PWin变化/板块强弱/影子/引擎归因
#   · decision_card 失败时静默回退旧版式(warn 到 stderr), 不阻断推送
# 2026-09-15 Claude: 推送/部署通道失败 → 出口非零 (pushed_failed + sys.exit(1))
#   背景: 09-05~09-14 Telegram token 401、09-14~ GitHub PAT 失效，旧版此处恒 exit 0，
#         上游 v6_daily_pipeline 只回显含 "✅" 的行 → 双重静默，11 天无人察觉。
#   契约变化: 调用方需把非零退出码当"未送达"处理（pipeline 已同步透传）。
#   新增 _redact(): git push 报错 stderr 可能带内嵌 token，落盘前抹除。
#   新增 --print-msg: 仅渲染 TG 消息到 stdout 不发送 —— 推送通道故障时
#                     用户仍能立刻看到日报正文（本次事故中 Telegram/GitHub 双断）。
# 2026-08-02 Claude: 初版, HTML+Telegram 双通道, P1温度+P2衰减+P3仓位
# 2026-08-03 Claude: 顶部接入脆弱度 danger 警示条（复用 analysis/risk_assessment.compute_from_db）
#               仅展示不联动: 非 danger/失败时红条为空, HTML/TG 与旧版一致
# 2026-08-03 Claude: 新增 --stale-days 数据滞后横幅（统一自愈失败时管道传入）
#               位置: header 之后、danger 红条之前; 橙黄警示样式, 消费端自证防静默错数据
#               下游: v6_daily_pipeline.py --report-only 自愈失败时传参
#               健康检查 FILE_GROUPS 已补 v6_daily_report.py
# 2026-08-05 Claude: 衰减榜标题下插入
# 2026-08-06 Claude: deploy+telegram 顺序改为 先部署→等Pages成功→再推Telegram(保证链接可用)
#               新增 _wait_for_pages_deploy 轮询 Actions API; 部署超时/失败仍推TG但附警告
# 2026-08-06 Claude: 上升榜加确定性过滤(当前分位≥75且上升) — 变化=赔率, 分位=确定性, 两者兼备才上榜
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
# 2026-08-07 Claude: 行业衰减排名(L2)区段增加查询框 — 与个股搜索一致，基于全量 indData 实时查询
#               显示 行业/股票数/平均变化(pp)/全行业排名; filterIndustry 改数据驱动(原行过滤stub未接线)
# 2026-08-14 Claude: P1 温度计重构 — 市场级原始水平+250日时间序列百分位复合(S1) 替换当日横截面分位
#               修复: 横截面分位中位数恒≈0.5 锁死温度~50(实盘49.6-50.0), (bull,低温)最高胜率档从未触发
#               新构造: S1=0.5*ts250(市场平均原始PWin)+0.5*ts250(MA20广度); 方向=反向(高温=过热=未来5日走弱)
#               分档阈值与POSITION_RULES 改从 backtest/v6_position_rules.json 数据驱动读取(缺失回退硬编码+旧阈值35/55)
#               契约: compute_temperature 8键不变; fallback 分支补 wp_median_rank 键; temp_color 语义翻转(低温绿/高温红)
#               健康检查: FILE_GROUPS 补 v6_thermometer_validation.py; HEALTH_CHECKS 补公式哨兵/规则JSON/序列缓存新鲜度
#               告警: 温度=模型当前视角, 跨模型版本不可直接比; 缓存按 model_ver 键控, 版本变化自动整窗重算
# 2026-08-14 Claude: 性能修复 — 新增 _predict_on_slice(切片直喂模型), _recompute_market_levels/_day_pwins
#               不再对全表(992万行)反复做 trade_date+isin 布尔掩码(2250次全表扫描→~15min→~3min)
# 2026-08-14 Claude: 规则表展示补 low_conf 标注 — JSON 规则值扩展为4元组(size,t1,d5,low_conf),
#               低样本(bear,<5 folds)cell 在 HTML 加"⚠️样本少"列; 消费端 *-unpack 兼容3/4元组
# 2026-08-19 Claude: 新增 Index Risk Gauge 常驻风险卡片 (Part B, 计划 enumerated-imagining-bunny)
#               新增 compute_index_risk(asof_date) — 读 sw_index_daily 4指数+sw_l2_index_daily L2 广度,
#               组合评分(analysis/index_risk.py) → 四色档位; 自包含, 失败返回 None 不破坏日报
#               修复: _market_level_series 历史 --asof 复算读到最新日温度 — 缓存命中 keep 未截断到 asof_date,
#               _build_composite_temps 前补 levels=levels[index<=asof_date] (2026-08-19 验证时发现)
#               HTML: {danger_html} 后插 {risk_html} 常驻卡片; TG: 高风险档消息头加预警, 常驻一行风险分
#               回测: 组合 score≥4 未来2-3日崩盘 lift 2.3, G1-G5 全过 (index_risk_gauge_validation_report.md)
#               健康检查: daily_bot_doc_check 需补 v6_daily_report.py 风险卡字段哨兵 (TODO A4)
# 2026-08-19 Claude: 新增明日盘中预案段 (复盘追问线程, 用户"盘中情景加入预案"提议)
#               方向3: 不预测"明天会跌", 而是 T-1 报告给出盘中条件触发器 — 触发即降仓/止损
#               数据: backtest/intraday_gap_backtest.py 回测 (2826 交易日, 事件=当日收盘大跌)
#               weak_gap≤-2.5%: 命中62.5%/40次/lift16.2; weak_low≤-4.0%: 命中70.5%/132次/lift18.3
#               08-19 深成低开-2.09%/创业板-2.70% 双预案均触发; 无 lookahead(open/low→close)
#               配置驱动: backtest/intraday_gap_rules.json (meta口径+disclaimer, preplans列表);
#               _load_intraday_rules() 全局缓存+JSON优先+硬编码回退, 同 _load_rules_config 模式
#               HTML: {risk_html} 后插 {preplan_html} 常驻预案表; TG: 常驻一行紧凑预案(名称→动作+命中率)
#               契约: generate_html/generate_telegram_msg 追加末尾位置参数 preplan_info=None, 向后兼容
#               健康检查: daily_bot_doc_check 需补 intraday_gap_rules.json FILE_GROUPS+哨兵 (task #15)
# 2026-09-25 Hermes: 新增两段 — ①A股情绪周期(市场结构/择时) ②影子组合·策略归因(复盘闭环)
#               ① analysis/emotion_cycle.py: EOD 算涨停/跌停/封板率/连板高度/晋级率/涨跌比 + 情绪分 + 三阶段
#                  验证 backtest/emotion_cycle_validation_report.md (2636 交易日): 情绪过热→未来偏弱(IC -0.06),
#                  跌停≥25家 T+1胜率60.2%(超跌反弹); 择时叠加 vs 满仓基线 +0.27pp
#               ② analysis/shadow_portfolio.py: 回测归因(引擎/Regime/PWin校准)+实盘影子组合(JSONL 前向跟踪)
#                  momentum D+5 51.5%/+1.95% vs reversion 45.4%/+0.37%; PWin≥0.75 实胜率 51.1%(单调校准)
#               契约: generate_html/generate_telegram_msg 追加末尾位置参数 emotion_info/attr_info/shadow_info=None, 向后兼容
#               产物: data_storage/shadow_portfolio_log.jsonl (幂等, 同日不重复记录)
# 2026-09-25 Hermes: 新增第③段 — 🌊 主升浪雷达(波段层), 面向用户"抓主升浪"交易风格
#               analysis/market_wave.py: ZigZag 摆动识别主升浪 + 当前状态判定(进行中/中·回调/初期/底部/回调)
#               backtest/entry_style_backtest.py: 左侧vs右侧信号质量 + 加仓节奏(总资金口径) + "入场太早"代价
#               ‼️ 结论: 指数级主升浪仅 0.6-1.1 次/年(用户预期2-3次需下沉行业/个股);
#                       左侧信号不劣于右侧(沪深300 56% vs 41%); 左侧后平均再跌10%/3-6周才见底;
#                       分批 vs 一次: 总收益接近, 但浮亏更浅(-6.6% vs -9.2%)、胜率更高(65% vs 56%)
# 2026-09-25 Hermes: 新增第④段 — 🎯 主线雷达(选池层), 承用户"主线"定义(题材/行业内普涨+强度+持续≥1月)
#               analysis/mainline.py: L2 行业 强度(中位收益)+广度(上涨占比) → 主线分; 含信号扫描
#               backtest/mainline_stock_scan.py: 个股级"一两周30%"驱动扫描(MFE 口径)
#               ‼️ 核心结论: ①板块动量对"10日≥30%"几乎无预测力(基线1.1%→最优1.6%);
#                          ②30% 真正来源=连板: 连板≥3 MFE中位+29.2%/≥30%命中49.4%(15.5x), ≥2 为36.1%;
#                          ③连板≥3 收盘中位仅+5.9% vs MFE+29.2% → 卖出纪律决定实际收益;
#                          ④故"主线"定位为**选池**, 入场信号靠连板/涨停形态。
─────────────
"""

import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
import time as time_module
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 2026-09-25 Hermes: 情绪周期(择时层) + 影子组合/归因(复盘层) + 主升浪雷达(波段层)
from analysis.emotion_cycle import compute_emotion                      # noqa: E402
from analysis.shadow_portfolio import record_picks, attribute_backtest, score_log  # noqa: E402
from analysis.market_wave import wave_radar                            # noqa: E402
from analysis.mainline import current_mainlines                        # noqa: E402

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
    "低温": "市场不拥挤、模型精选胜率较高（实证低温档未来 5 日偏强）。可积极建仓，遵守下方仓位规则。",
    "中温": "市场热度适中。适度参与，选股时关注 P(Win) 高分位的高置信度标的。",
    "高温": "市场过热，未来 5 日历史表现偏弱（过热风险）。建议降仓、偏向 T+1 快进快出、分散行业。",
}

GITHUB_BASE = "https://stranger971020.github.io/trend-trading-system/reports/v6_daily"

# ── 温度计重构(2026-08-14) 配置: 规则表 JSON 优先, 硬编码回退 ──
RULES_JSON_PATH = os.path.join(SCRIPT_DIR, "v6_position_rules.json")
INTRADAY_RULES_PATH = os.path.join(SCRIPT_DIR, "intraday_gap_rules.json")
MARKET_SERIES_PATH = os.path.join(PROJECT_ROOT, "data_storage", "v6_market_level_series.parquet")
TS_WINDOW = 250          # 时间序列百分位回看窗口(交易日)
TS_MIN_PERIODS = 60      # 窗口最短有效期
_FALLBACK_THRESHOLDS = {"low": 35.0, "high": 55.0}   # JSON 缺失时的旧分档
_FALLBACK_COMPOSITE = "S1"
_RULES_CACHE = None      # _load_rules_config 全局装载缓存
_RULES_META_CACHE = {"n_folds": 98}   # JSON meta 装载缓存(规则表 fold 数)

# 盘中预案回退值 (2026-08-19 intraday_gap_backtest.py 回测, JSON 缺失时使用)
_INTRADAY_CACHE = None
_INTRADAY_META_FALLBACK = {
    "source": "backtest/intraday_gap_backtest.py",
    "period": "2015-01-05 ~ 2026-08-19, 2826 交易日",
    "event": "当日收盘大跌: 上证≤-2.5% 或 深成/创业板≤-4% (109 天, 基线 3.9%)",
    "backtest_date": "2026-08-19",
    "disclaimer": "预案是『盘中确认后行动』的触发器, 非收盘预测。触发意味着当日收大跌概率显著升高但仍有误报(约3-4成), 故建议『降仓』而非『清仓』——误报成本是少赚一点反弹。",
}
_INTRADAY_PREPLANS_FALLBACK = [
    {"key": "weak_gap_2.5", "name": "弱势指数开盘低开≥2.5%",
     "trigger": "开盘(9:30 可判): min(深成,创业板) 低开 ≤ -2.5%",
     "action": "开盘即降仓", "hit_rate": 0.625, "fp_rate": 0.375,
     "recall": 0.229, "triggers": 40, "lift": 16.2, "window": "开盘", "covered_20260819": True},
    {"key": "weak_low_4.0", "name": "弱势指数盘中触及 -4%",
     "trigger": "盘中任意时刻: min(深成,创业板) 盘中最低 ≤ -4.0%",
     "action": "止损 / 进一步减仓", "hit_rate": 0.705, "fp_rate": 0.295,
     "recall": 0.853, "triggers": 132, "lift": 18.3, "window": "盘中", "covered_20260819": True},
]


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


def _predict_on_slice(model, day_df):
    """对单日切片(已含 ts_code 列)推断 P(Win) → Series(index=ts_code)。

    与 predict_pwin 等价的轻量版, 但避免对全表(992万行)反复做布尔掩码 ——
    _recompute_market_levels 每天×3引擎直接喂 groupby 切片, 提速一个量级。
    """
    mf = model.feature_name_
    avail = [c for c in mf if c in day_df.columns]
    missing = [c for c in mf if c not in day_df.columns]
    X = day_df[avail].copy()
    for m in missing:
        X[m] = 0.0
    X = X[mf].fillna(0)
    proba = model.predict_proba(X)
    pwin = proba[:, 2] if proba.ndim == 2 and proba.shape[1] >= 3 else proba[:, 1]
    return pd.Series(pwin, index=day_df["ts_code"].values, name="pwin")


# ═══════════════════════════════════════════════════════════════
# 温度计共享核心(2026-08-14 重构): 市场级水平 + 时间序列百分位
# 验证脚本 v6_thermometer_validation.py 与生产共用此组函数, 保证单一公式来源。
# ═══════════════════════════════════════════════════════════════

def _model_ver(models):
    return models.get("momentum", ("", ""))[1].split("_")[-1]


def _unwrap_model(m):
    """模型可能是裸 LightGBM 或 (model, filename) 元组, 统一解包。"""
    return m[0] if isinstance(m, tuple) else m


def _ts_rank(series, window=TS_WINDOW, min_periods=TS_MIN_PERIODS):
    """250日滚动百分位(只含当日及之前 → 因果), 返回 0~100。"""
    return series.rolling(window, min_periods=min_periods).rank(pct=True) * 100


def _build_composite_temps(df):
    """给含 m_mean_pwin/breadth/momb 的市场级水平 df 增加 ts_* 列与 S1/S2/S3 复合温度列。"""
    out = df.copy()
    for col in ["m_mean_pwin", "m_med_pwin", "breadth", "momb"]:
        out[f"ts_{col}"] = _ts_rank(out[col])
    # THERMOMETER_COMPOSITE_SENTINEL — 健康检查锚点: 禁止回退到当日横截面分位(中位数恒≈0.5 锁死温度)
    out["S1"] = 0.5 * out["ts_m_mean_pwin"] + 0.5 * out["ts_breadth"]
    out["S2"] = 0.5 * out["ts_m_mean_pwin"] + 0.5 * out["ts_momb"]
    out["S3"] = (out["ts_m_mean_pwin"] + out["ts_breadth"] + out["ts_momb"]) / 3
    return out


def _recompute_market_levels(feat, models, dates):
    """对给定交易日序列, 用同一最新模型推断三引擎 PWin, 计算市场级水平聚合。

    Returns:
        DataFrame(date, m_mean_pwin, m_med_pwin, breadth, momb, temp_prod, n_stocks, model_ver)
        temp_prod 为旧生产公式(C1)参照值, 仅用于验证对照, 不参与新温度。
    """
    model_ver = _model_ver(models)
    want = set(dates)
    recs = []
    # 先筛目标交易日再 groupby, 避免对全表(992万行)反复掩码; 切片直接喂模型
    sub = feat[feat["trade_date"].isin(want)]
    for d, gdf in sub.groupby("trade_date", sort=False):
        stocks = sorted(gdf["ts_code"].unique())
        if len(stocks) < 100:
            recs.append(dict(date=d, m_mean_pwin=float("nan"), m_med_pwin=float("nan"),
                             breadth=float("nan"), momb=float("nan"), temp_prod=50.0,
                             n_stocks=len(stocks)))
            continue
        parts = []
        ranks = pd.DataFrame(index=stocks)
        for eng in ["momentum", "reversion", "breakout"]:
            if eng not in models:
                continue
            pw = _predict_on_slice(_unwrap_model(models[eng]), gdf)
            parts.append(pw)
            ranks[f"{eng}_rank"] = pw.rank(pct=True)
        if not parts:
            recs.append(dict(date=d, m_mean_pwin=float("nan"), m_med_pwin=float("nan"),
                             breadth=float("nan"), momb=float("nan"), temp_prod=50.0,
                             n_stocks=len(stocks)))
            continue
        pwin_avg = pd.concat(parts, axis=1).mean(axis=1)
        avg_rank = ranks.mean(axis=1)
        recs.append(dict(
            date=d,
            m_mean_pwin=pwin_avg.mean(),
            m_med_pwin=pwin_avg.median(),
            breadth=(gdf["ma20_dev"] > 0).mean() * 100,
            momb=(gdf["mom20"] > 0).mean() * 100,
            temp_prod=avg_rank.median() * 60 + 20,   # C1 生产旧公式(参照)
            n_stocks=len(stocks),
        ))
    out = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    out["model_ver"] = model_ver
    return out


def _day_pwins(feat, models, date_str):
    """asof 日三引擎 PWin 推断 → {eng: Series(ts_code->pwin)}。展示用, 不依赖缓存。"""
    day_df = feat[feat["trade_date"] == date_str]
    out = {}
    for eng in ["momentum", "reversion", "breakout"]:
        if eng not in models:
            continue
        out[eng] = _predict_on_slice(_unwrap_model(models[eng]), day_df)
    return out


def _market_level_series(feat, models, asof_date, window_days=300):
    """尾窗市场级水平序列(同最新模型)。缓存命中增量补算, model_ver 变化整窗重算, 窗口外现算。

    Returns:
        (levels_df, last_day_pwins)
        levels_df: date 索引, 含聚合列 + ts_* 列 + S1/S2/S3 复合温度
        last_day_pwins: {eng: Series} asof 日各引擎推断(展示用)
    """
    all_dates = sorted(feat["trade_date"].unique())
    di = {d: i for i, d in enumerate(all_dates)}
    idx = di.get(asof_date)
    if idx is None:
        raise ValueError(f"asof_date {asof_date} 不在特征矩阵交易日中")
    start = max(0, idx - window_days + 1)
    want_dates = all_dates[start: idx + 1]
    model_ver = _model_ver(models)

    cached = None
    if os.path.exists(MARKET_SERIES_PATH):
        try:
            cached = pd.read_parquet(MARKET_SERIES_PATH)
            cached["date"] = cached["date"].astype(str)
        except Exception:
            cached = None

    if cached is None or cached.empty or str(cached["model_ver"].iloc[-1]) != model_ver:
        levels = _recompute_market_levels(feat, models, want_dates)
    else:
        have = set(cached["date"])
        missing = [d for d in want_dates if d not in have]
        if missing:
            levels_new = _recompute_market_levels(feat, models, missing)
            cached = pd.concat([cached, levels_new], ignore_index=True)
        keep = cached[cached["date"] >= want_dates[0]].copy()
        levels = keep.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    # 覆写缓存(整窗 300 日, 避免无限增长)
    levels["model_ver"] = model_ver
    try:
        cols = ["date", "m_mean_pwin", "m_med_pwin", "breadth", "momb", "temp_prod", "n_stocks", "model_ver"]
        levels[cols].to_parquet(MARKET_SERIES_PATH, index=False)
    except Exception as e:
        print(f"⚠️ 温度序列缓存写入失败(忽略): {e}", file=sys.stderr)

    levels = levels.set_index("date").sort_index()
    # 截断到 asof_date: 缓存命中路径 keep 含 asof 之后的未来日期,
    # 历史 --asof 复算若不截断, iloc[-1] 会读到最新日温度 (2026-08-19 修复).
    levels = levels[levels.index <= asof_date]
    levels = _build_composite_temps(levels)

    last_day_pwins = _day_pwins(feat, models, asof_date)
    return levels, last_day_pwins


def _load_rules_config():
    """读 backtest/v6_position_rules.json → (POSITION_RULES, thresholds, composite)。

    缺失/损坏 → 回退现有硬编码 POSITION_RULES + 旧阈值(35,55) + "S1"。全局装载一次。
    thresholds = (low, high), composite = 温度公式名(S1/S2/S3)。
    """
    global _RULES_CACHE
    if _RULES_CACHE is not None:
        return _RULES_CACHE
    rules = dict(POSITION_RULES)
    thresholds = (_FALLBACK_THRESHOLDS["low"], _FALLBACK_THRESHOLDS["high"])
    composite = _FALLBACK_COMPOSITE
    if os.path.exists(RULES_JSON_PATH):
        try:
            with open(RULES_JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
            th = data.get("thresholds", {})
            if "low" in th and "high" in th:
                thresholds = (float(th["low"]), float(th["high"]))
            composite = data.get("meta", {}).get("formula", _FALLBACK_COMPOSITE)
            n_folds = data.get("meta", {}).get("n_folds")
            if n_folds:
                _RULES_META_CACHE["n_folds"] = int(n_folds)
            new_rules = {}
            for r in data.get("rules", []):
                # 第4元素 low_conf: 样本不足(fold 数<min_folds)的 cell 标真, 展示加 ⚠
                new_rules[(r["regime"], r["temp"])] = (r["size"], r["t1_wr"], r["d5_wr"], r.get("low_conf", False))
            if new_rules:
                rules = new_rules
        except Exception as e:
            print(f"⚠️ v6_position_rules.json 解析失败, 回退硬编码规则: {e}", file=sys.stderr)
    _RULES_CACHE = (rules, thresholds, composite)
    return _RULES_CACHE


def _rules_n_folds():
    """规则表 fold 数(JSON meta.n_folds, 缺失回退 98)。"""
    _load_rules_config()
    return _RULES_META_CACHE["n_folds"]


def _load_intraday_rules():
    """读 backtest/intraday_gap_rules.json → (meta, preplans)。

    缺失/损坏 → 回退硬编码 _INTRADAY_META_FALLBACK + _INTRADAY_PREPLANS_FALLBACK。
    全局装载一次。meta = 回测口径/disclaimer, preplans = 预案列表(触发器+动作+回测指标)。
    """
    global _INTRADAY_CACHE
    if _INTRADAY_CACHE is not None:
        return _INTRADAY_CACHE
    meta = dict(_INTRADAY_META_FALLBACK)
    preplans = [dict(p) for p in _INTRADAY_PREPLANS_FALLBACK]
    if os.path.exists(INTRADAY_RULES_PATH):
        try:
            with open(INTRADAY_RULES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("meta"):
                meta = dict(meta, **data["meta"])
            if data.get("preplans"):
                preplans = [dict(p) for p in data["preplans"]]
        except Exception as e:
            print(f"⚠️ intraday_gap_rules.json 解析失败, 回退硬编码预案: {e}", file=sys.stderr)
    _INTRADAY_CACHE = (meta, preplans)
    return _INTRADAY_CACHE


def _regime(folds):
    """当前 regime: 取最后一个 fold 的 regime(folds 周频, 生产沿用现状)。"""
    return folds[-1].get("regime", "range") if folds else "range"


def _fold_gate_proxy(folds, regime):
    """当前 regime 内 fold 的平均 gate_blocked_pct(回测口径 range≈93%/bear≈34%); 无 fold 回退 0.25。"""
    sub = [f for f in folds if f.get("regime") == regime and "gate_blocked_pct" in f]
    if not sub:
        return 0.25
    return sum(f["gate_blocked_pct"] for f in sub) / len(sub)


# ═══════════════════════════════════════════════════════════════
# P1: 温度
# ═══════════════════════════════════════════════════════════════

def compute_temperature(feat, models, folds, asof_date):
    """计算当日温度 — 市场级水平 + 250日时间序列百分位复合信号(S1)。

    2026-08-14 重构: 旧实现 temp = median(当日横截面 PWin 分位)×60+20,
    当日横截面百分位中位数恒≈0.5 → 温度锁死 ~50, (bull,低温)最高胜率档从未触发。
    新实现: 全市场平均原始 PWin + MA20 广度 的 250日滚动百分位等权合成(S1)。
    方向=反向: 高温=市场过热=未来5日走弱(负IC), 低温=健康=高胜率。
    分档阈值与公式名从 backtest/v6_position_rules.json 读取(缺失回退 35/55 + S1)。
    """
    stocks = sorted(feat[feat["trade_date"] == asof_date]["ts_code"].unique())
    if len(stocks) < 100:
        # fallback: 数据不足, 补齐契约 8 键(含 wp_median_rank)
        return {"temp": 50, "level": "中温", "regime": _regime(folds), "gate_proxy": 0.25,
                "wp_median": 0.5, "wp_median_rank": 0.5, "model_ver": "N/A", "n_stocks": len(stocks)}

    _, thresholds, composite = _load_rules_config()
    levels, last_day_pwins = _market_level_series(feat, models, asof_date)
    temp = float(levels[composite].iloc[-1])
    if pd.isna(temp):
        temp = 50.0  # 窗口不足时退中值, 不崩
    lo, hi = thresholds
    level = "低温" if temp < lo else ("中温" if temp < hi else "高温")
    regime = _regime(folds)
    gate_proxy = _fold_gate_proxy(folds, regime)
    model_ver = _model_ver(models)

    if last_day_pwins:
        pwin_df = pd.DataFrame(last_day_pwins)
        wp_median = pwin_df.mean(axis=1).median()
        wp_median_rank = pwin_df.rank(pct=True).mean(axis=1).median()
    else:
        wp_median = float("nan")
        wp_median_rank = float("nan")

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
    """温度 → 颜色。2026-08-14 方向翻转: 低温=健康(绿), 高温=过热(红)。"""
    _, (lo, hi), _ = _load_rules_config()
    if temp < lo:
        return "#16a34a"
    elif temp < hi:
        return "#f59e0b"
    else:
        return "#ef4444"


def temp_bar(temp):
    """温度条 CSS(绿→琥珀→红, 与方向语义一致)"""
    pct = min(max(temp, 0), 100)
    return (
        f'<div style="background:#e2e8f0;border-radius:6px;height:12px;margin:10px 0">'
        f'<div style="background:linear-gradient(90deg,#16a34a,#f59e0b,#ef4444);'
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


def compute_index_risk(asof_date):
    """计算 Index Risk Gauge 组合评分 → 风险档位（常驻风险卡片）。

    数据: sw_index_daily 上证/深成/创业板 + sw_l2_index_daily L2 广度。
    完全自包含: 失败返回 None，不破坏日报。用 asof_date（非 datetime.now()）消除历史复算失真。
    回测依据: 组合 score≥4 → 未来 2-3 日崩盘 lift 2.3，G1-G5 全过
    （backtest/index_risk_gauge_validation_report.md, 2026-08-19）。
    """
    try:
        from analysis.index_risk import (compute_risk_score_series, score_to_level,
                                         compute_l2_down_pct, RISK_COMPONENTS)
        conn = sqlite3.connect(STOCK_DB)
        try:
            sh = pd.read_sql_query(
                "SELECT trade_date, close, vol FROM sw_index_daily "
                "WHERE ts_code='000001.SH' ORDER BY trade_date", conn)
            sc = pd.read_sql_query(
                "SELECT trade_date, close FROM sw_index_daily "
                "WHERE ts_code='399001.SZ' ORDER BY trade_date", conn)
            cyb = pd.read_sql_query(
                "SELECT trade_date, close FROM sw_index_daily "
                "WHERE ts_code='399006.SZ' ORDER BY trade_date", conn)
            raw_l2 = pd.read_sql_query(
                "SELECT trade_date, ts_code, close FROM sw_l2_index_daily ORDER BY trade_date", conn)
        finally:
            conn.close()
        l2 = compute_l2_down_pct(raw_l2)
        sig = compute_risk_score_series(sh, sc, cyb, l2)
        if sig.empty:
            return None
        sig = sig[sig["trade_date"] <= asof_date]
        if sig.empty:
            return None
        row = sig.iloc[-1]
        score = int(row["score"])
        level, label, color, advice = score_to_level(score)
        comp_labels = dict(RISK_COMPONENTS)
        active = [comp_labels[name] for name, _ in RISK_COMPONENTS if row[name] == 1]
        return {
            "level": level, "label": label, "color": color, "score": score,
            "advice": advice, "asof": str(row["trade_date"]),
            "active_signals": active,
            "signals": {name: int(row[name]) for name, _ in RISK_COMPONENTS},
        }
    except Exception as e:
        print(f"⚠️ Index Risk 计算失败（忽略，不影响日报）: {e}", file=sys.stderr)
        return None


def compute_card(asof):
    """决策单计算（2026-09-25 Hermes）: HTML 与 Telegram 共用一次计算, 避免重复开销。"""
    try:
        from analysis.decision_card import decision_card as _dc
        return _dc(asof)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] decision_card 失败, 回退旧版式: {_e}", file=sys.stderr)
        return None


def generate_html(report_date, next_date, feat_date, price_date, model_ver, temp_info, decay_info, current_regime, danger_info=None, stale_days=0, risk_info=None, preplan_info=None, emotion_info=None, attr_info=None, shadow_info=None, wave_info=None, mainline_info=None, card_info=None):
    """生成完整 HTML"""
    rules, (lo, hi), composite = _load_rules_config()
    n_folds_meta = _rules_n_folds()
    temp = temp_info["temp"]
    level = temp_info["level"]
    regime = temp_info.get("regime", current_regime)
    size, hist_t1, hist_d5, *_ = rules.get((regime, level), ("3只 (半仓)", 0.45, 0.45))

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
        <span>0 低温(优选)</span><span>50 中温</span><span>100 过热(警惕)</span>
      </div>
      <div style="font-size:.72rem;color:#94a3b8;text-align:center;margin-top:4px">
        分档: 低温 &lt;{fmt_num(lo,1)} · 中温 {fmt_num(lo,1)}-{fmt_num(hi,1)} · 高温 &gt;{fmt_num(hi,1)} ｜ 温度=市场热度合成({composite})，高温=过热风险
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
    conclusion_color = "#16a34a" if temp < lo else ("#f59e0b" if temp < hi else "#ef4444")
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

    # ── Index Risk Gauge 大盘技术风险卡（常驻四色，2026-08-19 新增）──
    risk_html = ""
    if risk_info:
        rl = risk_info
        lvl_color = {"low": "#16a34a", "medium": "#f59e0b", "high": "#f97316", "extreme": "#dc2626"}.get(rl["level"], "#94a3b8")
        bar_color = {"low": "#16a34a", "medium": "#f59e0b", "high": "#f97316", "extreme": "#dc2626"}.get(rl["level"], "#94a3b8")
        bar_w = min(100, rl["score"] / 6 * 100)
        active_txt = "、".join(rl["active_signals"]) if rl["active_signals"] else "无 — 市场技术面正常"
        risk_html = f"""
    <div class="section">
      <div class="section-title">📊 大盘技术风险 <span style="font-weight:400;color:#94a3b8">(Index Risk Gauge)</span></div>
      <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
        <div style="font-size:2.2rem;font-weight:800;color:{lvl_color};line-height:1.2">{rl['label']}风险</div>
        <div style="font-size:.95rem;color:#64748b;line-height:1.7">
          风险分 <b>{rl['score']}</b>/6 · 基准 <b>{rl['asof']}</b><br>
          <span style="font-size:.78rem">组合评分≥4（高/极高）历史：未来2-3日崩盘命中率 ≈45%，G1-G5 回测全过</span>
        </div>
      </div>
      <div style="background:#e2e8f0;border-radius:6px;height:10px;margin:12px 0">
        <div style="background:{bar_color};border-radius:6px;height:10px;width:{bar_w:.0f}%"></div>
      </div>
      <div style="font-size:.82rem;color:#475569;line-height:1.7">
        触发信号: <b>{active_txt}</b><br>
        <span style="color:#94a3b8">MA破位:{'✅' if rl['signals'].get('A_MA破位') else '—'} 深度回撤:{'✅' if rl['signals'].get('B_深度回撤') else '—'} 广度崩塌:{'✅' if rl['signals'].get('C_广度崩塌') else '—'} 波动率扩张:{'✅' if rl['signals'].get('D_波动率扩张') else '—'} 突发破位:{'✅' if rl['signals'].get('E_突发破位') else '—'} 弱势领跌:{'✅' if rl['signals'].get('F_弱势领跌') else '—'}</span>
      </div>
      <div style="margin-top:10px;padding:10px 12px;background:#f8fafc;border-radius:8px;font-size:.85rem;color:#475569">
        > {rl['advice']}
      </div>
    </div>"""

    # ── 明日盘中预案（T-1 条件触发器, 2026-08-19 新增; 配置 intraday_gap_rules.json）──
    preplan_html = ""
    if preplan_info:
        _pmeta, _pplans = preplan_info
        pp_rows = ""
        for i, pp in enumerate(_pplans, 1):
            pp_rows += f"""<tr>
              <td style="font-size:.8rem;font-weight:600">{i}. {pp['name']}</td>
              <td style="font-size:.78rem;color:#475569">{pp['trigger']}<br><span style="color:#94a3b8">动作: <b>{pp['action']}</b></span></td>
              <td style="font-size:.78rem;color:#475569">命中 {pp.get('hit_rate', 0)*100:.0f}% / 误报 {pp.get('fp_rate', 0)*100:.0f}%<br><span style="color:#94a3b8">触发 {pp.get('triggers', '-')} 次 · lift {pp.get('lift', '-')}</span></td>
            </tr>"""
        preplan_html = f"""
    <div class="section">
      <div class="section-title">📈 明日盘中预案 <span style="font-weight:400;color:#94a3b8">(盘中确认后行动)</span></div>
      <div style="font-size:.78rem;color:#64748b;margin-bottom:8px">
        若明日盘中出现以下条件 → 立即执行对应动作。回测口径: {_pmeta.get('period', '2015-01-05 ~ 2026-08-19, 2826 交易日')}，事件: {_pmeta.get('event', '当日收盘大跌')}
      </div>
      <table style="width:100%;border-collapse:collapse">
        <tr style="background:#f1f5f9">
          <th style="padding:6px 8px;text-align:left;width:28%">预案</th>
          <th style="padding:6px 8px;text-align:left">触发条件 / 动作</th>
          <th style="padding:6px 8px;text-align:left;width:24%">历史表现</th>
        </tr>
        {pp_rows}
      </table>
      <div style="margin-top:8px;padding:8px 10px;background:#f8fafc;border-radius:6px;font-size:.75rem;color:#94a3b8;line-height:1.6">
        {_pmeta.get('disclaimer', '')}
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

        # 上升榜(2026-08-06): 确定性过滤 — 只显示 当前分位≥75 且 上升(变化>0) 的股票，按变化排
        # 理由: 变化代表赔率(未定价改善), 当前分位代表确定性; 两者都要才是有投资价值的上升标的
        rise_pool = ranking[(ranking["rank_pct_today"] >= 75) & (ranking["decay_pct"] > 0)]
        if rise_pool.empty:
            rise_rows = "<tr><td colspan='7' style='color:#94a3b8;text-align:center'>无高分位(≥75)上升标的 — 市场整体偏弱</td></tr>"
        else:
            rise_rows = ""
            for i, (idx, row) in enumerate(rise_pool.tail(10)[::-1].iterrows(), 1):
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
          - <b>市场温度</b>：低温（temp&lt;{fmt_num(lo,0)}）=市场不拥挤、可积极；高温（temp&gt;{fmt_num(hi,0)}）=过热、降仓（实证反向）；回测口径 Gate 阻塞率仅作 regime 参考（range 市≈93%、bear 市≈34%）<br />
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
          <div style="font-weight:700;color:#16a34a;margin-bottom:8px">🟢 逆势上升 TOP 10（分位≥75 + 上升）</div>
          <table class="data-table">
            <tr><th>#</th><th>代码</th><th>名称</th><th>行业</th><th>PWin(前)</th><th>PWin(今)</th><th>变化(pp)</th></tr>
            {rise_rows}
          </table>
        </div>
      </div>
      <div style="margin-top:20px">
        <div style="font-weight:700;margin-bottom:8px">行业衰减排名（L2）</div>
        <!-- 行业搜索框（2026-08-07: 与个股搜索类似，基于全量 L2 行业 indData 实时查询） -->
        <div style="margin-bottom:14px">
          <input type="text" id="indSearch" placeholder="🔍 输入行业名称 搜索行业排名（L2）..."
            style="width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:.9rem;outline:none"
            oninput="filterIndustry()">
          <div id="indResult" style="margin-top:8px;font-size:.82rem;color:#475569"></div>
        </div>
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
  var result = document.getElementById('indResult');
  if (!q || q.length < 1) {{ result.innerHTML = ''; return; }}
  var matches = indData.filter(function(r) {{
    return (r.l2_name || '').toLowerCase().indexOf(q) >= 0;
  }});
  if (matches.length === 0) {{
    result.innerHTML = '<span style=color:#dc2626>未找到匹配行业</span>';
  }} else {{
    var html = '<table class=data-table style=margin-top:4px><tr><th>行业(L2)</th><th>股票数</th><th>平均变化(pp)</th><th>全行业排名<sup title="按全市场L2行业平均PWin百分位变化升序: 1=衰减最重, N=上升最猛">?</sup></th></tr>';
    for (var i = 0; i < Math.min(matches.length, 30); i++) {{
      var r = matches[i];
      var rank = indData.indexOf(r) + 1;
      var color = r.mean < -10 ? '#dc2626' : r.mean < 0 ? '#f59e0b' : '#16a34a';
      html += '<tr><td>' + r.l2_name + '</td><td>' + r.count + '</td><td style=color:' + color + ';font-weight:600>' + (r.mean>0?'+':'') + r.mean.toFixed(1) + 'pp</td><td>' + rank + '/' + indData.length + '</td></tr>';
    }}
    html += '</table>';
    if (matches.length > 30) html += '<div style=color:#94a3b8;font-size:.75rem>显示前30条，共' + matches.length + '条匹配</div>';
    result.innerHTML = html;
  }}
}}
</script>"""
    else:
        decay_html = '<div class="section"><div class="section-title">📉 Alpha Decay 衰减榜</div><p>⚠️ 数据不足，无法计算</p></div>'

    # ── 仓位规则 ──
    bull_high = rules.get(("bull", "高温"), (None, None, 0.323))
    bull_high_d5 = bull_high[2] if bull_high[2] is not None else 0.323
    rule_rows = ""
    for (r, tq), (sz, wr1, wr5, *low) in rules.items():
        is_current = (r == regime and tq == level)
        bg = ' style="background:#eef2ff"' if is_current else ""
        marker = " ← 当前" if is_current else ""
        low_note = ' <span title="fold 样本不足, 仅供参考" style="color:#dc2626">⚠️样本少</span>' if (low and low[0]) else ""
        rule_rows += f"<tr{bg}><td>{r}</td><td>{tq}</td><td><b>{sz}{marker}</b></td><td>{fmt_pct(wr1)}</td><td>{fmt_pct(wr5)}</td><td>{low_note}</td></tr>"

    rules_html = f"""
    <div class="section">
      <div class="section-title">📊 仓位配置规则</div>
      <div style="font-size:.82rem;color:#94a3b8;margin-bottom:16px">基于 {n_folds_meta} folds · 2.5 年历史 · 温度带+Regime 回归</div>
      <table class="data-table">
        <tr><th>Regime</th><th>温度</th><th>建议仓位</th><th>历史 T+1 WR</th><th>历史 D+5 WR</th><th>备注</th></tr>
        {rule_rows}
      </table>
      <div style="margin-top:14px;padding:12px;background:#fef2f2;border-radius:8px;font-size:.85rem;color:#dc2626">
        ⚠️ Bull+高温是陷阱！历史 D+5 仅 {fmt_pct(bull_high_d5)}。过度分散反而降胜率：4-6 个行业最优 (55.9% WR)。
      </div>
    </div>"""

    # ── 策略建议 ──
    strategies = []
    if temp > hi:
        strategies.append(f"🔴 温度偏高 ({fmt_num(temp, 1)}) — 市场过热，未来5日偏弱，建议降仓")
        if regime == "bull":
            strategies.append(f"⚠️ Bull+高温历史 D+5 仅 {fmt_pct(bull_high_d5)} — 缩短持仓周期，偏向 T+1 快进快出")
    elif temp > lo:
        strategies.append(f"🟡 温度适中 ({fmt_num(temp, 1)}) — 适度参与，仓位 {size}")
    else:
        strategies.append(f"🟢 温度偏低 ({fmt_num(temp, 1)}) — 市场不拥挤、胜率较高，建议 {size}")

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

    # ── 🌊 主升浪雷达（波段层，2026-09-25 Hermes 新增）──
    wave_html = ""
    if wave_info and wave_info.get("states"):
        _tone_color = {"up": "#16a34a", "pullback": "#f59e0b", "early": "#3b82f6",
                       "bottom": "#94a3b8", "down": "#dc2626"}
        _rows = ""
        for st in wave_info["states"]:
            _c = _tone_color.get(st["tone"], "#94a3b8")
            _rows += (f"<tr><td>{st['name']}<span style='color:#94a3b8;font-size:.72rem'> {st['code']}</span></td>"
                      f"<td style='color:{_c};font-weight:700'>{st['status']}</td>"
                      f"<td>{st['since_pivot']*100:+.1f}%</td>"
                      f"<td>{st['dd_from_leg_peak']*100:+.1f}%</td>"
                      f"<td>{st['leg_trading_days']}日</td></tr>")
        _stat_rows = ""
        for name, s in (wave_info.get("stats") or {}).items():
            _stat_rows += (f"<tr><td>{name}</td><td>{s['mains_per_year']} 次/年</td>"
                           f"<td>{s['amp_median']*100:+.0f}%</td><td>{s['days_median']} 天</td></tr>")
        wave_html = f"""
    <div class="section">
      <div class="section-title">🌊 主升浪雷达 <span style="font-weight:400;color:#94a3b8">(波段层 · ZigZag 摆动)</span></div>
      <table class="data-table"><thead><tr><th>指数</th><th>状态</th><th>距腿低点</th><th>距腿内高点</th><th>本腿</th></tr></thead>
      <tbody>{_rows}</tbody></table>
      <div style="margin-top:14px;font-size:.82rem;font-weight:600;color:#64748b;margin-bottom:6px">历史主升浪(≥20%)参考</div>
      <table class="data-table"><thead><tr><th>指数</th><th>频率</th><th>幅度中位</th><th>持续中位</th></tr></thead>
      <tbody>{_stat_rows}</tbody></table>
      <div style="margin-top:12px;padding:12px;background:#f8fafc;border-radius:8px;font-size:.85rem;color:#475569;line-height:1.7">
        &gt; <b>波段操作框架</b>（回测支撑，见 entry_style_validation_report.md）：<br>
        · 指数级主升浪仅 <b>0.6-1.1 次/年</b> → 抓 2-3 次必须下沉到<b>行业/个股</b>；<br>
        · 左侧(超跌)信号胜率并不低于右侧(沪深300 左侧56% vs 右侧41%) → <b>左侧本身不是错，无纪律才是</b>；<br>
        · 左侧信号后平均还要再跌 <b>约10%</b>、<b>3-6周</b>才见底 → 首仓务必轻、必须分批；<br>
        · 分批 vs 一次：总收益接近，但分批<b>浮亏更浅(-6.6% vs -9.2%)、胜率更高(65% vs 56%)</b>。
      </div>
    </div>"""

    # ── 🎯 主线雷达（选池层，2026-09-25 Hermes 新增）──
    mainline_html = ""
    if mainline_info and mainline_info.get("ranking"):
        _ml_rows = ""
        for r in mainline_info["ranking"]:
            _f5 = f"{r['str5']*100:+.1f}%" if r.get("str5") is not None else "—"
            _f20 = f"{r['str20']*100:+.1f}%" if r.get("str20") is not None else "—"
            _b5 = f"{r['br5']*100:.0f}%" if r.get("br5") is not None else "—"
            _ml_rows += (f"<tr><td>{r['ind']}</td><td>{r['ml20']:.0f}</td><td>{_f5}</td>"
                         f"<td>{_f20}</td><td>{_b5}</td><td>{r['n']}</td></tr>")
        mainline_html = f"""
    <div class="section">
      <div class="section-title">🎯 主线雷达 <span style="font-weight:400;color:#94a3b8">(申万L2 · 选池层)</span></div>
      <table class="data-table"><thead><tr><th>行业</th><th>主线分</th><th>强度5日</th><th>强度20日</th><th>广度5日</th><th>只数</th></tr></thead>
      <tbody>{_ml_rows}</tbody></table>
      <div style="margin-top:12px;padding:12px;background:#fff7ed;border-radius:8px;font-size:.84rem;color:#7c2d12;line-height:1.7">
        ⚠️ <b>主线是「选池」，不是「入场信号」</b>。回测（10.4M 样本）显示：<br>
        · 板块动量对"未来10日个股≥30%"<b>几乎无预测力</b>（基线 1.1% → 最优信号 1.6%）；<br>
        · 真正的 30% 来源是<b>连板</b>：连板≥3 未来10日内摸到 +30% 的概率 <b>49.4%</b>（15.5x），连板≥2 为 36.1%；<br>
        · 但连板≥3 的<b>收盘</b>涨幅中位仅 +5.9% vs 区间最高 +29.2% → <b>波动极大，卖出纪律决定实际收益</b>。
      </div>
    </div>"""

    # ── A股情绪周期（市场结构层，2026-09-25 Hermes 新增）──
    emotion_html = ""
    if emotion_info:
        e = emotion_info
        _ph_color = {"上升期": "#16a34a", "分歧期": "#f59e0b", "退潮期": "#dc2626"}.get(e.get("phase"), "#94a3b8")
        _sr = "—" if e.get("seal_rate") is None else f"{e['seal_rate']*100:.0f}%"
        _pr = "—" if e.get("promotion_rate") is None else f"{e['promotion_rate']*100:.0f}%"
        _ur = "—" if e.get("updown_ratio") is None else f"{e['updown_ratio']:.2f}"
        _hist = "".join(f"<span style='margin-right:10px'>{h['date'][-4:]}={h['emo_score']}</span>"
                        for h in e.get("hist5", [])[-5:])
        emotion_html = f"""
    <div class="section">
      <div class="section-title">🎭 A股情绪周期 <span style="font-weight:400;color:#94a3b8">(市场结构 · EOD)</span></div>
      <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
        <div style="font-size:2.2rem;font-weight:800;color:{_ph_color};line-height:1.2">{e.get('emo_score')}<span style="font-size:1rem;color:#94a3b8">/100</span></div>
        <div style="font-size:1.05rem;font-weight:700;color:{_ph_color}">{e.get('phase')}</div>
        <div style="font-size:.78rem;color:#94a3b8">近5日: {_hist}</div>
      </div>
      <div class="dashboard-row" style="margin-top:12px">
        <span>涨停 <b>{e.get('limit_up')}</b></span>
        <span>跌停 <b>{e.get('limit_down')}</b></span>
        <span>封板率 <b>{_sr}</b></span>
        <span>最高连板 <b>{e.get('max_streak')}</b> 板</span>
        <span>连板家数 <b>{e.get('multi_board')}</b></span>
        <span>晋级率 <b>{_pr}</b></span>
        <span>涨/跌 <b>{e.get('up_count')}/{e.get('down_count')}</b> 比 {_ur}</span>
      </div>
      <div style="margin-top:12px;padding:12px;background:#f8fafc;border-radius:8px;font-size:.85rem;color:#475569">
        &gt; {e.get('advice', '')}
      </div>
      <div style="margin-top:8px;font-size:.72rem;color:#94a3b8">
        口径: 涨停=收盘封板(主板10%/创业板·科创板20%); 封板率=封板数/曾涨停数; 晋级率=昨日涨停今日仍涨停占比。方向经 2636 交易日回测(情绪过热→未来偏弱)。
      </div>
    </div>"""

    # ── 影子组合 · 策略归因（复盘层，2026-09-25 Hermes 新增）──
    shadow_html = ""
    if attr_info or shadow_info:
        _eng_rows = ""
        for x in (attr_info or {}).get("by_engine", []):
            _eng_rows += (f"<tr><td>{x['engine']}</td><td>{x['n']}</td>"
                          f"<td>{x['t1_wr']*100:.1f}%</td><td>{x['d5_wr']*100:.1f}%</td>"
                          f"<td>{x['d5_ret']:+.2f}%</td></tr>")
        _reg_rows = ""
        for x in (attr_info or {}).get("by_regime", []):
            _reg_rows += (f"<tr><td>{x['regime']}</td><td>{x['n']}</td>"
                          f"<td>{x['t1_wr']*100:.1f}%</td><td>{x['d5_ret']:+.2f}%</td></tr>")
        _cal_rows = ""
        for c in (attr_info or {}).get("calibration", []):
            _cal_rows += (f"<tr><td>{c['bucket']}</td><td>{c['n']}</td>"
                          f"<td>{c['stated']:.2f}</td><td>{c['actual_t1']*100:.1f}%</td>"
                          f"<td>{c['actual_d5']*100:.1f}%</td></tr>")
        _p = (attr_info or {}).get("portfolio", {})
        _port_line = ""
        if _p:
            _port_line = (f"<b>{_p['n_days']}</b> 日 · 累计 <b>{_p['total_ret']*100:+.1f}%</b> · "
                          f"日胜率 <b>{_p['win_days']*100:.1f}%</b> · 最大回撤 <b>{_p['max_dd']*100:.1f}%</b> · "
                          f"日均 <b>{_p['avg_daily_bp']:+.0f}bp</b> <span style='color:#94a3b8'>(未计交易成本)</span>")
        _s = shadow_info or {}
        if _s.get("n_settled_t1"):
            _shadow_line = (f"实盘影子: 已结算 <b>{_s['n_settled_t1']}</b> 笔 · T+1 胜率 "
                            f"<b>{_s['t1_wr']*100:.1f}%</b> · 累计 <b>{_s['cum_t1']:+.1f}%</b>")
        else:
            _shadow_line = f"实盘影子: 已记录 <b>{_s.get('n_picks', 0)}</b> 笔，待次日收盘结算"
        _attr_src = (attr_info or {}).get("source", "")
        _attr_rng = (attr_info or {}).get("date_range", "")
        shadow_html = f"""
    <div class="section">
      <div class="section-title">📈 影子组合 · 策略归因 <span style="font-weight:400;color:#94a3b8">(复盘闭环)</span></div>
      <div style="font-size:.85rem;color:#475569;margin-bottom:10px">{_shadow_line}</div>
      <div style="display:flex;gap:24px;flex-wrap:wrap">
        <div style="flex:1;min-width:260px">
          <div style="font-size:.82rem;font-weight:600;color:#64748b;margin-bottom:6px">按引擎（回测）</div>
          <table class="data-table"><thead><tr><th>引擎</th><th>笔数</th><th>T+1胜率</th><th>D+5胜率</th><th>D+5均</th></tr></thead>
          <tbody>{_eng_rows}</tbody></table>
        </div>
        <div style="flex:1;min-width:220px">
          <div style="font-size:.82rem;font-weight:600;color:#64748b;margin-bottom:6px">按 Regime（回测）</div>
          <table class="data-table"><thead><tr><th>环境</th><th>笔数</th><th>T+1胜率</th><th>D+5均</th></tr></thead>
          <tbody>{_reg_rows}</tbody></table>
        </div>
      </div>
      <div style="margin-top:14px">
        <div style="font-size:.82rem;font-weight:600;color:#64748b;margin-bottom:6px">PWin 校准（模型说的概率 vs 实际胜率）</div>
        <table class="data-table"><thead><tr><th>PWin 档</th><th>笔数</th><th>模型均值</th><th>实T+1</th><th>实D+5</th></tr></thead>
        <tbody>{_cal_rows}</tbody></table>
      </div>
      <div style="margin-top:12px;padding:10px 12px;background:#f8fafc;border-radius:8px;font-size:.82rem;color:#475569">
        组合层面: {_port_line}<br>
        <span style="font-size:.72rem;color:#94a3b8">回测来源: {_attr_src} · {_attr_rng}</span>
      </div>
    </div>"""

    # ── 决策单区块（2026-09-25 Hermes）: 与 Telegram 推送同源 ──
    if card_info:
        _c = card_info
        _st = _c.get("std") or {}
        _exc = _st.get("excess") or {}
        _std_txt = (f"· 标准口径 {_st['state']}·{_st.get('ma_pos', '')}" if _st.get("state") else "")
        _exc_txt = (f"· 历史超额 {_exc['excess']}(相对胜率{_exc['rel_win']})" if _exc else "")
        _ind_b = "、".join(f"{x['ind']}({x['rank']:.0f}%)" for x in (_c.get("industries_buy") or [])[:6])
        _ind_a = "、".join(x["ind"] for x in (_c.get("industries_avoid") or [])[:6])
        _rows = []
        for _i, _s in enumerate((_c.get("stocks") or [])[:8], 1):
            _rows.append(
                f"<tr><td>{_i}</td><td>{_s['name']}</td><td>{_s['ts_code']}</td>"
                f"<td>{_s.get('mv', 0):.0f}亿</td><td>{_s.get('p_win', 0):.0f}%</td>"
                f"<td>{_s.get('expect', 0):+.1f}%</td></tr>")
        _neg_warn = ("<div style='color:#dc2626;font-size:.82rem;margin-top:6px'>"
                     "⚠️ 候选预期全为负 → 模型建议观望，不建仓</div>"
                     ) if any((s.get('expect') or 0) < 0 for s in (_c.get('stocks') or [])) else ""
        card_html = f"""
<div class="conclusion" style="border-left-color:#0ea5e9">
  <div class="conclusion-title">🎯 波段决策单（100-500亿域）</div>
  <div class="conclusion-sub">① 阶段 <b>{_c.get('regime', '—')}</b> {_std_txt}
     · 建议仓位 <b>{_c.get('position', '—')}</b> {_exc_txt}</div>
  <div style="font-size:.84rem;margin-top:8px">
    ② 行业 {('✅ ' + _ind_b) if _ind_b else ('○ ' + str(_c.get('arm_note', '不限行业')))}
    {('<br>&nbsp;&nbsp;&nbsp;🚫 回避: ' + _ind_a) if _ind_a else ''}</div>
  <table class="data-table" style="margin-top:10px"><thead><tr>
    <th>#</th><th>名称</th><th>代码</th><th>市值</th><th>P≧20%</th><th>预期</th></tr></thead>
    <tbody>{''.join(_rows) or '<tr><td colspan="6">无候选</td></tr>'}</tbody></table>
  {_neg_warn}
  <div style="font-size:.82rem;margin-top:10px;color:#475569">
    ④ 出场: 浮盈≥+10%后回撤8%卖 · 浮亏-10%止损（兜底）<br>
    ⑤ 纪律: 首仓≤1/3 · 同股间隔≥20日 · 单票≤25% · 行业≥4</div>
</div>"""
    else:
        card_html = ""

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
{risk_html}
{preplan_html}
{conclusion_html}
{card_html}
{wave_html}
{mainline_html}
{temp_html}
{emotion_html}
{decay_html}
{rules_html}
{strategy_html}
{shadow_html}

<div class="footer">
  V6 Gate Engine + Alpha Decay + 仓位配置规则<br>
  {n_folds_meta} folds · 2.5 年历史回测 · 不构成投资建议<br>
  模型: momentum / reversion / breakout @ {model_ver}
</div>

</body>
</html>"""

    return html


def generate_telegram_msg(report_date, next_date, temp_info, decay_info, danger_info=None, stale_days=0, risk_info=None, preplan_info=None, emotion_info=None, attr_info=None, shadow_info=None, wave_info=None, mainline_info=None, card_info=None):
    """生成 Telegram 推送消息"""
    rules, (lo, hi), _ = _load_rules_config()
    temp = temp_info["temp"]
    level = temp_info["level"]
    regime = temp_info["regime"]
    size, hist_t1, hist_d5, *_ = rules.get((regime, level), ("3只(半仓)", 0.45, 0.45))

    next_dt = datetime.strptime(next_date, "%Y%m%d")
    next_wd = ["周一", "周二", "周三", "周四", "周五"][next_dt.weekday()]

    report_dt = datetime.strptime(report_date, "%Y%m%d")
    report_wd = ["周一", "周二", "周三", "周四", "周五"][report_dt.weekday()]

    # ── 决策单整合 (2026-09-25 Hermes) ──
    # 去重择优: ①阶段(业内标准口径+实证超额) 替代旧「推断Regime」;
    #           ②行业(阶段自适应臂) 替代旧「🎯主线(选池)」;
    #           新增 ③个股候选/④出场/⑤纪律; 保留 V6 温度/风险/情绪/影子等独有段。
    _card = card_info if card_info is not None else compute_card(report_date)
    _std = (_card or {}).get("std") or {}
    _std_sfx = (f"（标准口径 {_std['state']}·{_std['ma_pos']}）" if _std.get("state") else "")
    _exc = _std.get("excess") or {}
    _exc_txt = (f" · 历史超额 <b>{_exc['excess']}</b>(相对胜率{_exc['rel_win']})" if _exc else "")
    _pos = (_card or {}).get("position") or size
    _act = (_card or {}).get("action_note") or ""

    lines = [
        f"📊 <b>V6日报 {report_date} {report_wd} → {next_date} {next_wd}</b>",
        "",
        f"🌡️ 温度 <b>{fmt_num(temp, 1)}/100</b> {level} · 阶段 <b>{(_card or {}).get('regime', regime)}</b>{_std_sfx}",
        f"① 建议仓位 <b>{_pos}</b> · V6档 {size} · 历史T+1胜率 {fmt_pct(hist_t1)}{_exc_txt}",
    ]
    if _act:
        lines.append(f"　本阶段动作: {_act}")

    # 警告前缀置顶：数据滞后 优先于 脆弱度 danger 优先于 大盘技术风险（非触发不显示，保持原格式）
    warn_prefix = []
    if stale_days > 0:
        warn_prefix.append(f"⚠️ <b>数据滞后 {stale_days} 个交易日</b> · 基于 {report_date}，谨慎交易")
    if danger_info:
        warn_prefix.append(f"⚠️ <b>市场脆弱度 DANGER</b> · {danger_info['alert_label']} · 建议 ≤{danger_info['pos_cap']}% 仓位（优先于 P3）")
    if risk_info and risk_info.get("level") in ("high", "extreme"):
        warn_prefix.append(f"⚠️ <b>大盘技术预警</b> · {risk_info['label']}风险（{risk_info['score']}/6）· {risk_info['advice']}")
    if warn_prefix:
        lines[0:0] = warn_prefix + [""]

    # 常驻风险分一行（无论档位都显示）
    if risk_info:
        lines.append(f"📊 大盘风险: {risk_info['label']}（{risk_info['score']}/6）· 触发: {('、'.join(risk_info['active_signals']) or '无')}")

    # 盘中预案（T-1 条件触发器, 2026-08-19 新增; 非预测"明天会跌", 而是盘中确认后的行动触发器）
    if preplan_info:
        _pplans = preplan_info[1]
        if _pplans:
            pp_txt = " ｜ ".join(
                f"{pp['name']}→{pp['action']}(命中{pp.get('hit_rate', 0)*100:.0f}%)"
                for pp in _pplans[:2]
            )
            lines.append(f"📈 盘中预案: {pp_txt}")

    # 主升浪雷达（波段层, 2026-09-25 新增）
    if wave_info and wave_info.get("states"):
        _w = " · ".join(f"{x['name']}{x['status']}" for x in wave_info["states"])
        lines.append(f"🌊 主升浪: {_w}")

    # ② 行业选择（阶段自适应臂, 2026-09-25 Hermes 整合）
    #    实测(backtest/report_signal_compare): 旧「主线」单独用超额仅+0.39%, 而
    #    全行业+2.46% / 震荡期「主线∩热度带」+2.88% / 过热期「主线」+2.77% → 按阶段自适应
    if _card:
        _ib = _card.get("industries_buy") or []
        _ia = _card.get("industries_avoid") or []
        _arm_note = _card.get("arm_note") or ""
        if _ib:
            _buy = "、".join("%s(%.0f%%)" % (x["ind"], x["rank"]) for x in _ib[:5])
            lines.append(f"② 行业: ✅ {_buy}" + (f"（{_arm_note}）" if _arm_note else ""))
        elif _arm_note:
            lines.append(f"② 行业: ○ {_arm_note}")
        if _ia:
            _avoid = "、".join(x["ind"] for x in _ia[:5])
            lines.append(f"　🚫 回避: {_avoid}")
        if _card.get("regime") == "过热" and mainline_info and mainline_info.get("ranking"):
            lines.append(f"　🎯 主线(过热期实测最优): {'、'.join(r['ind'] for r in mainline_info['ranking'][:5])}")

    # ③ 个股候选（选股层, 2026-09-25 Hermes 新增）
    if _card and _card.get("stocks"):
        _tn = "·震荡/过热压β" if _card.get("regime") in ("震荡/轮动", "过热") else ""
        lines.append(f"③ 个股候选(市值100-500亿{_tn}):")
        for _i, _s in enumerate(_card["stocks"][:8], 1):
            _mv = f" {_s['mv']:.0f}亿" if _s.get("mv") else ""
            _pw = f" P≧20%:{_s['p_win']:.0f}%" if "p_win" in _s else ""
            _ex = f" 预期{_s['expect']:+.1f}%" if "expect" in _s else ""
            lines.append(f"　{_i}. {_s['name']}({_s['ts_code']}){_mv}{_pw}{_ex}")

    # 情绪周期（市场结构层, 2026-09-25 新增）
    if emotion_info:
        e = emotion_info
        lines.append(f"🎭 情绪: {e['emo_score']}/100 {e['phase']} · 涨停{e['limit_up']}/跌停{e['limit_down']} "
                     f"· 最高{e['max_streak']}板 · 晋级{(e['promotion_rate'] or 0)*100:.0f}%")

    if decay_info:
        ranking = decay_info["ranking"]
        ind_stats = decay_info["ind_stats"]
        worst = list(ind_stats.head(3).index)
        best = list(ind_stats.tail(3).index)
        lines.append(f"📉 衰减板块: {' '.join(worst)}")
        lines.append(f"🟢 逆势板块: {' '.join(best)}")
        # PWin 摘要（2026-08-06: 用户要求 Telegram 显示 P(Win) 值；最上升与 HTML 榜同过滤: 分位≥75+上升）
        if ranking is not None and not ranking.empty:
            top_d = ranking.head(1).iloc[0]
            d_name = top_d.get("name", "") or top_d.name
            lines.append(f"🔻 最衰减 {d_name}: PWin {top_d['pwin_past']:.3f}→{top_d['pwin_today']:.3f} ({top_d['decay_pct']:+.0f}pp)")
            rise_pool = ranking[(ranking["rank_pct_today"] >= 75) & (ranking["decay_pct"] > 0)]
            if not rise_pool.empty:
                top_r = rise_pool.tail(1).iloc[0]
                r_name = top_r.get("name", "") or top_r.name
                lines.append(f"🚀 最上升 {r_name}: PWin {top_r['pwin_past']:.3f}→{top_r['pwin_today']:.3f} ({top_r['decay_pct']:+.0f}pp, 分位{top_r['rank_pct_today']:.0f}%)")

    # 警示
    if temp < lo:
        lines.append("🟢 温度偏低=市场不拥挤、胜率较高，可积极")
    elif temp > hi:
        lines.append("🔴 温度偏高=市场过热，未来5日偏弱，建议降仓")
    if regime == "bull" and level == "高温":
        bull_high = rules.get(("bull", "高温"), (None, None, 0.323))
        lines.append(f"⚠️ Bull+高温=危险信号，历史D+5仅{fmt_pct(bull_high[2])}")

    # 影子组合/归因（复盘层, 2026-09-25 新增）
    if shadow_info and shadow_info.get("n_settled_t1"):
        s = shadow_info
        lines.append(f"📈 影子实盘: {s['n_settled_t1']}笔 T+1胜率 {s['t1_wr']*100:.0f}% · 累计 {s['cum_t1']:+.1f}%")
    if attr_info:
        _eng = {x["engine"]: x for x in attr_info.get("by_engine", [])}
        if _eng:
            _best = max(_eng.values(), key=lambda x: x["d5_ret"])
            lines.append(f"🧪 引擎归因: {_best['engine']} D+5 {_best['d5_ret']:+.1f}%/胜率 {_best['d5_wr']*100:.0f}% 最优")

    url = f"{GITHUB_BASE}/v6_daily_{report_date}.html"
    # ④ 出场 / ⑤ 纪律（2026-09-25 Hermes 新增）
    lines.append("④ 出场: 浮盈≥+10%后从最高点回撤8%卖 · 浮亏-10%无条件止损(兜底)")
    lines.append("⑤ 纪律: 首仓≤1/3 · 同股间隔≥20日 · 单票≤25% · 行业≥4")
    lines.append("")
    lines.append(f'📄 <a href="{url}">完整报告</a>')

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def _redact(s: str) -> str:
    """2026-09-15 Claude: 抹掉输出中的凭据，防止写进日志/Telegram。

    git push 失败时 stderr 可能带 remote URL；本仓库 remote 内嵌 GitHub PAT。
    """
    s = re.sub(r"gh[pousr]_[A-Za-z0-9]{20,}", "***REDACTED***", s)
    # 不加 \b: Telegram URL 形如 .../bot<token>/sendMessage，'bot' 与数字间无词边界，
    # 加 \b 会漏匹配（已实测 bot8622871331:AAF... 逃逸）
    s = re.sub(r"\d{8,12}:AA[A-Za-z0-9_\-]{30,}", "***REDACTED***", s)
    return s


def _wait_for_pages_deploy(commit_sha: str, timeout_s: int = 600, poll_s: int = 15) -> bool:
    """等待 GitHub Pages Actions 部署完成（2026-08-06: 部署成功后再推 Telegram，保证链接可用）。

    Returns:
        True = 部署成功（或无可查询 token 时退化为立即返回）; False = 超时/失败
    """
    import json as _json
    import re as _re
    import subprocess
    import time as _time
    import urllib.request

    remote = subprocess.run(["git", "config", "remote.origin.url"],
                            capture_output=True, text=True).stdout.strip()
    m = _re.search(r"https://([^:]+):([^@]+)@github.com/(.+)", remote)
    if not m:
        return True  # 无 token，跳过等待（退化为立即推）
    token = m.group(2)
    repo = m.group(3).replace(".git", "")
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/actions/runs?head_sha={commit_sha}&per_page=1",
                headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
            runs = _json.load(urllib.request.urlopen(req, timeout=10)).get("workflow_runs", [])
            if runs:
                run = runs[0]
                if run["status"] == "completed":
                    return run.get("conclusion") == "success"
        except Exception:
            pass
        _time.sleep(poll_s)
    return False


def main():
    parser = argparse.ArgumentParser(description="V6 每日交易日报")
    parser.add_argument("--asof", type=str, default="", help="分析基准日 YYYYMMDD")
    parser.add_argument("--telegram", action="store_true", help="推送 Telegram")
    # 2026-09-15 Claude: 排障/预览用 — 推送通道不可用时也能看到日报正文
    parser.add_argument("--print-msg", action="store_true",
                        help="仅渲染 Telegram 消息到 stdout（不发送）")
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

    # ── Index Risk Gauge 大盘技术风险（常驻四色卡片，2026-08-19 新增）──
    risk_info = compute_index_risk(trading_date)
    if risk_info:
        print(f"📊 大盘风险: {risk_info['label']}（{risk_info['score']}/6）触发 {risk_info['active_signals'] or '无'}", file=sys.stderr)
    else:
        print("ℹ️ Index Risk: 数据不足/失败，不显示风险卡", file=sys.stderr)

    # ── 数据滞后（管道统一自愈失败时传入）──
    stale_days = getattr(args, "stale_days", 0) or 0
    if stale_days > 0:
        print(f"⚠️ 数据滞后 {stale_days} 个交易日，报告将带滞后横幅", file=sys.stderr)

    # ── 盘中预案（T-1 条件触发器; 配置 intraday_gap_rules.json, 缺失自动回退硬编码）──
    preplan_info = _load_intraday_rules()
    if preplan_info and preplan_info[1]:
        print(f"ℹ️ 盘中预案: {len(preplan_info[1])} 条已载入", file=sys.stderr)

    # ── A股情绪周期（市场结构层, 2026-09-25 Hermes 新增）──
    emotion_info = None
    try:
        emotion_info = compute_emotion(trading_date)
        if emotion_info:
            print(f"🎭 情绪周期: {emotion_info['emo_score']}/100 {emotion_info['phase']} "
                  f"(涨停{emotion_info['limit_up']}/跌停{emotion_info['limit_down']}/"
                  f"高度{emotion_info['max_streak']}/晋级{emotion_info['promotion_rate']})", file=sys.stderr)
        else:
            print("ℹ️ 情绪周期: 数据不足，跳过", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 情绪周期计算失败: {_redact(str(e))}", file=sys.stderr)

    # ── 主升浪雷达（波段层, 2026-09-25 Hermes 新增）──
    wave_info = None
    try:
        wave_info = wave_radar()
        if wave_info and wave_info.get("states"):
            _s = " · ".join(f"{x['name']}={x['status']}" for x in wave_info["states"])
            print(f"🌊 主升浪雷达: {_s}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 主升浪雷达失败: {_redact(str(e))}", file=sys.stderr)

    # ── 主线雷达（选池层, 2026-09-25 Hermes 新增）──
    mainline_info = None
    try:
        mainline_info = current_mainlines(trading_date, topk=10)
        if mainline_info and mainline_info.get("ranking"):
            _top = " · ".join(r["ind"] for r in mainline_info["ranking"][:4])
            print(f"🎯 主线雷达: {_top}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 主线雷达失败: {_redact(str(e))}", file=sys.stderr)

    # ── 影子组合：记录当日建议 + 策略归因（复盘层, 2026-09-25 Hermes 新增）──
    attr_info = shadow_info = None
    try:
        _rules_now, _th, _cmp = _load_rules_config()
        _size_label = _rules_now.get((temp_info.get("regime"), temp_info.get("level")), ("3只 (半仓)",))[0]
        if decay_info and decay_info.get("ranking") is not None:
            _rp = record_picks(trading_date, decay_info["ranking"], _size_label)
            print(f"📈 影子组合: 记录 {_rp.get('status')} · {_rp.get('n', _rp.get('n', 0))} 只", file=sys.stderr)
        attr_info = attribute_backtest()
        shadow_info = score_log()
        if attr_info:
            _ep = attr_info.get("portfolio", {})
            print(f"📈 策略归因: 引擎 {len(attr_info['by_engine'])} · 组合累计 "
                  f"{_ep.get('total_ret', 0)*100:+.1f}% · 最大回撤 {_ep.get('max_dd', 0)*100:.1f}%", file=sys.stderr)
        if shadow_info and shadow_info.get("n_settled_t1"):
            print(f"📈 影子实盘: 已结算 {shadow_info['n_settled_t1']} 笔 · T+1胜率 "
                  f"{shadow_info['t1_wr']*100:.1f}% · 累计 {shadow_info['cum_t1']:+.1f}%", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 影子组合计算失败: {_redact(str(e))}", file=sys.stderr)

    # ── 决策单（2026-09-25 Hermes）: 算一次, HTML 与 Telegram 共用, 避免重复开销 ──
    card_info = compute_card(trading_date)

    # ── 生成 HTML ──
    html = generate_html(trading_date, next_date, feat_date, price_date, model_ver,
                         temp_info, decay_info, temp_info["regime"], danger_info, stale_days, risk_info, preplan_info,
                         emotion_info, attr_info, shadow_info, wave_info, mainline_info, card_info)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    html_path = os.path.join(REPORTS_DIR, f"v6_daily_{trading_date}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML: {html_path}", file=sys.stderr)

    # ── Deploy（先部署，2026-08-06: 等部署成功后再推 Telegram，保证 TG 链接可用）──
    # 2026-09-15 Claude: 新增 push_failed 跟踪 + 出口非零（见文件尾 sys.exit）
    pushed_failed = False
    deployed_sha = None
    if args.deploy:
        import subprocess
        subprocess.run(["git", "add", f"reports/v6_daily/"], cwd=PROJECT_ROOT, capture_output=True)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
        if status.returncode != 0:
            subprocess.run(["git", "commit", "-m", f"📊 V6日报更新 {trading_date}"], cwd=PROJECT_ROOT, capture_output=True)
            result = subprocess.run(["git", "push"], cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                deployed_sha = subprocess.run(["git", "rev-parse", "HEAD"],
                                              cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
                print(f"✅ 已部署到 GitHub Pages: {GITHUB_BASE}/v6_daily_{trading_date}.html", file=sys.stderr)
            else:
                pushed_failed = True
                print(f"❌ git push 失败: {_redact(result.stderr)}", file=sys.stderr)
        else:
            print("ℹ️ 无变更，跳过 git push", file=sys.stderr)

    # ── Telegram（等 GitHub Pages 部署成功后再推）──
    if args.print_msg:
        _msg = generate_telegram_msg(trading_date, next_date, temp_info, decay_info,
                                     danger_info, stale_days, risk_info, preplan_info,
                                     emotion_info, attr_info, shadow_info, wave_info, mainline_info, card_info)
        print("\n" + "═" * 30 + " Telegram 消息预览 " + "═" * 30)
        print(_msg)
        print("═" * 76 + "\n")
    if args.telegram:
        if deployed_sha:
            deploy_ok = _wait_for_pages_deploy(deployed_sha, timeout_s=600)
            if deploy_ok:
                print("✅ GitHub Pages 部署成功，推送 Telegram", file=sys.stderr)
            else:
                print("⚠️ GitHub Pages 部署超时/失败，仍推送 Telegram（链接可能延迟）", file=sys.stderr)
        try:
            from notify.telegram_sender import send_single_message
            msg = generate_telegram_msg(trading_date, next_date, temp_info, decay_info, danger_info, stale_days, risk_info, preplan_info, emotion_info, attr_info, shadow_info, wave_info, mainline_info, card_info)
            ok = send_single_message(msg, parse_mode="HTML")
            if ok:
                print("✅ Telegram 已推送", file=sys.stderr)
            else:
                pushed_failed = True
                print("❌ Telegram 推送失败", file=sys.stderr)
        except Exception as e:
            pushed_failed = True
            print(f"❌ Telegram 异常: {_redact(str(e))}", file=sys.stderr)

    # 2026-09-15 Claude: 推送通道失败必须反映到退出码。
    #   背景: 09-05~09-14 Telegram 401 静默失败 11 天 — 旧版此处恒 exit 0，
    #   且上游管道只回显含 "✅" 的行，❌ 全被吞掉，cron/看门狗无从察觉。
    if pushed_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
