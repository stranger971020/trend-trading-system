#!/usr/bin/env python3
"""
v6_alpha_decay_ranking.py — P2: 全市场 Alpha Decay 衰减榜

用 model.feature_name_ 对齐特征列，对全市场沪深全量(约4999只)股票计算:
  本周 P(Win) vs N 周前 P(Win) → 衰减排名

输出:
  1. 全市场 TOP 50 衰减榜（跌幅最大 + 逆势上升）
  2. 行业级衰减聚合（哪些行业在加速衰减）
  3. 使用建议（交叉对比你的持仓）

依赖:
  - feature_matrix_v5.parquet (预计算特征)
  - data_storage/lgb_models/v6_{engine}_*.pkl (训练好的模型)

── Changelog ──
# 2026-08-06 Claude: 方案A — 衰减度量改「全市场PWin百分位变化(pp)」，消除低基数百分比放大
#               原 decay_pct=(今/前-1)% 使 0.008→0.079 显示 +867% 失真(绝对值仍低)
# 2026-08-06 Claude: 同模型对比(方案2) — 本周/N周前均用同一最新模型预测，隔离模型重训练校准漂移
#               原实现跨模型版本使全量池重训后衰减榜 99.9% 股票"衰减"(PWin校准差异~5x)
# 2026-08-02 Claude: 用 model.feature_name_ 解决特征对齐，3 engine < 3s
─────────────
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "data_storage", "lgb_models")
FEAT_PARQUET = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
INDUSTRY_CSV = os.path.join(PROJECT_ROOT, "data_storage", "stock_industry_mapping.csv")

ENGINES = ["momentum", "reversion", "breakout"]
LOOKBACK_WEEKS = 4  # 对比 N 周前的 P(Win)


def load_latest_models():
    """加载每个引擎的最新模型"""
    models = {}
    for eng in ENGINES:
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(f"v6_{eng}_")])
        if not files:
            print(f"⚠️ 无 {eng} 模型", file=sys.stderr)
            continue
        with open(os.path.join(MODEL_DIR, files[-1]), "rb") as f:
            models[eng] = pickle.load(f)
    return models


def load_model_at_date(eng, date_str):
    """加载最接近指定日期的模型（≤ date_str 的最新版本）"""
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(f"v6_{eng}_")])
    candidates = [f for f in files if f.split("_")[-1].replace(".pkl", "") <= date_str]
    if not candidates:
        return None
    with open(os.path.join(MODEL_DIR, candidates[-1]), "rb") as f:
        return pickle.load(f)


def predict_pwin(model, feat_df, ts_codes, date_str):
    """对指定日期+股票列表推断 P(Win)"""
    mf = model.feature_name_
    # 过滤到该日期的数据
    mask = (feat_df["trade_date"] == date_str) & (feat_df["ts_code"].isin(ts_codes))
    day_data = feat_df[mask].copy()
    if len(day_data) == 0:
        return None

    available = [c for c in mf if c in day_data.columns]
    missing = [c for c in mf if c not in day_data.columns]
    X = day_data[available].copy()
    for m in missing:
        X[m] = 0.0
    X = X[mf].fillna(0)

    proba = model.predict_proba(X)
    pwin = proba[:, 2] if proba.ndim == 2 and proba.shape[1] >= 3 else proba[:, 1]

    return pd.DataFrame(
        {"ts_code": day_data["ts_code"].values, "pwin": pwin}
    ).set_index("ts_code")


def compute_decay_ranking(feat_df, lookback_weeks=LOOKBACK_WEEKS):
    """
    全市场衰减排名:
    1. 取最新日期 → 本周 P(Win)
    2. 取 N 周前日期 → 历史 P(Win)
    3. decay = 本周 / 历史 (越小越惨)
    """
    # 确定日期
    all_dates = sorted(feat_df["trade_date"].unique())
    today_date = all_dates[-1]

    # 找 N 周前最接近的交易日
    today_dt = pd.to_datetime(today_date)
    target = (today_dt - pd.DateOffset(weeks=lookback_weeks)).strftime("%Y%m%d")
    # 找 ≤ target 的最新交易日
    past_dates = [d for d in all_dates if d <= target]
    if len(past_dates) < 2:
        print(f"⚠️ 回看期数据不足 (需要 {lookback_weeks} 周前)", file=sys.stderr)
        return None, None, None
    past_date = past_dates[-1]

    # 两天的股票交集
    today_stocks = set(feat_df[feat_df["trade_date"] == today_date]["ts_code"])
    past_stocks = set(feat_df[feat_df["trade_date"] == past_date]["ts_code"])
    common_stocks = sorted(today_stocks & past_stocks)

    print(f"本周: {today_date} ({len(today_stocks)} stocks)", file=sys.stderr)
    print(f"对比: {past_date} ({len(past_stocks)} stocks)", file=sys.stderr)
    print(f"交集: {len(common_stocks)} stocks", file=sys.stderr)

    if len(common_stocks) < 100:
        print("⚠️ 交集太少，可能日期选择有问题", file=sys.stderr)
        return None, None, None

    # 加载模型
    # 同模型对比(2026-08-06 方案2): 用同一最新模型预测"本周"和"N周前"，隔离特征驱动衰减。
    # 原实现 past 用 load_model_at_date 会跨模型版本(全量池重训后新旧PWin校准差异~5x)，
    # 使衰减榜被模型重训练污染(99.9%股票"衰减")。load_model_at_date 保留供外部调用。
    latest_models = load_latest_models()

    # 本周推断
    today_pwins = {}
    for eng in ENGINES:
        if eng not in latest_models:
            continue
        result = predict_pwin(latest_models[eng], feat_df, common_stocks, today_date)
        if result is not None:
            today_pwins[eng] = result

    # N 周前推断（同模型）
    past_pwins = {}
    for eng in ENGINES:
        if eng not in latest_models:
            continue
        result = predict_pwin(latest_models[eng], feat_df, common_stocks, past_date)
        if result is not None:
            past_pwins[eng] = result

    if not today_pwins or not past_pwins:
        print("❌ 推断失败", file=sys.stderr)
        return None, None, None

    # 合并: 取三引擎平均 P(Win)
    today_avg = pd.DataFrame(index=pd.Index(common_stocks, name="ts_code"))
    past_avg = pd.DataFrame(index=pd.Index(common_stocks, name="ts_code"))

    for eng in ENGINES:
        if eng in today_pwins and eng in past_pwins:
            today_avg[eng] = today_pwins[eng]["pwin"]
            past_avg[eng] = past_pwins[eng]["pwin"]

    today_avg["pwin_today"] = today_avg.mean(axis=1)
    past_avg["pwin_past"] = past_avg.mean(axis=1)

    # 衰减排名
    # 方案A(2026-08-06): 百分位排名变化替代百分比——消除低基数放大(0.008→0.079显示+867%失真)+免疫模型校准
    # decay = 全市场 PWin 百分位(今) − 百分位(前)，单位百分位点(pp)
    result = today_avg[["pwin_today"]].join(past_avg[["pwin_past"]])
    result["rank_pct_past"] = result["pwin_past"].rank(pct=True) * 100
    result["rank_pct_today"] = result["pwin_today"].rank(pct=True) * 100
    result["decay_pct"] = result["rank_pct_today"] - result["rank_pct_past"]
    result = result.sort_values("decay_pct")

    return result, today_date, past_date


def add_industry(result, industry_df):
    """添加行业标签"""
    if industry_df is None:
        result["l1_name"] = "未知"
        return result
    mapping = industry_df.set_index("ts_code")["l1_name"].to_dict()
    result["l1_name"] = result.index.map(mapping).fillna("未分类")
    return result


def build_report(result, industry_df, today_date, past_date):
    """生成 markdown 报告"""
    lines = []
    lines.append("# Alpha Decay 全市场衰减榜")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**对比周期**: {past_date} → {today_date} (间隔 {LOOKBACK_WEEKS} 周)")
    lines.append(f"**覆盖股票**: {len(result)} 只")
    lines.append(f"**计算方法**: 三引擎 (momentum/reversion/breakout) P(Win) 均值对比（同模型）")
    lines.append("")
    lines.append("> decay = 全市场 PWin 百分位(今) − 百分位(前)，单位百分位点(pp)。负值 = 衰减，越小越惨。")
    lines.append("")

    result = add_industry(result, industry_df)

    # ── TOP 30 跌幅榜 ──
    lines.append("## 🔴 衰减 TOP 30（PWin 全市场分位降幅最大）")
    lines.append("")
    lines.append("| # | 股票 | 行业 | 分位(前) | 分位(今) | 变化(pp) |")
    lines.append("|---|------|------|---------|---------|---------|")
    top30 = result.head(30)
    for i, (idx, row) in enumerate(top30.iterrows(), 1):
        lines.append(
            f"| {i} | {idx} | {row.get('l1_name', '-')} | {row['rank_pct_past']:.0f}% | "
            f"{row['rank_pct_today']:.0f}% | {row['decay_pct']:+.1f}pp |"
        )
    lines.append("")

    # ── TOP 30 上升榜 ──
    lines.append("## 🟢 逆势上升 TOP 30（PWin 全市场分位升幅最大）")
    lines.append("")
    lines.append("| # | 股票 | 行业 | 分位(前) | 分位(今) | 变化(pp) |")
    lines.append("|---|------|------|---------|---------|---------|")
    top30_up = result.tail(30)[::-1]
    for i, (idx, row) in enumerate(top30_up.iterrows(), 1):
        lines.append(
            f"| {i} | {idx} | {row.get('l1_name', '-')} | {row['rank_pct_past']:.0f}% | "
            f"{row['rank_pct_today']:.0f}% | {row['decay_pct']:+.1f}pp |"
        )
    lines.append("")

    # ── 行业衰减聚合 ──
    lines.append("## 行业级 Alpha Decay 热力")
    lines.append("")
    industry_stats = (
        result.groupby("l1_name")
        .agg(
            n_stocks=("decay_pct", "count"),
            avg_decay=("decay_pct", "mean"),
            median_decay=("decay_pct", "median"),
            pct_decaying=("decay_pct", lambda x: (x < 0).mean()),
        )
        .sort_values("avg_decay")
    )

    lines.append("| 行业 | 股票数 | 平均衰减(pp) | 中位衰减(pp) | 衰减占比 | 风险 |")
    lines.append("|------|--------|-------------|-------------|---------|------|")
    for idx, row in industry_stats.head(15).iterrows():
        risk = "🔴" if row["avg_decay"] < -10 else ("🟡" if row["avg_decay"] < 0 else "🟢")
        lines.append(
            f"| {idx} | {int(row['n_stocks'])} | {row['avg_decay']:+.1f}pp | "
            f"{row['median_decay']:+.1f}pp | {row['pct_decaying']:.0%} | {risk} |"
        )
    lines.append("")

    # ── 使用建议 ──
    lines.append("## 使用建议")
    lines.append("")
    lines.append("1. **交叉对比**: 将你的持仓 vs 衰减 TOP 30 名单比对。如果持仓出现在榜单上 → 检查是否需要提前减仓")
    lines.append("2. **自选观察**: 从 🔴 榜单挑 2-3 只观察是否超跌反弹；从 🟢 榜单挑 2-3 只观察趋势能否延续")
    lines.append("3. **行业轮动**: 关注「衰减占比」最高的行业——如果某个行业 >50% 的股票在衰减，可能是板块级别的风险")
    lines.append("4. **同模型对比**: 本期 PWin 对比使用同一最新模型预测本周与 N周前（2026-08-06 方案2），")
    lines.append("   隔离了模型重训练的校准漂移，衰减反映的是特征驱动的真实变化")
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(description="P2: 全市场 Alpha Decay 衰减榜")
    parser.add_argument("--save", type=str, default="", help="保存报告")
    parser.add_argument("--json-out", type=str, default="", help="导出全量衰减数据 JSON")
    parser.add_argument("--weeks", type=int, default=LOOKBACK_WEEKS, help="回看周数")
    args = parser.parse_args()

    # ── 加载特征矩阵 ──
    print("加载特征矩阵...", file=sys.stderr)
    feat_df = pd.read_parquet(FEAT_PARQUET)
    feat_df["trade_date"] = feat_df["trade_date"].astype(str)

    # ── 加载行业映射 ──
    industry_df = None
    if os.path.exists(INDUSTRY_CSV):
        industry_df = pd.read_csv(INDUSTRY_CSV)

    # ── 计算衰减排名 ──
    result, today_date, past_date = compute_decay_ranking(feat_df, args.weeks)

    if result is None:
        print("❌ 计算失败", file=sys.stderr)
        sys.exit(1)

    # ── 导出 JSON ──
    if args.json_out:
        out = result.reset_index()
        out.to_json(args.json_out, orient="records", indent=2, force_ascii=False)
        print(f"✅ 全量数据已导出: {args.json_out}", file=sys.stderr)

    # ── 生成报告 ──
    report = build_report(result, industry_df, today_date, past_date)
    output = "\n".join(report)
    print(output)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n✅ 报告已保存: {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
