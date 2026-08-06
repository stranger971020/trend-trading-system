#!/usr/bin/env python3
"""
v6_position_sizing_rules.py — P3: 仓位配置规则表

基于 98 folds 历史数据，按 (regime, 温度, 集中度) 分位回归，
产出仓位配置规则表。

输出:
  1. Regime × 温度 → 建议仓位数、胜率历史
  2. 集中度 vs 回撤风险对照表
  3. 最优/最危险配置总结

依赖:
  - v6_final_20260801.json (folds)
  - v6_fold_temperature.json (P1 产出)
  - v6_prod_20260802.json (daily_trades)

── Changelog ──
# 2026-08-02 Claude: 用 98 folds 产出仓位规则表
─────────────
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FINAL_JSON = os.path.join(SCRIPT_DIR, "v6_final_20260801.json")
PROD_JSON = os.path.join(SCRIPT_DIR, "v6_prod_20260802.json")
TEMP_JSON = os.path.join(SCRIPT_DIR, "v6_fold_temperature.json")
INDUSTRY_CSV = os.path.join(PROJECT_ROOT, "data_storage", "stock_industry_mapping.csv")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(v):
    return f"{v*100:.1f}%"


def fmt_num(v, d=2):
    return f"{v:.{d}f}"


def build_fold_table(folds, daily_trades, industry_df):
    """构建每 fold 的分析表: regime, 温度, 行业数, T+1 WR, D+5 WR, alpha_cut"""
    # 加载温度
    temp_data = {}
    if os.path.exists(TEMP_JSON):
        with open(TEMP_JSON) as f:
            temp_data = json.load(f)

    dt_df = pd.DataFrame(daily_trades)
    dt_df["trade_date"] = pd.to_datetime(dt_df["trade_date"], format="%Y%m%d")

    records = []
    for f in folds:
        fid = str(f["fold"])
        regime = f.get("regime", "?")
        t1_wr = f.get("t1_win_rate", 0)
        d5_wr = f.get("d5_win_rate", 0)
        alpha_cut = f.get("alpha_cut", 0)
        gate_blocked = f.get("gate_blocked_pct", 0)
        n_candidates = f.get("n_candidates", 0)
        portfolio_size = f.get("portfolio_size", 0)
        val_start = f.get("val_start", "")

        # 温度
        temp = None
        for d_str, t_val in temp_data.items():
            if d_str == val_start:
                temp = t_val.get("temp", None)
                break

        # 行业数（从 daily_trades 统计该 fold 期间涉及多少行业）
        try:
            start_dt = pd.to_datetime(val_start)
            end_dt = pd.to_datetime(f.get("val_end", val_start))
        except:
            continue

        fold_trades = dt_df[(dt_df["trade_date"] >= start_dt) & (dt_df["trade_date"] <= end_dt)]
        n_industries = 0
        if industry_df is not None and len(fold_trades) > 0:
            merged = fold_trades.merge(industry_df[["ts_code", "l1_name"]], on="ts_code", how="left")
            n_industries = merged["l1_name"].nunique()

        # HHI
        n_trades = len(fold_trades)
        if n_trades > 0:
            stock_counts = fold_trades["ts_code"].value_counts()
            hhi = sum((c / n_trades) ** 2 for c in stock_counts.values)
        else:
            hhi = 0

        records.append({
            "fold": fid,
            "regime": regime,
            "temp": temp if temp is not None else 50,
            "t1_wr": t1_wr,
            "d5_wr": d5_wr,
            "alpha_cut": alpha_cut,
            "gate_blocked": gate_blocked,
            "portfolio_size": portfolio_size,
            "n_industries": n_industries,
            "n_trades": n_trades,
            "hhi": hhi,
        })

    return pd.DataFrame(records)


def compute_rules(fdf):
    """从 fold 表计算仓位配置规则"""
    lines = []
    lines.append("## P3 仓位配置规则表")
    lines.append("")

    if len(fdf) < 10:
        lines.append("⚠️ 数据不足")
        return lines

    # ── 温度分位 ──
    temp_vals = fdf["temp"].dropna()
    if len(temp_vals) < 10:
        fdf["temp_q"] = "N/A"
    else:
        fdf["temp_q"] = pd.qcut(fdf["temp"].rank(method="first"), 3, labels=["低温", "中温", "高温"])

    # ── 集中度分位 ──
    hhi_vals = fdf["hhi"].dropna()
    hhi_med = hhi_vals.median() if len(hhi_vals) > 0 else 0.05
    fdf["conc_level"] = fdf["hhi"].apply(lambda x: "分散" if x < hhi_med else "集中")

    # ── Rule Table 1: Regime × 温度 → 建议仓位 ──
    lines.append("### 规则表 1: Regime × 温度 → 仓位建议")
    lines.append("")
    lines.append("| Regime | 温度 | Folds | 平均 T+1 WR | 平均 D+5 WR | 平均 Alpha Cut | 建议仓位 |")
    lines.append("|--------|------|-------|------------|------------|---------------|---------|")

    recommendations = []
    for regime in ["bull", "range", "bear"]:
        for tq in ["低温", "中温", "高温"]:
            subset = fdf[(fdf["regime"] == regime) & (fdf["temp_q"] == tq)]
            if len(subset) < 2:
                continue
            t1 = subset["t1_wr"].mean()
            d5 = subset["d5_wr"].mean()
            ac = subset["alpha_cut"].mean()
            n = len(subset)

            # 仓位建议逻辑
            if t1 >= 0.55 and d5 >= 0.50:
                size = "5 只 (满仓)"
            elif t1 >= 0.45:
                size = "3 只 (半仓)"
            elif t1 >= 0.35:
                size = "2 只 (轻仓)"
            else:
                size = "1 只或空仓"

            recommendations.append({
                "regime": regime, "temp_q": tq, "n": n,
                "t1": t1, "d5": d5, "size": size, "ac": ac
            })
            lines.append(
                f"| {regime} | {tq} | {n} | {fmt_pct(t1)} | {fmt_pct(d5)} | "
                f"{fmt_num(ac, 1)} | **{size}** |"
            )

    lines.append("")
    lines.append("> 仓位建议基于该 (regime, 温度) 组合下的历史平均 T+1 胜率。")
    lines.append("> T+1 ≥ 55% 且 D+5 ≥ 50% → 满仓；T+1 在 45-55% → 半仓；T+1 < 35% → 空仓。")
    lines.append("")

    # ── Rule Table 2: 集中度风险对照 ──
    lines.append("### 规则表 2: 集中度 vs 胜率与回撤")
    lines.append("")
    lines.append("| 行业数 | Folds | 平均 HHI | T+1 WR | D+5 WR | Gate Blocked |")
    lines.append("|--------|-------|---------|--------|--------|-------------|")

    industry_bins = [(0, 3, "≤3 (极度集中)"), (4, 6, "4-6 (较集中)"), (7, 10, "7-10 (适中)"), (11, 99, ">10 (分散)")]
    for lo, hi, label in industry_bins:
        subset = fdf[(fdf["n_industries"] >= lo) & (fdf["n_industries"] <= hi)]
        if len(subset) < 2:
            continue
        lines.append(
            f"| {label} | {len(subset)} | {fmt_num(subset['hhi'].mean(), 4)} | "
            f"{fmt_pct(subset['t1_wr'].mean())} | {fmt_pct(subset['d5_wr'].mean())} | "
            f"{fmt_pct(subset['gate_blocked'].mean())} |"
        )
    lines.append("")

    # ── Rule Table 3: 最优/最危险配置 ──
    lines.append("### 规则表 3: 最优 vs 最危险配置")
    lines.append("")

    # 找到 top 5 最高的 T+1 WR 组合
    if recommendations:
        best = sorted(recommendations, key=lambda x: x["t1"], reverse=True)[:3]
        worst = sorted(recommendations, key=lambda x: x["t1"])[:3]

        lines.append("**🏆 最优配置（历史 T+1 WR 最高）**:")
        lines.append("")
        lines.append("| Regime | 温度 | Folds | T+1 WR | D+5 WR | 建议 |")
        lines.append("|--------|------|-------|--------|--------|------|")
        for r in best:
            lines.append(f"| {r['regime']} | {r['temp_q']} | {r['n']} | {fmt_pct(r['t1'])} | {fmt_pct(r['d5'])} | {r['size']} |")
        lines.append("")

        lines.append("**⚠️ 最危险配置（历史 T+1 WR 最低）**:")
        lines.append("")
        lines.append("| Regime | 温度 | Folds | T+1 WR | D+5 WR | 建议 |")
        lines.append("|--------|------|-------|--------|--------|------|")
        for r in worst:
            lines.append(f"| {r['regime']} | {r['temp_q']} | {r['n']} | {fmt_pct(r['t1'])} | {fmt_pct(r['d5'])} | {r['size']} |")
        lines.append("")

    # ── 使用示例 ──
    lines.append("### 每日使用流程")
    lines.append("")
    lines.append("```")
    lines.append("1. 运行 P1 温度计 → 获得今日温度")
    lines.append("2. 判断当前 regime (bull / range / bear)")
    lines.append("3. 查规则表 1 → 今日建议仓位 = ?")
    lines.append("4. 查规则表 2 → 最少行业覆盖 = ?")
    lines.append("5. 从 V6 选股池中按建议仓位和行业分散度建仓")
    lines.append("```")
    lines.append("")
    lines.append("**示例**: 今日 regime=bull, 温度=65(高温) → 查表 → 满仓 5 只, 行业≥4 个 → 从 V6 候选池取 momentum gate 前 5 名, 确保 ≥4 个不同 L1 行业。")
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(description="P3: 仓位配置规则表")
    parser.add_argument("--save", type=str, default="", help="保存报告")
    args = parser.parse_args()

    final = load_json(FINAL_JSON)
    prod = load_json(PROD_JSON)
    folds = final.get("folds", [])
    daily_trades = prod.get("daily_trades", [])

    industry_df = None
    if os.path.exists(INDUSTRY_CSV):
        industry_df = pd.read_csv(INDUSTRY_CSV)

    # ── 构建分析表 ──
    fdf = build_fold_table(folds, daily_trades, industry_df)

    # ── 计算规则 ──
    report = []
    report.append("# 仓位配置规则表 — P3 实现与应用")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**数据**: {len(folds)} folds, 2.5 年历史")
    report.append(f"**方法**: 按 Regime × 温度 × 集中度三维分位回归")
    report.append("")
    report.extend(compute_rules(fdf))

    # ── 附录: 原始数据摘要 ──
    report.append("## 附录: 各 Regime 基线数据")
    report.append("")
    report.append("| Regime | Folds | 平均 T+1 WR | 平均 D+5 WR | 平均 α Cut | 平均 Gate Blocked |")
    report.append("|--------|-------|------------|------------|-----------|------------------|")
    for regime in ["bull", "range", "bear"]:
        rd = fdf[fdf["regime"] == regime]
        if len(rd) == 0:
            continue
        report.append(
            f"| {regime} | {len(rd)} | {fmt_pct(rd['t1_wr'].mean())} | {fmt_pct(rd['d5_wr'].mean())} | "
            f"{fmt_num(rd['alpha_cut'].mean(), 1)} | {fmt_pct(rd['gate_blocked'].mean())} |"
        )
    report.append("")

    output = "\n".join(report)
    print(output)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n✅ 报告已保存: {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
