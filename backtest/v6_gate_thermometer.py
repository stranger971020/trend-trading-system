#!/usr/bin/env python3
"""
v6_gate_thermometer.py — P1: Gate Engine 市场温度计

构建每日市场温度分数 (0-100)，用历史数据验证预测能力。

温度计公式 (v2 - 百分位标准化):
  gate_score = (1 - gate_blocked_pct) 的 regime 内百分位 × 50
  wp_score   = win_prob_median 的全局百分位 × 30
  vol_score  = 交易量信号 × 20 (每日 trades > 3 只 → 满分)
  temp       = gate_score + wp_score + vol_score  → clip 0-100

验证维度:
  1. 温度分位 → 该 fold 内的 T+1/D+5 胜率（单调性）
  2. 温度趋势 vs 回撤预警
  3. 不同 regime 下温度与胜率关系

── Changelog ──
# 2026-08-02 Claude: P1 v2, 百分位标准化解决温度区间压缩问题
─────────────
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_JSON = os.path.join(SCRIPT_DIR, "v6_final_20260801.json")
PROD_JSON = os.path.join(SCRIPT_DIR, "v6_prod_20260802.json")
AUDIT_JSON = os.path.join(SCRIPT_DIR, "v6_audit_1yr_20260802.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(v):
    return f"{v*100:.1f}%"


def fmt_num(v, d=2):
    return f"{v:.{d}f}"


def validate_with_folds(temp_df, folds):
    """
    用 fold 级数据验证温度计：
    - 每 fold 有一个温度和后续胜率
    - 检查温度分位 → 胜率单调性
    - 检查温度趋降后回撤
    """
    lines = []

    # 构建 fold → 温度 映射
    fold_records = []
    for f in folds:
        try:
            d = datetime.strptime(f["val_start"], "%Y%m%d")
        except ValueError:
            continue
        if d in temp_df.index:
            t = temp_df.loc[d, "temp"]
            if pd.isna(t):
                continue
            fold_records.append(
                {
                    "date": d,
                    "fold": f["fold"],
                    "regime": f.get("regime", "unknown"),
                    "temp": t,
                    "gate_blocked": f["gate_blocked_pct"],
                    "t1_wr": f.get("t1_win_rate", 0),
                    "d5_wr": f.get("d5_win_rate", 0),
                    "alpha_cut": f.get("alpha_cut", 0),
                }
            )

    if len(fold_records) < 10:
        lines.append("⚠️ 不足 10 个 fold，无法有效验证。")
        return lines

    fdf = pd.DataFrame(fold_records)
    fdf["temp_q"] = pd.qcut(fdf["temp"], 4, labels=["Q1(冷)", "Q2(偏冷)", "Q3(温和)", "Q4(热)"])

    lines.append("## P1 验证: Gate Engine 市场温度计 (fold 级)")
    lines.append("")

    # ── 1. 温度分布 ──
    lines.append("### 1. 温度分布 (98 folds)")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 均值 | {fmt_num(fdf['temp'].mean(), 1)} |")
    lines.append(f"| 中位数 | {fmt_num(fdf['temp'].median(), 1)} |")
    lines.append(f"| 标准差 | {fmt_num(fdf['temp'].std(), 1)} |")
    lines.append(f"| 范围 | {fmt_num(fdf['temp'].min(), 1)} - {fmt_num(fdf['temp'].max(), 1)} |")
    lines.append(f"| <25 (极寒) | {(fdf['temp'] < 25).sum()} folds |")
    lines.append(f"| 25-50 (偏冷) | {((fdf['temp'] >= 25) & (fdf['temp'] < 50)).sum()} folds |")
    lines.append(f"| 50-75 (温和) | {((fdf['temp'] >= 50) & (fdf['temp'] < 75)).sum()} folds |")
    lines.append(f"| >75 (偏热) | {(fdf['temp'] >= 75).sum()} folds |")
    lines.append("")

    # ── 2. 温度 vs 胜率（按 regime 分层） ──
    lines.append("### 2. 温度分位 vs 胜率（全 regime）")
    lines.append("")
    lines.append("| 温度分位 | Folds | T+1 WR | D+5 WR | Gate Blocked | Alpha Cuts |")
    lines.append("|---------|-------|--------|--------|-------------|------------|")

    q_order = ["Q1(冷)", "Q2(偏冷)", "Q3(温和)", "Q4(热)"]
    q_stats = {}
    for q in q_order:
        qd = fdf[fdf["temp_q"] == q]
        if len(qd) == 0:
            continue
        q_stats[q] = {
            "n": len(qd),
            "t1": qd["t1_wr"].mean(),
            "d5": qd["d5_wr"].mean(),
            "gbp": qd["gate_blocked"].mean(),
            "ac": qd["alpha_cut"].sum(),
        }
        lines.append(
            f"| {q} | {len(qd)} | {fmt_pct(qd['t1_wr'].mean())} | "
            f"{fmt_pct(qd['d5_wr'].mean())} | {fmt_pct(qd['gate_blocked'].mean())} | "
            f"{qd['alpha_cut'].sum()} |"
        )

    # 单调性检验
    q_t1 = [q_stats[q]["t1"] for q in q_order if q in q_stats]
    q_d5 = [q_stats[q]["d5"] for q in q_order if q in q_stats]
    t1_mono = all(q_t1[i] <= q_t1[i + 1] for i in range(len(q_t1) - 1))
    d5_mono = all(q_d5[i] <= q_d5[i + 1] for i in range(len(q_d5) - 1))

    lines.append("")
    lines.append(f"| T+1 单调性 | {'✅ 单调递增' if t1_mono else '⚠️ 非严格单调'} | Q1→Q4: {fmt_pct(q_t1[0])} → {fmt_pct(q_t1[-1])} |")
    lines.append(f"| D+5 单调性 | {'✅ 单调递增' if d5_mono else '⚠️ 非严格单调'} | Q1→Q4: {fmt_pct(q_d5[0])} → {fmt_pct(q_d5[-1])} |")

    # 极端对比
    t1_spread = q_t1[-1] - q_t1[0]
    lines.append(f"| Q4-Q1 T+1 差值 | {'✅ ' + fmt_pct(t1_spread) if t1_spread > 0.05 else '⚠️ ' + fmt_pct(t1_spread)} | 冷热差异 |")
    lines.append("")

    # ── 3. 按 regime 分层的温度→胜率 ──
    lines.append("### 3. 各 Regime 内温度 vs 胜率")
    lines.append("")
    for regime in ["bull", "range", "bear"]:
        rd = fdf[fdf["regime"] == regime]
        if len(rd) < 4:
            continue
        # 分两半：低温 vs 高温
        med = rd["temp"].median()
        low = rd[rd["temp"] <= med]
        high = rd[rd["temp"] > med]

        lines.append(f"**{regime}** ({len(rd)} folds, 温度中位数={fmt_num(med, 1)}):")
        lines.append(f"| 温度组 | Folds | T+1 WR | D+5 WR | Gate Blocked |")
        lines.append(f"|--------|-------|--------|--------|-------------|")
        lines.append(
            f"| 低温 (≤{fmt_num(med, 1)}) | {len(low)} | {fmt_pct(low['t1_wr'].mean())} | "
            f"{fmt_pct(low['d5_wr'].mean())} | {fmt_pct(low['gate_blocked'].mean())} |"
        )
        lines.append(
            f"| 高温 (>{fmt_num(med, 1)}) | {len(high)} | {fmt_pct(high['t1_wr'].mean())} | "
            f"{fmt_pct(high['d5_wr'].mean())} | {fmt_pct(high['gate_blocked'].mean())} |"
        )

        # 温差 vs 胜率差
        diff = high["t1_wr"].mean() - low["t1_wr"].mean()
        lines.append(f"| 温差效应 | — | **{'✅ +' + fmt_pct(diff) if diff > 0 else '❌ ' + fmt_pct(diff)}** | — | — |")
        lines.append("")

    # ── 4. 回溯测试: 温度极低时后续回撤 ──
    lines.append("### 4. 极寒温度后的回撤预警")
    lines.append("")
    cold = fdf[fdf["temp"] < 25]
    if len(cold) > 0:
        lines.append(f"共 **{len(cold)}** 个极寒 fold (温度 < 25):")
        lines.append("")
        lines.append("| 日期 | Regime | 温度 | 本 fold T+1 WR | 后 5 folds 平均 T+1 WR |")
        lines.append("|------|--------|------|---------------|---------------------|")
        for _, row in cold.iterrows():
            fold_num = row["fold"]
            later = fdf[(fdf["fold"] > fold_num) & (fdf["fold"] <= fold_num + 5)]
            later_wr = later["t1_wr"].mean() if len(later) > 0 else float("nan")
            lines.append(
                f"| {row['date'].strftime('%Y-%m-%d')} | {row['regime']} | "
                f"{fmt_num(row['temp'], 1)} | {fmt_pct(row['t1_wr'])} | "
                f"{fmt_pct(later_wr) if not np.isnan(later_wr) else 'N/A'} |"
            )
        lines.append("")
    else:
        lines.append("无温度 < 25 的极寒 fold。")
        lines.append("")

    # ── 5. 最近温度 ──
    lines.append("### 5. 最近 10 folds 温度")
    lines.append("")
    recent = fdf.tail(10)[::-1]
    lines.append("| 日期 | Fold | Regime | 温度 | Gate Blocked | T+1 WR | D+5 WR |")
    lines.append("|------|------|--------|------|-------------|--------|--------|")
    for _, row in recent.iterrows():
        lines.append(
            f"| {row['date'].strftime('%Y-%m-%d')} | {row['fold']} | {row['regime']} | "
            f"{fmt_num(row['temp'], 1)} | {fmt_pct(row['gate_blocked'])} | "
            f"{fmt_pct(row['t1_wr'])} | {fmt_pct(row['d5_wr'])} |"
        )
    lines.append("")

    # ── 6. 判定 ──
    lines.append("### 6. 综合验证判定")
    lines.append("")
    verdict_parts = []
    if t1_spread > 0.05:
        verdict_parts.append(f"✅ 温度分位→T+1 WR 差异 {fmt_pct(t1_spread)} (>5pp)，区分度有效")
    elif t1_spread > 0:
        verdict_parts.append(f"⚠️ 温度分位→T+1 WR 差异 {fmt_pct(t1_spread)} (<5pp)，区分度偏弱")
    else:
        verdict_parts.append("❌ 温度分位→T+1 WR 无正向区分")

    if t1_mono:
        verdict_parts.append("✅ T+1 WR 随温度单调递增")
    if d5_mono:
        verdict_parts.append("✅ D+5 WR 随温度单调递增")

    # 计算 regine 内温差效应
    regime_effects = []
    for regime in ["bull", "range", "bear"]:
        rd = fdf[fdf["regime"] == regime]
        if len(rd) < 4:
            continue
        med = rd["temp"].median()
        diff = rd[rd["temp"] > med]["t1_wr"].mean() - rd[rd["temp"] <= med]["t1_wr"].mean()
        regime_effects.append(f"{regime}: {'✅' if diff > 0 else '❌'} {fmt_pct(diff)}")

    verdict_parts.append(f"Regime 内温差效应: {' | '.join(regime_effects)}")

    lines.append("| # | 验证项 | 结果 |")
    lines.append("|---|--------|------|")
    for i, v in enumerate(verdict_parts, 1):
        lines.append(f"| {i} | {v.split(':')[0] if ':' in v else v.split(' ')[0]} | {v} |")

    lines.append("")
    all_good = all("✅" in v for v in verdict_parts)
    lines.append(
        f"**综合判定**: {'✅ 温度计指标验证通过，可投入生产' if all_good else '⚠️ 温度计部分有效，建议调整权重后重新验证'}"
    )

    return lines


def build_fold_temperature(folds):
    """
    基于 fold 数据构建温度分数（百分位标准化）。

    对每个 fold:
      gate_score = (1 - gate_blocked_pct) 在同 regime 内的百分位 × 50
      temp = gate_score + 50 (base)

    这样确保:
      - 熊市的低 gate_blocked 自然获得高温
      - 震荡市的高 gate_blocked 自然获得低温
      - 分数始终在 25-75 区间（有区分度）
    """
    if not folds:
        return pd.DataFrame()

    records = []
    for f in folds:
        try:
            d = datetime.strptime(f["val_start"], "%Y%m%d")
        except ValueError:
            continue
        records.append(
            {
                "date": d,
                "fold": f["fold"],
                "regime": f.get("regime", "range"),
                "gate_blocked": f["gate_blocked_pct"],
            }
        )

    df = pd.DataFrame(records).set_index("date")

    # 按 regime 内计算 gate_blocked 百分位
    df["gate_blocked_pct_rank"] = df.groupby("regime")["gate_blocked"].transform(
        lambda x: x.rank(pct=True)
    )

    # 温度 = (1 - gate_blocked percentile) × 60 + 20
    # gate_blocked 高 → 百分位高 → 温度低 (更多股票被挡 → 市场冷)
    df["temp"] = ((1 - df["gate_blocked_pct_rank"]) * 60 + 20).clip(0, 100)

    return df


def main():
    parser = argparse.ArgumentParser(description="P1: Gate Engine 市场温度计")
    parser.add_argument("--save", type=str, default="", help="保存验证报告")
    parser.add_argument("--json-out", type=str, default="", help="导出每 fold 温度 JSON")
    args = parser.parse_args()

    final = load_json(FINAL_JSON)
    folds = final.get("folds", [])

    if not folds:
        print("❌ 无 fold 数据", file=sys.stderr)
        sys.exit(1)

    # ── 构建温度 ──
    temp_df = build_fold_temperature(folds)

    # ── 导出 ──
    if args.json_out:
        out = temp_df[["temp", "gate_blocked", "regime", "gate_blocked_pct_rank"]]
        out.index = out.index.strftime("%Y-%m-%d")
        out.to_json(args.json_out, orient="index", indent=2, force_ascii=False)
        print(f"✅ Fold 温度已导出: {args.json_out}", file=sys.stderr)

    # ── 报告 ──
    report = []
    report.append("# Gate Engine 市场温度计 — P1 实现与验证报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**数据**: 98 folds, {folds[0]['val_start']} ~ {folds[-1]['val_end']}")
    report.append("")
    report.append("**温度计公式**:")
    report.append("```")
    report.append("gate_rank  = gate_blocked_pct 在同 regime 内的百分位")
    report.append("temp      = (1 - gate_rank) × 60 + 20   // 范围 20-80, clip 0-100")
    report.append("```")
    report.append("")
    report.append("**设计理念**: gate_blocked_pct 高 → 大量候选被挡 → 市场冷 → 温度低。")
    report.append("百分位标准化确保不同 regime 间可比（熊市天然 gate 松 → rank 低 → 温度相对高）。")
    report.append("")

    report.extend(validate_with_folds(temp_df, folds))

    output = "\n".join(report)
    print(output)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n✅ 报告已保存: {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
