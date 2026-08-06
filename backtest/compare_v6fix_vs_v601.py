#!/usr/bin/env python3
"""对比 V6-01 基线 与 V6-Fix（HERMES-20260801-003）Walk-Forward 报告。

用法:
  python3 backtest/compare_v6fix_vs_v601.py \
      --base backtest/v6_walkforward_report_20260801.json \
      --fix  backtest/v6_walkforward_report_20260801_v6fix.json
"""
from __future__ import annotations

import argparse
import json

KEY_ORDER = [
    "n_folds", "date_range", "Strategy_Rank_IC", "rank_ic_positive_folds",
    "Actual_Win_Rate_Verified_T1", "Actual_Win_Rate_Verified_5D",
    "Actual_Win_Rate_Verified_Combined", "WinRate_ge55_folds",
    "Max_Drawdown_Protected", "Portfolio_Avg_Volatility_Top5",
    "Avg_T1_Risk_High_Count", "avg_gate_blocked_pct", "regime_distribution",
]

GOOD = {"higher": ["Strategy_Rank_IC", "rank_ic_positive_folds",
                   "Actual_Win_Rate_Verified_T1", "Actual_Win_Rate_Verified_5D",
                   "Actual_Win_Rate_Verified_Combined", "WinRate_ge55_folds"],
        "lower": ["Max_Drawdown_Protected", "Portfolio_Avg_Volatility_Top5",
                  "Avg_T1_Risk_High_Count", "avg_gate_blocked_pct"]}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:+.4f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--md", default="", help="输出 markdown 对比文件")
    args = ap.parse_args()

    base = load(args.base)["summary"]
    fix = load(args.fix)["summary"]

    lines = ["| 指标 | V6-01 基线 | V6-Fix | 变化 | 判定 |", "|---|---|---|---|---|"]
    for key in KEY_ORDER:
        if key not in base or key not in fix:
            continue
        b, f = base[key], fix[key]
        if isinstance(b, dict) and isinstance(f, dict):
            lines.append(f"| {key} | `{b}` | `{f}` | — | — |")
            continue
        # 非数值字段（str/bool/None）仅展示，不做差值
        if not isinstance(b, (int, float)) or not isinstance(f, (int, float)):
            lines.append(f"| {key} | `{b}` | `{f}` | — | — |")
            continue
        delta = f - b
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        verdict = ""
        if key in GOOD["higher"]:
            verdict = "✅ 改善" if delta > 0 else ("❌ 恶化" if delta < 0 else "＝持平")
        elif key in GOOD["lower"]:
            # Max_Drawdown_Protected 为负值：数值越小（更负）回撤越大，属恶化
            # 其他 lower 指标（波动/风险计数/拦截率）数值降低才是改善
            if key == "Max_Drawdown_Protected":
                verdict = "✅ 改善" if delta > 0 else ("❌ 恶化" if delta < 0 else "＝持平")
            else:
                verdict = "✅ 改善" if delta < 0 else ("❌ 恶化" if delta > 0 else "＝持平")
        lines.append(f"| {key} | {b} | {f} | {direction}{delta:.4f} | {verdict} |")

    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n对比表已保存: {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
