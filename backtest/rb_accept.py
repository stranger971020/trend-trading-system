"""验收脚本：检查回测指标、看板状态、报告三者一致

2026-11-21 Claude。运行后应输出「全部通过」。
用法：python3 backtest/rb_accept.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "backtest", "risk_budget_metrics.json")
REPORT = os.path.join(ROOT, "backtest", "risk_budget_report.md")
STATE = os.path.expanduser("~/projects/hermes-dashboard/risk_state.json")

fails: list[str] = []
checks: list[tuple[str, bool, str]] = []


def ck(name: str, cond: bool, detail: str = ""):
    checks.append((name, bool(cond), detail))
    if not cond:
        fails.append(f"{name} {detail}")


def near(a, b, tol=0.02):
    return a is not None and b is not None and abs(a - b) <= tol


def main():
    if not os.path.exists(MET):
        print(f"❌ 缺 {MET}，先跑 backtest/rb_export.py")
        return 1
    m = json.load(open(MET))
    fc = m["fragile_crash_15"]
    ss = m["second_shock_15"]
    pk = m["peak_12_18"]

    # 1) 核心尾部改善
    ck("脆弱/崩跌：持有 CVaR5 ≈ -39.5%", near(fc["hold"]["cvar5"], -0.3953, 0.01),
       f"实际 {fc['hold']['cvar5']}")
    ck("脆弱/崩跌：规则 CVaR5 ≈ -8.3%", near(fc["rule"]["cvar5"], -0.0828, 0.01),
       f"实际 {fc['rule']['cvar5']}")
    ck("脆弱/崩跌：P(跌>10%) 43% → 0.6%",
       fc["hold"]["p_lt_10"] > 0.40 and fc["rule"]["p_lt_10"] < 0.02)
    ck("规则均值优于持有", fc["rule"]["mean"] > fc["hold"]["mean"],
       f"{fc['rule']['mean']} vs {fc['hold']['mean']}")

    # 2) 第二次冲击
    ck("第二次冲击：持有均值 ≈ -6.7%", near(ss["hold"]["mean"], -0.0671, 0.01),
       f"实际 {ss['hold']['mean']}")
    ck("第二次冲击：规则均值 ≈ -1.0%", near(ss["rule"]["mean"], -0.0097, 0.01),
       f"实际 {ss['rule']['mean']}")
    ck("第二次冲击：规则 CVaR5 > 持有", ss["rule"]["cvar5"] > ss["hold"]["cvar5"],
       f"{ss['rule']['cvar5']} vs {ss['hold']['cvar5']}")

    # 3) 用户场景切片
    ck("峰值 +12~18%：最终浮亏概率 26% → 0%",
       pk["hold_p_final_neg"] > 0.20 and pk["rule_p_final_neg"] < 0.02,
       f"{pk['hold_p_final_neg']} → {pk['rule_p_final_neg']}")
    ck("峰值 +12~18%：回吐>10pp 42% → 3%",
       pk["hold_p_giveback_gt10"] > 0.35 and pk["rule_p_giveback_gt10"] < 0.08,
       f"{pk['hold_p_giveback_gt10']} → {pk['rule_p_giveback_gt10']}")

    # 4) 分年度稳定
    by = m.get("by_year") or []
    ck("分年度 5 年都有数据", len(by) >= 5, f"{len(by)} 年")
    ck("分年度规则 CVaR5 都在 -15%~-5%",
       all(-0.15 < r["rule_cvar5"] < -0.05 for r in by),
       str([r["rule_cvar5"] for r in by]))
    ck("分年度规则 CVaR5 均优于持有",
       all(r["rule_cvar5"] > r["hold_cvar5"] for r in by))

    # 5) 成本敏感性
    cs = m.get("cost_sensitivity") or []
    ck("成本 0.05~0.30% 结论不变",
       len(cs) == 3 and all(-0.12 < r["cvar5"] < -0.08 for r in cs),
       str([r["cvar5"] for r in cs]))

    # 6) 集中组合
    conc = m.get("concentration_fragile_crash") or []
    ck("集中组合 1/3/5 只均覆盖", {r["n"] for r in conc} == {1, 3, 5},
       str({r["n"] for r in conc}))
    for n in (1, 3, 5):
        h = next(r for r in conc if r["n"] == n and r["rule"] == "hold")
        r = next(r for r in conc if r["n"] == n and r["rule"] == "R7+trail8")
        ck(f"{n}只：规则 CVaR5 优于持有", r["cvar5"] > h["cvar5"],
           f"{r['cvar5']} vs {h['cvar5']}")
        ck(f"{n}只：规则 MDD 优于持有", r["mdd_mean"] > h["mdd_mean"],
           f"{r['mdd_mean']} vs {h['mdd_mean']}")

    # 7) 每日状态
    if os.path.exists(STATE):
        s = json.load(open(STATE))
        comp = s.get("components") or {}
        ck("风险分数可复现", int(sum(bool(v) for v in comp.values())) == int(s.get("score")),
           f"{sum(bool(v) for v in comp.values())} vs {s.get('score')}")
        score = int(s.get("score", 0))
        st = "normal" if score <= 1 else "warning" if score <= 3 else "fragile" if score <= 5 else "crash"
        ck("风险状态与分数一致", st == s.get("state"), f"{st} vs {s.get('state')}")
        trim = {"normal": 0.0, "warning": 0.0, "fragile": 0.25, "crash": 0.40}[st]
        mm = s.get("metrics") or {}
        if (mm.get("g60") or 0) < -0.05 and int(mm.get("shock10") or 0) >= 2:
            trim = max(trim, 0.50)
        trim = min(trim, 0.50)
        ck("初减建议可复现", abs(float(s["action"]["initial_trim"]) - trim) < 1e-9,
           f"{s['action']['initial_trim']} vs {trim}")
        ck("移动止损 8%", abs(float(s["action"]["trail_stop"]) - 0.08) < 1e-9)
        ck("回测锚点已注入", bool((s.get("validation") or {}).get("fragile_crash_15")))
    else:
        ck("每日状态文件存在", False, STATE)

    # 8) 报告关键结论
    if os.path.exists(REPORT):
        txt = open(REPORT, encoding="utf-8").read()
        for key in ("R7 + trail8%", "-39.5%", "-8.3%", "-6.56%", "第二次冲击", "8% 移动止损"):
            ck(f"报告含「{key}」", key in txt)
    else:
        ck("报告存在", False, REPORT)

    print(f"验收：{len(checks)} 项，失败 {len(fails)} 项")
    for name, ok, detail in checks:
        print(("  ✅ " if ok else "  ❌ ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if fails:
        print("\n❌ 未通过：" + "; ".join(fails))
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
