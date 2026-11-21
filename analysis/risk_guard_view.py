"""风险守卫的「整合视图」——把保守层与 V6 进攻层合成一份可渲染对象

2026-11-21 Claude。设计原则：
  - risk_state.py 只算事实（状态/初减/止损/新鲜度），不判断 V6 的仓位。
  - 本模块只被 v6_daily_report.py 调用，负责解决「进攻 vs 防守」的呈现冲突：
      * 已有浮盈仓：以 frag_state 为动作权威（初减 + 8% 峰值回撤止损）
      * 新仓/总仓：以 V6 温度档/决策单为方向，硬顶取 danger / index_risk / 阶段仓位最严者
      * 数据不可用/滞后：才暂停新建仓建议；有效但 fragile/crash 不暂停，只显示上限
  - 输出会写入 V6 summary["risk_guard"]，看板 ⑦ 直接消费，避免二次判断。
"""
from __future__ import annotations

import re

KNOWN = ("normal", "warning", "fragile", "crash")
RISK_CAP = {"high": 0.50, "extreme": 0.30}
LABEL = {"normal": "正常", "warning": "警戒", "fragile": "脆弱", "crash": "崩跌"}


def _parse_card_ceiling(position: str | None) -> float | None:
    """'30-50%（试探建仓）' → 0.50；'0-30%（防守）' → 0.30；'30-60%' → 0.60。"""
    if not position:
        return None
    s = str(position).replace("％", "%")
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return min(float(m.group(2)) / 100.0, 1.0)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return min(float(m.group(1)) / 100.0, 1.0)
    return None


def _fresh_ok(guard: dict | None, trading_date: str,
              expected_date: str | None, stale_days: int,
              load_error: str | None) -> tuple[bool, list[str]]:
    """返回 (ok, warnings)。warnings 刻意压成一条，避免看板多句重复。"""
    if load_error:
        return False, [f"风险状态读取失败：{load_error} — 新建仓建议暂停"]
    if not isinstance(guard, dict):
        return False, ["风险状态未计算 — 新建仓建议暂停"]
    if guard.get("schema") != 1:
        return False, [f"风险状态 schema={guard.get('schema')}，期望 1 — 新建仓建议暂停"]
    if guard.get("state") not in KNOWN:
        return False, [f"风险状态非法：{guard.get('state')!r} — 新建仓建议暂停"]
    if not isinstance(guard.get("action"), dict):
        return False, ["风险状态缺 action — 新建仓建议暂停"]
    if str(guard.get("asof")) != str(trading_date):
        return False, [f"风险状态基准日与日报不一致（守卫 {guard.get('asof')} vs 日报 {trading_date}） — 新建仓建议暂停"]

    fr = guard.get("freshness") or {}
    fr_ok = bool(fr.get("ok", True))
    sd = int(stale_days or fr.get("stale_days") or 0)
    expected = expected_date or fr.get("expected")
    reasons: list[str] = []
    if sd:
        reasons.append(f"数据滞后 {sd} 个交易日")
    if expected and str(expected) != str(guard.get("asof")):
        reasons.append(f"期望 {expected} ≠ 基准 {guard.get('asof')}")
    if not fr_ok and not reasons:
        reasons.append("守卫自报数据滞后")
    if reasons:
        return False, ["；".join(reasons) + " — 新建仓建议暂停"]
    return True, []


def resolve(guard: dict | None, *, trading_date: str,
            expected_date: str | None = None, stale_days: int = 0,
            danger_info: dict | None = None, risk_info: dict | None = None,
            card_info: dict | None = None, v6_size: str = "—",
            load_error: str | None = None) -> dict:
    ok, warnings = _fresh_ok(guard, trading_date, expected_date, stale_days, load_error)
    g = guard if isinstance(guard, dict) else {}
    state = g.get("state") if g.get("state") in KNOWN else None
    label = LABEL.get(state, "—")
    act = g.get("action") or {}
    trim = float(act.get("initial_trim") or 0)
    already = bool(act.get("initial_trim_already_active"))
    incr = float(act.get("incremental_trim") or 0)
    prev_state = act.get("previous_state")

    # ── 防守层：已有浮盈仓 ──
    if not ok:
        existing = {
            "authority": "legacy", "type": "legacy", "trim": 0.0,
            "trail_stop": 0.08, "trim_already_active": False,
            "text": "风险状态不可用，沿用原纪律：浮盈≥+10%后回撤8%卖 · 浮亏-10%止损",
        }
    elif state in ("fragile", "crash"):
        text = act.get("trim_text") or f"初减 {trim:.0%}；剩余仓位自峰值回撤 8% 即减/退"
        existing = {
            "authority": "frag_state", "type": "guard", "trim": trim,
            "trail_stop": float(act.get("trail_stop") or 0.08),
            "trim_already_active": already, "incremental_trim": incr,
            "previous_state": prev_state, "text": text,
        }
    else:
        existing = {
            "authority": "legacy", "type": "legacy", "trim": 0.0,
            "trail_stop": 0.08, "trim_already_active": False,
            "text": "不初减；持仓沿用「浮盈≥+10% 后回撤 8% 卖」",
        }

    # ── 进攻层：V6 新仓/总仓 ──
    candidates: list[tuple[float, str]] = []
    if danger_info:
        try:
            cap = float(danger_info.get("pos_cap")) / 100.0
            if 0 < cap <= 1:
                candidates.append((cap, "脆弱度 danger"))
        except (TypeError, ValueError):
            pass
    if risk_info and risk_info.get("level") in RISK_CAP:
        candidates.append((RISK_CAP[risk_info["level"]], f"指数风险 {risk_info.get('label')}"))
    cap_card = _parse_card_ceiling((card_info or {}).get("position"))
    if cap_card is not None:
        candidates.append((cap_card, "阶段仓位"))

    divergence = bool(
        risk_info and risk_info.get("level") in ("high", "extreme")
        and state in ("normal", "warning"))

    if not ok:
        new_entry = {
            "authority": "v6_temperature+card", "status": "paused",
            "ceiling": None, "ceiling_source": None,
            "ceiling_text": "暂停（风险状态不可用/滞后）",
            "divergence": None,
            "text": "风险状态未计算/数据滞后 — 新建仓建议暂停",
        }
    elif candidates:
        ceiling, source = min(candidates, key=lambda x: (x[0], {"脆弱度 danger": 0, "指数风险": 1, "阶段仓位": 2}.get(x[1], 9)))
        new_entry = {
            "authority": "v6_temperature+card", "status": "cap",
            "ceiling": ceiling, "ceiling_source": source,
            "ceiling_text": f"≤{ceiling:.0%}（{source}）",
            "divergence": (
                f"技术面分歧：指数风险 {risk_info.get('label')}，但脆弱度未确认 → 不额外减仓，仅收紧新仓上限"
                if divergence else None),
            "text": f"按 V6 温度档/决策单执行；新仓/总仓上限 {ceiling:.0%}（{source}）",
        }
    else:
        new_entry = {
            "authority": "v6_temperature+card", "status": "normal",
            "ceiling": None, "ceiling_source": None,
            "ceiling_text": "无额外上限（按 V6 温度档）",
            "divergence": None,
            "text": f"按 V6 温度档执行（{v6_size}），无额外上限",
        }

    conflicts: list[str] = []
    if new_entry.get("divergence"):
        conflicts.append(new_entry["divergence"])
    if new_entry.get("ceiling_source") in ("脆弱度 danger", "指数风险"):
        conflicts.append(f"新仓上限严于阶段仓位：{new_entry['ceiling_source']}")
    if ok and state in ("fragile", "crash"):
        conflicts.append("出场规则切换：脆弱/崩跌时以峰值回撤 8% 替代 +10% 激活")
    if ok and already:
        conflicts.append("前日已处脆弱/崩跌；初减按已执行处理，不重复减仓（若未执行则补执行）")

    return {
        "ok": ok,
        "schema": 1,
        "asof": g.get("asof"),
        "expected_date": g.get("expected_date") or expected_date,
        "stale_days": int(stale_days or (g.get("freshness") or {}).get("stale_days") or 0),
        "freshness": g.get("freshness") or {},
        "state": state,
        "state_label": label,
        "score": g.get("score"),
        "components": g.get("components") or {},
        "metrics": g.get("metrics") or {},
        "history": g.get("history") or [],
        "validation": g.get("validation") or {},
        "action": act,
        "labels": {"offense": "进攻/新仓", "defense": "防守/已有浮盈"},
        "existing": existing,
        "new_entry": new_entry,
        "conflicts": conflicts,
        "warnings": warnings,
    }
