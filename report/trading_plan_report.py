"""
每日交易参考报告 — 手机竖屏优化版
- 引用块置顶核心结论
- Emoji 导航分区
- 压缩信号标签
- 个股卡片式排版（3行/只，每板块3只）
- Telegram HTML 格式输出
- 总计 ≤ 4096 字符（单段推送）
"""

import logging
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
import html as _html

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))
MAX_SECTORS_SHOW = 5
MAX_STOCKS_PER_SECTOR = 3
MAX_WATCH_SHOW = 2
MAX_WARN_SHOW = 4
MAX_WEAK_SHOW = 4


def _now_beijing():
    return datetime.now(BEIJING_TZ)


def _compress_signal(s: dict) -> str:
    """板块信号压缩为单行标签 (~35 字符)"""
    parts = []
    r = s.get("rsi", 50)
    if r > 80:
        parts.append(f"RSI{r:.0f}↓⚠️")
    elif r > 70:
        parts.append(f"RSI{r:.0f}↓")
    elif r > 60:
        parts.append(f"RSI{r:.0f}↑")
    elif r > 50:
        parts.append(f"RSI{r:.0f}↗")
    elif r > 40:
        parts.append(f"RSI{r:.0f}→")
    elif r > 30:
        parts.append(f"RSI{r:.0f}↓")
    else:
        parts.append(f"RSI{r:.0f}⚠️")

    if s.get("divergence"):
        parts.append("⚠️MACD背离")
    else:
        dif = s.get("macd_dif", 0)
        hist = s.get("macd_hist", 0)
        if dif > 0 and hist > 0:
            parts.append("MACD多")
        elif dif > 0 and hist < 0:
            parts.append("MACD柱缩")
        elif dif < 0 and hist < 0:
            parts.append("MACD空")
        elif dif < 0 and hist > 0:
            parts.append("MACD转多")

    vr = s.get("vol_ratio", 1)
    c5 = s.get("chg_5d", 0)
    if vr > 1.3 and c5 > 0:
        parts.append("量增✅")
    elif vr < 0.8 and c5 > 0:
        parts.append("量缩⚠️")

    if s.get("top_candle"):
        parts.append("🔴见顶")

    return " | ".join(parts)


def _stock_card(p: dict, parent: dict) -> list:
    """单只个股卡片，3行 HTML。用板块百分比映射到个股价格"""
    n = p.get("name", "?")
    c = p.get("code", "?")
    sc = p.get("score", "?")
    px = p.get("price", 0)
    mom = p.get("momentum_20d", 0)
    vr = p.get("vol_ratio", 0)

    lines = [
        f"  <b>{n} ({c})</b> | 总分 {sc}",
        f"  现价 {px} | 动量 {mom:+.1f}% | 量比 {vr:.2f}",
    ]

    # 风险提示行
    rw = p.get("risk_warnings", "")
    if rw:
        lines.append(f"  {rw}")

    if parent and px > 0:
        try:
            sp = parent.get("price", px)
            if sp > 0:
                el = round(px * parent["entry_low"] / sp, 2)
                eh = round(px * parent["entry_high"] / sp, 2)
                sl = round(px * parent["stop_loss"] / sp, 2)
                t1 = round(px * parent["target_1"] / sp, 2)
                sl_pct = (sl / px - 1) * 100
                t1_pct = (t1 / px - 1) * 100
                lines.append(
                    f"  入场 {el}~{eh} | <b>止损 {sl} ({sl_pct:+.1f}%)</b> | <b>目标 {t1} ({t1_pct:+.1f}%)</b>"
                )
        except Exception:
            pass

    return lines


def generate_daily_trading_report(
    l2_tech_result: dict,
    regime_result: dict = None,
    sentiment_result: dict = None,
    persistence_result: dict = None,
    stock_result: dict = None,
    key_levels_result: dict = None,
    risk_assessment: dict = None,
    news_overlay: dict = None,
) -> str:
    now = _now_beijing()
    date_str = now.strftime("%Y-%m-%d")
    lines = []
    zones = (l2_tech_result or {}).get("zones", {})
    chase = zones.get("chase", [])
    watch = zones.get("watch", [])
    top_warn = zones.get("top_warn", [])
    weak = zones.get("weak", [])

    # ── 仓位计算（市场脆弱度优先，追涨区数量次之） ──
    ccount = len(chase)
    risk_note = ""
    ra = risk_assessment or {}

    if ra.get("alert_level") in ("danger", "warning"):
        pos = f"≤{ra['pos_cap']}%"
        pos_desc = ra.get("pos_desc_override", "减仓观望")
        risk_note = ra.get("alert_label", "")
    else:
        # 正常市场：基于追涨区数量定仓位
        if ccount >= 8:
            pos, pos_desc = "40-50%", "追涨板块充足，中等仓位"
        elif ccount >= 3:
            pos, pos_desc = "25-35%", "有可操作板块，控制仓位"
        else:
            pos, pos_desc = "10-20%", "追涨板块少，轻仓试单"

    # 市场状态
    reg = ""
    if regime_result:
        rv2 = regime_result.get("regime_v2_label", "")
        rdesc = regime_result.get("regime_v2_desc", "")
        sp = regime_result.get("strategy_params", {})
        reg = f" | {rv2}"
        if rdesc:
            reg += f" {rdesc}"
        if sp.get("n"):
            reg += f" | 选{sp['n']}持{sp['hold']}"
        if sp.get("sl"):
            reg += f" 止{sp['sl']}%"
        if sp.get("tp"):
            reg += f" 盈{sp['tp']}%"

    # ── 数据日期（从 L2 技术指标结果中提取最新交易日） ──
    data_date = (l2_tech_result or {}).get("date", "")
    if not data_date:
        data_date = "N/A"
    data_date_str = f"📅 数据截至 {data_date[:4]}-{data_date[4:6]}-{data_date[6:]}" if data_date != "N/A" else "📅 数据日期: N/A"

    # ═══════════════════════ 标题 + 引用块 ═══════════════════════
    lines.append(f"📊 每日交易参考 — {date_str}")
    lines.append(f"生成: {now.strftime('%H:%M')} CST")
    lines.append(data_date_str)
    lines.append("")
    alert_level = ra.get("alert_level", "normal") if ra else "normal"
    if alert_level == "danger":
        risk_title = "⛔ 市场脆弱：高度谨慎"
    elif alert_level == "warning":
        risk_title = "⚠️ 市场偏弱：控制仓位"
    elif alert_level == "caution":
        risk_title = "📊 反弹乏力：不宜追高"
    else:
        risk_title = "✅ 大盘无系统性风险"
    bq = f"<blockquote><b>{risk_title} | 建议仓位 {pos}</b>\n{pos_desc}{reg}"
    if risk_note:
        bq += f"\n⚠️ {risk_note}"
    # 关键点位（R5 整合）
    if key_levels_result and key_levels_result.get("success"):
        lvls = key_levels_result["levels"]
        st = key_levels_result["status"]
        bq += f"\n📊 支撑 {lvls['strong_support']} | 阻力 {lvls['resistance_mid']} · {st['status']} {st['action_cn']}"
    bq += "</blockquote>"
    lines.append(bq)
    lines.append("")

    # 市场情绪仪表盘 + 策略建议
    if sentiment_result and sentiment_result.get("indicators"):
        inds = sentiment_result["indicators"]
        parts = []
        tips = []
        for key in ('leverage', 'turnover'):
            ind = inds.get(key)
            if ind and "N/A" not in str(ind.get('value', '')):
                pct = ind.get('pct', 50)
                parts.append(f"{ind['label']} {ind['value']} ({pct}%分位 {ind.get('signal', '')})")
                if key == 'leverage' and pct is not None:
                    if pct < 5:
                        tips.append("  融资出清极端低位 → 关注反弹机会，非进一步减仓")
                    elif pct < 20:
                        tips.append("  融资萎缩 → 配合warning确认但不足以上升到danger")
                    elif pct > 80:
                        tips.append("  杠杆偏高 → warning假信号概率高，可少减仓位")
        if parts:
            lines.append("📊 市场情绪 · " + " | ".join(parts))
            for t in tips:
                lines.append(t)
            lines.append("")

    # 舆情叠加
    if news_overlay:
        ns = news_overlay.get("news_sentiment", {})
        ov = news_overlay.get("overlay", {})
        if ns.get("total_news", 0) > 0:
            sent_emoji = {"calm":"✅","negative":"⚠️","panic":"🔴","mild":"📊"}
            emoji = sent_emoji.get(ns.get("sentiment_level",""), "📊")
            lines.append(f"{emoji} 舆情 · 今日{ns.get('total_news', 0)}条 负面{ns.get('negative_pct', 0)}%")
            sug = ov.get("suggestion", "")
            if sug:
                lines.append(f"  {sug}")
            lines.append("")

    if not l2_tech_result:
        lines.append("L2 数据不可用")
        return "\n".join(lines)

    # ═══════════════════════ 板块扫描 ═══════════════════════
    lines.append("━" * 20)
    lines.append("<b>📊 板块扫描</b> L2 二级行业")
    lines.append("━" * 20)
    lines.append("")

    # 🚀 追涨区（已按 score 降序）
    if chase:
        lines.append(f"🚀 <b>追涨区</b> {len(chase)}个")
        for s in chase[:MAX_SECTORS_SHOW]:
            sig = _compress_signal(s)
            sl_pct = (s["stop_loss"] / s["price"] - 1) * 100
            t1_pct = (s["target_1"] / s["price"] - 1) * 100
            lines.append(f"  <b>{s['name']}</b> ({s['price']}) {sig}")
            lines.append(
                f"  入场 {s['entry_low']}~{s['entry_high']} | <b>止损 {s['stop_loss']} ({sl_pct:+.1f}%)</b> | <b>目标 {s['target_1']} (+{s['expected_return_t1']}%)</b>"
            )
        if len(chase) > MAX_SECTORS_SHOW:
            lines.append(f"  …另 {len(chase)-MAX_SECTORS_SHOW} 个")
        lines.append("")

    # 👀 观察区
    if watch:
        w_sig = "、".join(
            f"{s['name']}({_compress_signal(s)})" for s in watch[:MAX_WATCH_SHOW]
        )
        lines.append(f"👀 <b>观察区</b> {len(watch)}个: {w_sig}")
        lines.append("")

    # 🔻 高位区（score 越低越危险，已排前）
    if top_warn:
        lines.append(f"🔻 <b>高位区</b> {len(top_warn)}个")
        for s in top_warn[:MAX_WARN_SHOW]:
            sig = _compress_signal(s)
            lines.append(f"  {s['name']}: {sig}")
        if len(top_warn) > MAX_WARN_SHOW:
            lines.append(f"  …另 {len(top_warn)-MAX_WARN_SHOW} 个")
        lines.append("")

    # ⛔ 弱势区
    if weak:
        wn = "、".join(s["name"] for s in weak[:MAX_WEAK_SHOW])
        lines.append(f"⛔ <b>弱势区</b> {len(weak)}个: {wn}…")
        lines.append("")

    # ═══════════════════════ 精选个股 ═══════════════════════
    lines.append("━" * 20)
    lines.append("<b>📈 精选个股</b> L2 追涨板块")
    lines.append("━" * 20)
    lines.append("")

    stock_picks = l2_tech_result.get("stock_picks", [])
    smap = {s["code"]: s for s in chase}

    if stock_picks:
        grouped = OrderedDict()
        for p in stock_picks:
            grouped.setdefault(p.get("sector", ""), []).append(p)

        sorder = sorted(
            grouped.keys(),
            key=lambda s: max(p["score"] for p in grouped[s]),
            reverse=True,
        )

        for sec in sorder:
            parent = smap.get(sec, {})
            sname = parent.get("name", sec)
            picks = grouped[sec]
            if not picks:
                continue
            ts = max(p["score"] for p in picks)
            lines.append(f"🚀 <b>{sname}</b> 强度 {ts}")
            for p in picks[:MAX_STOCKS_PER_SECTOR]:
                lines.extend(_stock_card(p, parent))
            lines.append("")
    else:
        lines.append("  当前无符合条件的个股")
        lines.append("")

    # ═══════════════════════ 顶部预警 ═══════════════════════
    lines.append("━" * 20)
    lines.append("<b>⚠️ 顶部预警</b>")
    lines.append("━" * 20)
    lines.append("")

    if top_warn:
        critical = [s for s in top_warn if s.get("divergence") or s.get("top_candle")]
        if critical:
            for s in critical[:4]:
                reason = "MACD顶背离" if s.get("divergence") else "🔴黄昏星/乌云"
                lines.append(f"  ⚠️ <b>{s['name']}</b> — {reason}")
            lines.append("")
        else:
            lines.append("  无明确顶部信号")
            lines.append("")
    else:
        lines.append("  无高位板块")
        lines.append("")

    # ═══════════════════════ 交易计划 ═══════════════════════
    lines.append("━" * 20)
    lines.append("<b>📋 今日交易计划</b>")
    lines.append("━" * 20)
    lines.append("")
    lines.append(f"  <b>仓位</b> {pos}　{pos_desc}")
    if risk_note:
        lines.append(f"  ⚠️ {risk_note}")
    lines.append("")
    lines.append("  ① 持仓缩量回MA20 → 持有")
    lines.append("  ② 放量跌破止损 → 离场")
    lines.append("  ③ warning解除+行业≥70%上涨 → 一步回满")
    lines.append("  ④ 市场脆弱期间 → 半仓防御")
    lines.append("")
    lines.append("  <b>每笔亏损 ≤ 总资金 1%</b>")
    lines.append("")
    # 回补信号
    if ra.get("re_entry_signal"):
        lines.append("  📈 回补信号: 行业≥70%上涨 → 可一步回满仓位")
        lines.append("")
    # 融资余额提醒（始终显示）
    lines.append("  ⚠️ 融资余额连续3日下降>1% → 去杠杆风险，提前减仓")
    lines.append("")
    lines.append("━" * 20)
    lines.append("⚠️ 技术指标参考，不构成投资建议")
    lines.append("📖 策略说明: https://stranger971020.github.io/trend-trading-system/reports/trading_report_guide.html")
    lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# HTML 报告生成（基于相同数据，输出完整可视化页面）
# ════════════════════════════════════════════════════════════

_FOLD_COUNTER = [0]  # mutable for closure


def _fold_id():
    _FOLD_COUNTER[0] += 1
    return f"fold-{_FOLD_COUNTER[0]}"


def _fold_btn(fid: str, label: str) -> str:
    """展开/折叠按钮 HTML"""
    return (
        f'<button class="fold-btn" onclick="toggleFold(\'{fid}\')" '
        f'data-label="{_html.escape(label)}">▸ 展开全部 ({label})</button>'
    )


def _fold_section(items: list, visible: int, render_item) -> str:
    """通用折叠区块：显示前 visible 个，其余可展开"""
    parts = []
    for i, item in enumerate(items):
        if i < visible:
            parts.append(render_item(item))
        else:
            break
    if len(items) > visible:
        hidden = items[visible:]
        fid = _fold_id()
        parts.append(f'<div id="{fid}" class="fold-body">')
        for h in hidden:
            parts.append(render_item(h))
        parts.append("</div>")
        parts.append(_fold_btn(fid, f"{len(hidden)}个"))
    return "\n".join(parts)


def _sig_html(s: dict) -> str:
    """板块信号的彩色 HTML 标签"""
    tags = []
    r = s.get("rsi", 50)
    if r > 80:
        tags.append(f'<span class="tag tag-red">RSI{r:.0f}↓⚠️</span>')
    elif r > 70:
        tags.append(f'<span class="tag tag-orange">RSI{r:.0f}↓</span>')
    elif r > 60:
        tags.append(f'<span class="tag tag-green">RSI{r:.0f}↑</span>')
    elif r > 50:
        tags.append(f'<span class="tag tag-green">RSI{r:.0f}↗</span>')
    elif r > 40:
        tags.append(f'<span class="tag tag-gray">RSI{r:.0f}→</span>')
    elif r > 30:
        tags.append(f'<span class="tag tag-gray">RSI{r:.0f}↓</span>')
    else:
        tags.append(f'<span class="tag tag-red">RSI{r:.0f}⚠️</span>')

    if s.get("divergence"):
        tags.append('<span class="tag tag-red">⚠️MACD背离</span>')
    else:
        dif = s.get("macd_dif", 0)
        hist = s.get("macd_hist", 0)
        if dif > 0 and hist > 0:
            tags.append('<span class="tag tag-green">MACD多</span>')
        elif dif > 0 and hist < 0:
            tags.append('<span class="tag tag-yellow">MACD柱缩</span>')
        elif dif < 0 and hist < 0:
            tags.append('<span class="tag tag-red">MACD空</span>')
        elif dif < 0 and hist > 0:
            tags.append('<span class="tag tag-yellow">MACD转多</span>')

    vr = s.get("vol_ratio", 1)
    c5 = s.get("chg_5d", 0)
    if vr > 1.3 and c5 > 0:
        tags.append('<span class="tag tag-green">量增✅</span>')
    elif vr < 0.8 and c5 > 0:
        tags.append('<span class="tag tag-yellow">量缩⚠️</span>')

    if s.get("top_candle"):
        tags.append('<span class="tag tag-red">🔴见顶</span>')

    return " ".join(tags)


def _zone_item_html(s: dict) -> str:
    """单个追涨区/高位区板块的 HTML"""
    sig = _sig_html(s)
    sl_pct = (s["stop_loss"] / s["price"] - 1) * 100
    t1_pct = (s["target_1"] / s["price"] - 1) * 100
    return (
        f'<div class="zone-item">'
        f'<div class="zone-name"><b>{_html.escape(s["name"])}</b> '
        f'<span class="price">{s["price"]}</span></div>'
        f'<div class="zone-sig">{sig}</div>'
        f'<div class="zone-pl">入场 {s["entry_low"]}~{s["entry_high"]} | '
        f'<b class="neg">止损 {s["stop_loss"]} ({sl_pct:+.1f}%)</b> | '
        f'<b class="pos">目标 {s["target_1"]} (+{s["expected_return_t1"]}%)</b></div>'
        f"</div>"
    )


def _warn_item_html(s: dict) -> str:
    """高位预警板块的 HTML"""
    sig = _sig_html(s)
    return (
        f'<div class="zone-item warn-item">'
        f'<b>{_html.escape(s["name"])}</b>: {sig}'
        f"</div>"
    )


def _stock_card_html(p: dict, parent: dict) -> str:
    """单只个股卡片 HTML"""
    n = p.get("name", "?")
    c = p.get("code", "?")
    sc = p.get("score", 0)
    px = p.get("price", 0)
    mom = p.get("momentum_20d", 0)
    vr = p.get("vol_ratio", 0)

    score_class = "score-high" if sc >= 8 else ("score-mid" if sc >= 6 else "score-low")

    lines = [
        f'<div class="stock-card">',
        f'<div class="stock-head">',
        f'<span class="stock-score {score_class}">{sc}</span>',
        f'<span class="stock-name">{_html.escape(n)} ({c})</span>',
        f"</div>",
        f'<div class="stock-metrics">',
        f'现价 {px} | 动量 {mom:+.1f}% | 量比 {vr:.2f}',
        f"</div>",
    ]

    # 风险提示行
    rw = p.get("risk_warnings", "")
    if rw:
        lines.append(f'<div class="stock-risk">{_html.escape(rw)}</div>')

    if parent and px > 0:
        try:
            sp = parent.get("price", px)
            if sp > 0:
                el = round(px * parent["entry_low"] / sp, 2)
                eh = round(px * parent["entry_high"] / sp, 2)
                sl = round(px * parent["stop_loss"] / sp, 2)
                t1 = round(px * parent["target_1"] / sp, 2)
                sl_pct = (sl / px - 1) * 100
                t1_pct = (t1 / px - 1) * 100
                lines.append(
                    f'<div class="stock-pl">入场 {el}~{eh} | '
                    f'<b class="neg">止损 {sl} ({sl_pct:+.1f}%)</b> | '
                    f'<b class="pos">目标 {t1} ({t1_pct:+.1f}%)</b></div>'
                )
        except Exception:
            pass

    lines.append("</div>")
    return "\n".join(lines)


def generate_daily_trading_html(
    l2_tech_result: dict,
    regime_result: dict = None,
    risk_assessment: dict = None,
    sentiment_result: dict = None,
    news_overlay: dict = None,
) -> str:
    """生成完整的可视化 HTML 报告（替代纯文本包装）"""
    now = datetime.now(timezone(timedelta(hours=8)))
    date_str = now.strftime("%Y-%m-%d")
    data_date = (l2_tech_result or {}).get("date", "")
    if not data_date:
        data_date = "N/A"
    data_date_str = f"数据截至 {data_date[:4]}-{data_date[4:6]}-{data_date[6:]}" if data_date != "N/A" else "数据日期: N/A"
    ra = risk_assessment or {}
    zones = (l2_tech_result or {}).get("zones", {})
    chase = zones.get("chase", [])
    watch = zones.get("watch", [])
    top_warn = zones.get("top_warn", [])
    weak = zones.get("weak", [])
    stock_picks = (l2_tech_result or {}).get("stock_picks", [])
    smap = {s["code"]: s for s in chase}

    # ── 仓位计算（市场脆弱度优先） ──
    ccount = len(chase)
    wcount = len(top_warn)
    risk_note = ""

    if ra.get("alert_level") in ("danger", "warning"):
        pos = f"≤{ra['pos_cap']}%"
        pos_desc = ra.get("pos_desc_override", "减仓观望")
        risk_note = ra.get("alert_label", "")
    else:
        if ccount >= 8:
            pos, pos_desc = "40-50%", "追涨板块充足，中等仓位"
        elif ccount >= 3:
            pos, pos_desc = "25-35%", "有可操作板块，控制仓位"
        else:
            pos, pos_desc = "10-20%", "追涨板块少，轻仓试单"

    alert_level = ra.get("alert_level", "normal") if ra else "normal"
    if alert_level == "danger":
        risk_title = "⛔ 市场脆弱：高度谨慎"
    elif alert_level == "warning":
        risk_title = "⚠️ 市场偏弱：控制仓位"
    elif alert_level == "caution":
        risk_title = "📊 反弹乏力：不宜追高"
    else:
        risk_title = "✅ 大盘无系统性风险"

    # ── 市场状态 ──
    reg = ""
    if regime_result:
        rv2 = regime_result.get("regime_v2_label", "")
        rdesc = regime_result.get("regime_v2_desc", "")
        sp = regime_result.get("strategy_params", {})
        reg = f" | {rv2} {rdesc}" if rdesc else f" | {rv2}"
        if sp.get("n"):
            reg += f" | 选{sp['n']}持{sp['hold']}"
        if sp.get("sl"):
            reg += f" 止{sp['sl']}%"
        if sp.get("tp"):
            reg += f" 盈{sp['tp']}%"

    # ═══════════════════════════ 构建页面 ═══════════════════════════
    body = []

    # ── 核心结论 ──
    body.append(
        f'<div class="conclusion">'
        f'<div class="conclusion-title">{risk_title} · 建议仓位 <b>{pos}</b></div>'
        f'<div class="conclusion-sub">{pos_desc}{_html.escape(reg)}</div>'
        + (f'<div class="conclusion-warn">{risk_note}</div>' if risk_note else "")
        + "</div>"
    )

    # ── 市场情绪仪表盘 ──
    if sentiment_result and sentiment_result.get("indicators"):
        inds = sentiment_result["indicators"]
        sent_parts = []
        for key in ('leverage', 'turnover'):
            ind = inds.get(key)
            if ind and "N/A" not in str(ind.get('value', '')):
                sent_parts.append(
                    f'<span style="color:#6366f1">{ind["label"]}</span> '
                    f'{ind["value"]} '
                    f'<span style="color:{('#dc2626' if ind.get('pct', 50) >= 80 else '#16a34a')}">'
                    f'({ind.get("pct", "")}%分位 {ind.get("signal", "")})</span>'
                )
        if sent_parts:
            body.append(f'<div class="section" style="padding:10px 20px;font-size:.82rem">')
            body.append(f'📊 市场情绪 · {" | ".join(sent_parts)}')
            # 策略建议
            for key in ('leverage', 'turnover'):
                ind = inds.get(key)
                if ind and key == 'leverage':
                    pct = ind.get('pct', 50)
                    if pct is not None:
                        if pct < 5:
                            body.append(f'<div style="color:#16a34a;margin-top:6px">✅ 融资出清极端低位 → 关注反弹机会，非进一步减仓</div>')
                        elif pct < 20:
                            body.append(f'<div style="color:#dc2626;margin-top:6px">⚠️ 融资萎缩 → 配合warning确认但不足以上升到danger</div>')
                        elif pct > 80:
                            body.append(f'<div style="color:#dc2626;margin-top:6px">⚠️ 杠杆偏高 → warning假信号概率高，可少减仓位</div>')
            body.append(f'</div>')

    # ── 舆情监控 ──
    if news_overlay:
        ns = news_overlay.get("news_sentiment", {})
        ov = news_overlay.get("overlay", {})
        if ns.get("total_news", 0) > 0:
            sent_emoji = {"calm":"✅","negative":"⚠️","panic":"🔴","mild":"📊"}
            body.append(f'<div class="section" style="padding:10px 20px;font-size:.82rem">')
            body.append(f'{sent_emoji.get(ns.get("sentiment_level",""), "📊")} 舆情 · 今日{ns.get("total_news", 0)}条 负面{ns.get("negative_pct", 0)}%')
            sug = ov.get("suggestion", "")
            if sug:
                color = "#dc2626" if ns.get("sentiment_level") in ("negative","panic") else "#6366f1"
                body.append(f'<div style="color:{color};margin-top:4px">{ov["suggestion"]}</div>')
            body.append(f'</div>')

    # ── 板块扫描 ──
    body.append('<div class="section"><div class="section-title">📊 板块扫描</div>')

    # 🚀 追涨区
    if chase:
        body.append('<div class="zone-card chase">')
        body.append(f'<div class="zone-header">🚀 追涨区 <span class="count">{len(chase)}个</span></div>')
        body.append(_fold_section(chase, 5, _zone_item_html))
        body.append("</div>")

    # 👀 观察区
    if watch:
        body.append('<div class="zone-card watch">')
        body.append(f'<div class="zone-header">👀 观察区 <span class="count">{len(watch)}个</span></div>')
        body.append(_fold_section(watch, 3, lambda s: f'<div class="w-item">{_html.escape(s["name"])}: {_sig_html(s)}</div>'))
        body.append("</div>")

    # 🔻 高位区
    if top_warn:
        body.append('<div class="zone-card warn">')
        body.append(f'<div class="zone-header">🔻 高位区 <span class="count">{len(top_warn)}个</span></div>')
        body.append(_fold_section(top_warn, 5, _warn_item_html))
        body.append("</div>")

    # ⛔ 弱势区
    if weak:
        weak_names = [s["name"] for s in weak]
        visible = weak_names[:6]
        body.append('<div class="zone-card weak">')
        body.append(f'<div class="zone-header">⛔ 弱势区 <span class="count">{len(weak)}个</span></div>')
        body.append(f'<div class="w-item">{"、".join(visible)}</div>')
        if len(weak) > 6:
            fid = _fold_id()
            body.append(f'<div id="{fid}" class="fold-body">')
            body.append(f'<div class="w-item">{"、".join(weak_names[6:])}</div>')
            body.append("</div>")
            body.append(_fold_btn(fid, f"{len(weak)-6}个"))
        body.append("</div>")

    body.append("</div>")  # end section

    # ── 精选个股 ──
    if stock_picks:
        body.append('<div class="section"><div class="section-title">📈 精选个股</div>')
        grouped = OrderedDict()
        for p in stock_picks:
            grouped.setdefault(p.get("sector", ""), []).append(p)
        sorder = sorted(grouped.keys(), key=lambda s: max(p["score"] for p in grouped[s]), reverse=True)

        for sec in sorder:
            parent = smap.get(sec, {})
            sname = parent.get("name", sec)
            picks = grouped[sec]
            if not picks:
                continue
            ts = max(p["score"] for p in picks)
            body.append(f'<div class="sector-block">')
            body.append(f'<div class="sector-title">🚀 {_html.escape(sname)} <span class="score-badge">强度 {ts}</span></div>')
            body.append(_fold_section(picks, 3, lambda p: _stock_card_html(p, parent)))
            body.append("</div>")
        body.append("</div>")

    # ── 顶部预警 ──
    if top_warn:
        body.append('<div class="section"><div class="section-title">⚠️ 顶部预警</div>')
        critical = [s for s in top_warn if s.get("divergence") or s.get("top_candle")]
        if critical:
            for s in critical[:4]:
                reason = "MACD顶背离" if s.get("divergence") else "🔴黄昏星/乌云"
                body.append(f'<div class="warn-item"><b>{_html.escape(s["name"])}</b> — {reason}</div>')
        else:
            body.append('<div class="warn-item">无明确顶部信号</div>')
        body.append("</div>")

    # ── 交易计划 ──
    body.append('<div class="section"><div class="section-title">📋 今日交易计划</div>')
    body.append(f'<div class="plan-item"><b>仓位</b> {pos}　{pos_desc}</div>')
    if risk_note:
        body.append(f'<div class="plan-item">{risk_note}</div>')
    body.append(
        '<div class="plan-list">'
        "① 持仓缩量回MA20 → 持有<br>"
        "② 放量跌破止损 → 离场<br>"
        "③ warning解除+行业≥70%上涨 → 一步回满<br>"
        "④ 市场脆弱期间 → 半仓防御"
        "</div>"
    )
    body.append('<div class="plan-item"><b>每笔亏损 ≤ 总资金 1%</b></div>')
    if ra.get("re_entry_signal"):
        body.append(
            f'<div class="plan-item" style="color:#16a34a;font-size:.82rem;margin-top:6px">'
            f'📈 回补信号: 行业≥70%上涨 → 可一步回满仓位'
            f'</div>'
        )
    body.append(
        f'<div class="plan-item" style="color:#dc2626;font-size:.82rem;margin-top:6px">'
        f'⚠️ 融资余额连续3日下降>1% → 去杠杆风险，提前减仓'
        f'</div>'
    )
    body.append("</div>")

    content = "\n".join(body)

    # ═══════════════════════════ 完整 HTML 页面 ═══════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>交易参考 {date_str}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Helvetica Neue',system-ui,sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.6;padding:16px;max-width:900px;margin:0 auto;-webkit-font-smoothing:antialiased}}

/* ── 头部 ── */
.header{{background:linear-gradient(135deg,#1e293b,#334155);color:#fff;padding:24px 28px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:1.3rem;font-weight:700;letter-spacing:-0.3px}}
.header .time{{font-size:.82rem;color:#94a3b8;margin-top:4px}}

/* ── 核心结论 ── */
.conclusion{{background:linear-gradient(135deg,#fef2f2,#fff);border-left:4px solid #ef4444;border-radius:10px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.conclusion-title{{font-size:1.05rem;font-weight:700;color:#dc2626}}
.conclusion-title b{{font-size:1.15rem}}
.conclusion-sub{{font-size:.88rem;color:#64748b;margin-top:4px}}
.conclusion-warn{{font-size:.85rem;color:#dc2626;margin-top:4px}}

/* ── 通用区块 ── */
.section{{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.section-title{{font-size:1.05rem;font-weight:700;padding-bottom:12px;margin-bottom:16px;border-bottom:2px solid #e2e8f0}}

/* ── 板块分区卡片 ── */
.zone-card{{border-radius:10px;padding:14px 16px;margin-bottom:14px}}
.zone-card.chase{{background:#f0fdf4;border:1px solid #bbf7d0}}
.zone-card.watch{{background:#eff6ff;border:1px solid #bfdbfe}}
.zone-card.warn{{background:#fff7ed;border:1px solid #fed7aa}}
.zone-card.weak{{background:#f8fafc;border:1px solid #e2e8f0}}
.zone-header{{font-size:.95rem;font-weight:700;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,.06)}}
.zone-header .count{{font-size:.78rem;font-weight:400;color:#64748b;margin-left:4px}}

/* ── 板块项目 ── */
.zone-item{{padding:8px 0;border-bottom:1px solid #f1f5f9}}
.zone-item:last-child{{border-bottom:none}}
.zone-name{{font-size:.92rem;margin-bottom:2px}}
.zone-name .price{{color:#64748b;font-size:.82rem}}
.zone-sig{{margin:4px 0}}
.zone-pl{{font-size:.82rem;color:#475569}}
.zone-pl .neg{{color:#16a34a}}
.zone-pl .pos{{color:#dc2626}}

/* ── 板块标签 ── */
.tag{{display:inline-block;padding:1px 8px;border-radius:4px;font-size:.72rem;font-weight:600;margin:1px 2px}}
.tag-green{{background:#dcfce7;color:#166534}}
.tag-red{{background:#fef2f2;color:#dc2626}}
.tag-orange{{background:#fff7ed;color:#c2410c}}
.tag-yellow{{background:#fefce8;color:#a16207}}
.tag-gray{{background:#f1f5f9;color:#64748b}}

/* ── 观察区/弱势区项目 ── */
.w-item{{font-size:.85rem;padding:2px 0;color:#475569}}
.warn-item{{padding:4px 0;font-size:.88rem}}
.warn-item b{{color:#dc2626}}

/* ── 个股区块 ── */
.sector-block{{margin-bottom:16px}}
.sector-block:last-child{{margin-bottom:0}}
.sector-title{{font-size:.95rem;font-weight:700;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e2e8f0}}
.sector-title .score-badge{{display:inline-block;padding:1px 10px;border-radius:10px;background:#eef2ff;color:#4f46e5;font-size:.78rem;margin-left:6px;vertical-align:middle}}

.stock-card{{background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:10px 14px;margin-bottom:8px}}
.stock-card:last-child{{margin-bottom:0}}
.stock-head{{display:flex;align-items:center;gap:8px;margin-bottom:2px}}
.stock-score{{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem;color:#fff;flex-shrink:0}}
.score-high{{background:#4f46e5}}
.score-mid{{background:#0891b2}}
.score-low{{background:#78716c}}
.stock-name{{font-weight:600;font-size:.9rem}}
.stock-metrics{{font-size:.82rem;color:#64748b;margin:2px 0 2px 40px}}
.stock-risk{{font-size:.78rem;color:#dc2626;margin:2px 0 2px 40px}}
.stock-pl{{font-size:.82rem;color:#475569;margin-left:40px}}
.stock-pl .neg{{color:#16a34a}}
.stock-pl .pos{{color:#dc2626}}

/* ── 折叠按钮 ── */
.fold-body{{display:none}}
.fold-btn{{display:inline-block;padding:5px 14px;margin-top:6px;border:1px solid #cbd5e1;border-radius:6px;background:#f8fafc;color:#475569;font-size:.78rem;cursor:pointer;user-select:none}}
.fold-btn:hover{{background:#e2e8f0}}

/* ── 交易计划 ── */
.plan-item{{font-size:.88rem;padding:3px 0}}
.plan-list{{font-size:.85rem;color:#475569;padding:8px 0 4px 12px;line-height:1.8}}

/* ── 底部 ── */
.footer{{text-align:center;padding:12px 0;color:#94a3b8;font-size:.78rem}}

@media(max-width:640px){{body{{padding:8px}}.header{{padding:18px 20px}}.section{{padding:14px 16px}}}}
</style>
</head>
<body>
<div class="header"><h1>📊 每日交易参考</h1><div class="time">{date_str} · {now.strftime('%H:%M')} CST</div><div style="font-size:.78rem;color:#94a3b8;margin-top:2px">{_html.escape(data_date_str)}</div></div>
{content}
<div class="footer">⚠️ 技术指标参考，不构成投资建议<br><a href="https://stranger971020.github.io/trend-trading-system/reports/trading_report_guide.html" style="color:#6366f1;text-decoration:none">📖 完整策略说明</a></div>
<script>
function toggleFold(fid){{
  var el=document.getElementById(fid);if(!el)return;
  var h=el.style.display!=='none'&&el.style.display!=='';
  el.style.display=h?'none':'block';
  var btn=el.nextElementSibling;
  if(btn&&btn.classList.contains('fold-btn')){{
    var label=btn.getAttribute('data-label');
    btn.textContent=h?'▸ 展开全部 ('+label+')':'▸ 收起';
  }}
}}
</script>
</body>
</html>"""
    return html
