"""
HTML 报告生成器
- 自包含 HTML 文件（零外部依赖）
- 纯 CSS 图表（柱状图、进度条、热力图）
- 三模块完整展示
- GitHub Pages 友好
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import (
    BEIJING_TZ_OFFSET,
    HIGH_PERSISTENCE,
    MEDIUM_PERSISTENCE,
)

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=BEIJING_TZ_OFFSET))


def _now_beijing() -> datetime:
    return datetime.now(_BEIJING_TZ)


def _color_for_score(score: float, max_score: float = 10.0) -> str:
    """根据分数返回 HSL 颜色（红→黄→绿渐变）。"""
    ratio = max(0, min(1, score / max_score))
    hue = ratio * 120  # 0=红, 60=黄, 120=绿
    return f"hsl({hue:.0f}, 65%, 48%)"


def _color_for_momentum(momentum: float) -> str:
    """动量百分比 → 颜色。"""
    if momentum > 3:
        return "#22c55e"
    elif momentum > 0:
        return "#86efac"
    elif momentum > -3:
        return "#f59e0b"
    else:
        return "#ef4444"


def _bar_width(value: float, max_val: float, min_val: float = 0) -> int:
    """计算 CSS 柱状图宽度百分比。"""
    if max_val == min_val:
        return 50
    return max(2, min(100, int((value - min_val) / (max_val - min_val) * 100)))


def _sign(num: float) -> str:
    return "+" if num > 0 else ""


def _sentiment_badge(sentiment: str) -> tuple[str, str, str]:
    """返回 (bg_color, text_color, emoji)。"""
    if sentiment == "Bullish":
        return "#dcfce7", "#166534", "📈"
    elif sentiment == "Bearish":
        return "#fee2e2", "#991b1b", "📉"
    else:
        return "#fef3c7", "#92400e", "📊"


def _persistence_class(score: float) -> tuple[str, str, str]:
    """返回 (label, color, bg_color)。"""
    if score >= HIGH_PERSISTENCE:
        return "高持续性", "#16a34a", "#f0fdf4"
    elif score >= MEDIUM_PERSISTENCE:
        return "中等持续性", "#d97706", "#fffbeb"
    else:
        return "低持续性", "#dc2626", "#fef2f2"


_FOLD_COUNTER = 0


def _fold_id() -> str:
    """生成唯一的折叠区域 ID。"""
    global _FOLD_COUNTER
    _FOLD_COUNTER += 1
    return f"fold-{_FOLD_COUNTER}"


def _build_fold_button(fold_id: str, hidden_count: int) -> str:
    """生成折叠/展开按钮 HTML。"""
    return (
        f'<button class="toggle-btn" onclick="toggleFold(this,\'{fold_id}\')" '
        f'data-label="{hidden_count}项">▸ 展开全部 ({hidden_count}项)</button>'
    )


def _render_table_with_fold(
    visible_rows: str,
    hidden_rows: str,
    hidden_count: int,
    fold_id: str,
) -> str:
    """将可见行和隐藏行包裹为带折叠按钮的表格结构。

    在调用此函数之前，<table> 和 <thead> 应该已经在 visible_rows 之前输出，
    visible_rows 应该以 </tbody> 结束。
    """
    if hidden_count <= 0:
        return visible_rows

    return (
        f"{visible_rows}\n"
        f'<tbody id="{fold_id}" style="display:none;">\n'
        f"{hidden_rows}\n"
        f"</tbody>\n"
        f"{_build_fold_button(fold_id, hidden_count)}"
    )


# ============================================================
# CSS 样式（内嵌）
# ============================================================

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    background: #f1f5f9;
    color: #1e293b;
    line-height: 1.6;
    padding: 16px;
    max-width: 1100px;
    margin: 0 auto;
}

/* ===== HEADER ===== */
.header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.header h1 { font-size: 1.5rem; font-weight: 700; }
.header .meta { font-size: 0.9rem; opacity: 0.85; }
.header .data-freshness {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
}
.fresh { background: #22c55e; color: white; }
.stale { background: #f59e0b; color: #1e293b; }

/* ===== EXECUTIVE SUMMARY ===== */
.summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}
.summary-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-left: 5px solid #e2e8f0;
}
.summary-card h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; color: #64748b; }

/* Sentiment big badge */
.sentiment-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.sentiment-detail { font-size: 0.9rem; color: #475569; margin-top: 8px; }

/* Position gauge */
.position-gauge {
    margin-top: 8px;
}
.gauge-bar {
    height: 8px; border-radius: 4px;
    background: linear-gradient(to right, #ef4444 0%, #f59e0b 40%, #22c55e 70%, #22c55e 100%);
    margin-bottom: 8px;
    position: relative;
}
.gauge-marker {
    width: 18px; height: 18px; border-radius: 50%;
    border: 3px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    position: relative; top: -13px;
    transition: left 0.3s;
}
.gauge-text { font-weight: 700; font-size: 1.1rem; }

/* Risk card */
.risk-list { list-style: none; }
.risk-list li {
    padding: 6px 0; border-bottom: 1px solid #fee2e2;
    font-size: 0.9rem;
}
.risk-list li:last-child { border-bottom: none; }
.no-risk { color: #16a34a; font-weight: 600; }

/* ===== SECTION TITLES ===== */
.section {
    background: white;
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ===== DISTRIBUTION BARS ===== */
.dist-stats {
    display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;
}
.dist-stat {
    flex: 1; min-width: 80px; text-align: center;
    padding: 10px; border-radius: 8px;
}
.dist-stat .count { font-size: 1.4rem; font-weight: 700; }
.dist-stat .lbl { font-size: 0.8rem; color: #64748b; }

/* Momentum bar chart */
.bar-chart { margin-top: 8px; }
.bar-row {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 4px; font-size: 0.82rem;
}
.bar-row .name { width: 72px; text-align: right; flex-shrink: 0; color: #475569; }
.bar-row .bar-wrap {
    flex: 1; height: 18px; background: #f1f5f9;
    border-radius: 3px; overflow: hidden;
}
.bar-row .bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.5s;
    min-width: 2px;
}
.bar-row .val { width: 56px; text-align: right; flex-shrink: 0; font-weight: 600; font-size: 0.78rem; }

/* ===== DIVERGENCE WARNING ===== */
.warning-box {
    background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 8px; padding: 16px; margin-top: 16px;
}
.warning-box h4 { color: #dc2626; margin-bottom: 8px; }

/* ===== TABLE ===== */
.ranking-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.85rem;
}
.ranking-table th {
    background: #f8fafc; padding: 10px 8px;
    text-align: left; font-weight: 600; color: #64748b;
    border-bottom: 2px solid #e2e8f0;
    white-space: nowrap;
}
.ranking-table td {
    padding: 8px; border-bottom: 1px solid #f1f5f9;
    vertical-align: middle;
}
.ranking-table tr:hover { background: #f8fafc; }
.ranking-table .rank { width: 32px; text-align: center; font-weight: 700; color: #94a3b8; }
.ranking-table .sector-name { font-weight: 600; }

/* Mini bar in table cell */
.mini-bar {
    display: inline-block; height: 6px; border-radius: 3px;
    vertical-align: middle; margin-right: 6px;
}

/* Persistence badge in table */
.p-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 600;
}

/* Summary row */
.table-summary {
    display: flex; gap: 20px; margin-top: 16px;
    font-size: 0.9rem; flex-wrap: wrap;
}
.table-summary span { font-weight: 600; }

/* ===== STOCK CARDS ===== */
.stock-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
}
.stock-card {
    background: white; border-radius: 10px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
}
.stock-card-header {
    padding: 12px 16px;
    font-weight: 700;
    font-size: 0.95rem;
    border-bottom: 1px solid #e2e8f0;
}
.stock-card-body { padding: 12px 16px; }
.stock-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #f8fafc;
}
.stock-row:last-child { border-bottom: none; }
.stock-score {
    width: 36px; height: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    color: white; flex-shrink: 0;
}
.stock-info { flex: 1; }
.stock-info .code { font-weight: 600; font-size: 0.88rem; }
.stock-info .name { font-size: 0.82rem; color: #64748b; }
.stock-metrics { text-align: right; font-size: 0.78rem; }
.stock-metrics .positive { color: #dc2626; }
.stock-metrics .negative { color: #16a34a; }

/* ===== FOOTER ===== */
.footer {
    background: white; border-radius: 12px;
    padding: 20px 28px; margin-top: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    font-size: 0.82rem; color: #94a3b8;
}
.footer .status-grid {
    display: flex; gap: 24px; margin-bottom: 12px; flex-wrap: wrap;
}
.footer .status-item { display: flex; align-items: center; gap: 6px; }
.footer .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
}
.footer .dot-success { background: #22c55e; }
.footer .dot-warning { background: #f59e0b; }
.footer .dot-failed { background: #ef4444; }
.footer .dot-skipped { background: #94a3b8; }
.footer .disclaimer { margin-top: 12px; padding-top: 12px; border-top: 1px solid #f1f5f9; }

/* ===== COLLAPSIBLE ===== */
.toggle-btn {
    display: inline-block; padding: 6px 16px; margin-top: 8px;
    border: 1px solid #cbd5e1; border-radius: 6px;
    background: #f8fafc; color: #475569;
    font-size: 0.8rem; cursor: pointer;
    text-align: center;
    user-select: none;
}
.toggle-btn:hover { background: #e2e8f0; }

/* ===== RESPONSIVE ===== */
@media (max-width: 640px) {
    body { padding: 8px; }
    .header { padding: 20px; }
    .header h1 { font-size: 1.2rem; }
    .summary { grid-template-columns: 1fr; }
    .stock-grid { grid-template-columns: 1fr; }
    .bar-row .name { width: 56px; font-size: 0.72rem; }
    .ranking-table { font-size: 0.75rem; }
}
"""


# ============================================================
# HTML 报告生成
# ============================================================

def generate_html_report(
    sentiment_result: dict,
    persistence_result: dict,
    stock_result: dict,
    module_status: dict,
    data_summary: dict,
    l3_leading_result: dict | None = None,
    l3_persistence_result: dict | None = None,
    l2_leading_result: dict | None = None,
    l2_persistence_result: dict | None = None,
    regime_result: dict | None = None,
    crowding_result: dict | None = None,
    portfolio_result: dict | None = None,
    anomaly_result: dict | None = None,
    time_slot: str = "evening",
    persistence_trend: dict | None = None,
    stock_derived_industry_result: dict | None = None,
) -> str:
    """生成自包含 HTML 报告。

    Args:
        sentiment_result: 模块1输出
        persistence_result: 模块2输出（含 df）
        stock_result: 模块3输出
        module_status: 各模块执行状态
        data_summary: 数据摘要
        l2_leading_result: 模块0-L2 二级行业领先信号输出
        l2_persistence_result: 模块2 L2 持续性输出
        l3_leading_result: 模块0 L3 领先信号输出（可选）
        l3_persistence_result: 模块2 L3 持续性输出（可选）
        regime_result: 宏观状态机输出（可选）
        crowding_result: 拥挤度预警输出（可选）
        stock_derived_industry_result: 个股推算行业指标输出（可选）

    Returns:
        完整的 HTML 字符串
    """
    now = _now_beijing()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    latest_date = data_summary.get("latest_date", "N/A")

    # 判断数据新鲜度（2个交易日内算新鲜）
    freshness_class = "fresh"
    l3d = data_summary.get("l3_latest_date", "N/A")
    sd = data_summary.get("stock_latest_date", "N/A")
    freshness_text = f"行业: {latest_date} | L3: {l3d} | 个股: {sd}"
    try:
        data_dt = datetime.strptime(str(latest_date), "%Y%m%d").date()
        days_behind = (now.date() - data_dt).days
        if days_behind > 3:
            freshness_class = "stale"
            freshness_text = f"数据延迟 {days_behind}天"
    except Exception:
        pass

    parts: list[str] = []

    # ---- HTML 头部 ----
    parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股趋势交易系统 · 日报 {date_str}</title>
<style>{CSS}</style>
<script>
function toggleFold(btn, tbodyId) {{
    var tbody = document.getElementById(tbodyId);
    if (!tbody) return;
    var hidden = tbody.style.display === 'none' || tbody.style.display === '';
    tbody.style.display = hidden ? 'table-row-group' : 'none';
    var label = btn.getAttribute('data-label');
    btn.textContent = hidden ? '▸ 收起' : ('▸ 展开全部 (' + label + ')');
}}
</script>
</head>
<body>
""")

    # ---- Header ----
    parts.append(f"""
<div class="header">
  <div>
    <h1>📊 A股趋势交易系统</h1>
    <div class="meta">日报 · {date_str} ({weekday}) · 生成 {time_str} CST</div>
  </div>
  <div class="data-freshness {freshness_class}">{freshness_text}</div>
</div>
""")

    # ---- Executive Summary ----
    parts.append(_build_executive_summary(sentiment_result, l3_leading_result))

    # === 一、大盘研判 ===
    l1_date = data_summary.get("latest_date", "N/A")
    parts.append(f"""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#6366f1;border-top:2px solid #6366f1;border-bottom:2px solid #6366f1;">一、大盘研判 — 申万一级行业 · 整体情绪与持续性 <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(数据: {l1_date})</span></div>""")
    parts.append(_build_module1(sentiment_result))
    parts.append(_build_module2(persistence_result))

    # === 持续性量化参考 ===
    parts.append(_build_persistence_legend())

    # === 持续性趋势 ===
    if persistence_trend:
        parts.append(_build_trend_section(persistence_trend))

    # === 二、投资方向 ===
    if l2_leading_result is not None:
        l2_date = data_summary.get("l2_latest_date", data_summary.get("latest_date", "N/A"))
        parts.append(f"""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#16a34a;border-top:2px solid #16a34a;border-bottom:2px solid #16a34a;">二、投资方向 — 申万二级行业 · ETF/赛道选择 <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(数据: {l2_date})</span></div>""")
        parts.append(_build_l2_section(l2_leading_result, l2_persistence_result))

    # === 三、个股推算行业指标 ===
    if stock_derived_industry_result is not None and stock_derived_industry_result.get("status") == "success":
        sd_date = data_summary.get("stock_latest_date", "N/A")
        parts.append(f"""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#8b5cf6;border-top:2px solid #8b5cf6;border-bottom:2px solid #8b5cf6;">三、个股推算行业指标 — 基于个股数据自下而上整合 <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(个股数据: {sd_date})</span></div>""")
        parts.append(_build_stock_derived_industry(stock_derived_industry_result))

    # === 四、个股精选 ===
    stock_date = data_summary.get("stock_latest_date", "N/A")
    mf_date = data_summary.get("moneyflow_latest_date", "N/A")
    source_level = stock_result.get("source_level", "二级")
    source_label = "二级行业" if source_level == "L2" else ("一级行业" if source_level == "L1" else "三级行业")
    parts.append(f"""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#dc2626;border-top:2px solid #dc2626;border-bottom:2px solid #dc2626;">四、个股精选 — 从强势{source_label}中优选（已排除银行/非银金融） <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(个股: {stock_date} | 资金流: {mf_date})</span></div>""")
    parts.append(_build_module3(stock_result))

    # === 五、风控汇总 ===
    if regime_result or crowding_result:
        parts.append("""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#f59e0b;border-top:2px solid #f59e0b;border-bottom:2px solid #f59e0b;">五、风控与监控</div>""")
        parts.append(_build_risk_section(regime_result, crowding_result, portfolio_result, anomaly_result))

    # ---- Footer ----
    parts.append(_build_footer(module_status, data_summary))

    # ---- HTML 尾部 ----
    parts.append("</body>\n</html>")

    return "\n".join(parts)


def _build_executive_summary(sentiment_result: dict, l3_leading_result: dict | None = None) -> str:
    """构建三列摘要卡片。"""
    sent = sentiment_result.get("sentiment", "N/A")
    avg_mom = sentiment_result.get("avg_momentum", 0)
    warnings = sentiment_result.get("divergence_warnings", [])
    position = sentiment_result.get("position_advice", "N/A")
    bullish = sentiment_result.get("bullish_count", 0)
    bearish = sentiment_result.get("bearish_count", 0)
    neutral = sentiment_result.get("neutral_count", 0)

    # 情绪卡片
    bg, color, emoji = _sentiment_badge(sent)
    mom_str = f"{_sign(avg_mom)}{avg_mom:.2f}%"
    mom_color = _color_for_momentum(avg_mom)

    # 仓位指针位置
    if "7-8" in position or "8" in position:
        gauge_pct = 85
    elif "5" in position:
        gauge_pct = 50
    elif "2-3" in position or "2" in position:
        gauge_pct = 20
    else:
        gauge_pct = 35

    # L3 领先信号增强仓位建议
    l3_note = ""
    if l3_leading_result and l3_leading_result.get("status") == "success":
        leading_count = l3_leading_result.get("leading_count", 0)
        if leading_count >= 10:
            l3_note = f"（{leading_count}个L3领先→可适当上浮）"

    # 风险卡片
    if warnings:
        risk_html = "<ul class='risk-list'>"
        for w in warnings[:8]:
            risk_html += f"<li>⚠️ {w}</li>"
        risk_html += "</ul>"
    else:
        risk_html = "<div class='no-risk'>✅ 无顶背离预警</div>"

    return f"""
<div class="summary">
  <div class="summary-card" style="border-left-color: {color};">
    <h3>市场情绪</h3>
    <div class="sentiment-badge" style="background:{bg};color:{color};">{emoji} {sent}</div>
    <div class="sentiment-detail">
      行业平均动量: <b style="color:{mom_color}">{mom_str}</b><br>
      上涨 {bullish} · 下跌 {bearish} · 持平 {neutral}
    </div>
  </div>

  <div class="summary-card" style="border-left-color: #6366f1;">
    <h3>仓位建议</h3>
    <div class="position-gauge">
      <div class="gauge-bar"></div>
      <div class="gauge-marker" style="left: calc({gauge_pct}% - 9px); background: {_color_for_score(gauge_pct/100*10)};"></div>
    </div>
    <div class="gauge-text">{position}{l3_note}</div>
  </div>

  <div class="summary-card" style="border-left-color: {'#ef4444' if warnings else '#22c55e'};">
    <h3>风险预警</h3>
    {risk_html}
    <div style="margin-top:8px;font-size:0.82rem;color:#64748b;">
      顶背离行业: {len(warnings)} 个
    </div>
  </div>
</div>
"""


def _build_module1(sentiment_result: dict) -> str:
    """构建合并模块第一部分：4 统计大数 + MACD 顶背离预警。

    动量分布柱状图已整合到 _build_module2 的表格中（20日动量列）。
    """
    bullish = sentiment_result.get("bullish_count", 0)
    bearish = sentiment_result.get("bearish_count", 0)
    neutral = sentiment_result.get("neutral_count", 0)
    total = bullish + bearish + neutral

    # 顶背离预警
    warnings = sentiment_result.get("divergence_warnings", [])
    if warnings:
        warn_html = f"""<div class="warning-box">
  <h4>⚠️ MACD 顶背离预警</h4>
  <p style="color:#7f1d1d;">以下行业出现价格新高但 MACD DIF 未创新高，提示见顶风险：</p>
  <p style="margin-top:6px;"><b>{'、'.join(warnings)}</b></p>
</div>"""
    else:
        warn_html = """<div class="warning-box" style="border-color:#bbf7d0;background:#f0fdf4;">
  <h4 style="color:#16a34a;">✅ 无顶背离信号</h4>
  <p style="color:#166534;">当前无行业触发 MACD 顶背离，市场结构健康。</p>
</div>"""

    return f"""
<div class="section">
  <div class="section-title">📋 市场情绪与板块持续性</div>

  <div class="dist-stats">
    <div class="dist-stat" style="background:#fef2f2;">
      <div class="count" style="color:#dc2626;">{bullish}</div>
      <div class="lbl">上涨行业</div>
    </div>
    <div class="dist-stat" style="background:#f8fafc;">
      <div class="count" style="color:#64748b;">{neutral}</div>
      <div class="lbl">持平行业</div>
    </div>
    <div class="dist-stat" style="background:#f0fdf4;">
      <div class="count" style="color:#16a34a;">{bearish}</div>
      <div class="lbl">下跌行业</div>
    </div>
    <div class="dist-stat" style="background:#f8fafc;">
      <div class="count" style="color:#6366f1;">{total}</div>
      <div class="lbl">总计</div>
    </div>
  </div>

  {warn_html}
</div>
"""


def _build_persistence_legend() -> str:
    """持续性量化参考卡。

    基于 2000-2026 全部 L1 行业历史数据回测。
    """
    return """
<div class="section" style="background:linear-gradient(135deg,#f0fdf4,#fef3c7,#fef2f2);">
  <div class="section-title">📐 持续性量化参考</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;font-size:.82rem;">
    <div style="background:#fff;border-radius:8px;padding:14px;border-left:4px solid #16a34a;">
      <b style="color:#16a34a;">🔥 高持续性 (≥7.0)</b>
      <div style="margin-top:6px;color:#475569;">
        历史20日后均收益 <b style="color:#16a34a;">+2.90%</b><br>
        胜率 <b>60%</b> · 适合重仓参与
      </div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:14px;border-left:4px solid #d97706;">
      <b style="color:#d97706;">⚡ 中等持续性 (5.0-7.0)</b>
      <div style="margin-top:6px;color:#475569;">
        历史20日后均收益 <b style="color:#d97706;">+0.65%</b><br>
        胜率 <b>51%</b> · 轻仓试探或观望
      </div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:14px;border-left:4px solid #dc2626;">
      <b style="color:#dc2626;">⚠️ 低持续性 (<5.0)</b>
      <div style="margin-top:6px;color:#475569;">
        历史20日后均收益 <b style="color:#dc2626;">+0.34%</b><br>
        胜率 <b>50%</b> · 随机游走，建议回避
      </div>
    </div>
  </div>
  <p style="font-size:.72rem;color:#94a3b8;margin-top:8px;">📊 基于 2000-2026 年 31 个申万 L1 行业历史数据回测 | 买点信号: 持续性从低→中突破时</p>
</div>
"""


def _build_l2_section(l2_leading, l2_persistence):
    """L2 投资方向：领先信号 + 持续性排名（带折叠）。"""
    FOLD_THRESHOLD = 5
    ldf = l2_leading.get("df") if l2_leading else None
    pdf = l2_persistence.get("df") if l2_persistence else None

    parts = []
    # Leading signals
    if ldf is not None and not ldf.empty:
        leading_count = l2_leading.get("leading_count", 0)
        strong = l2_leading.get("strong_leading", [])
        top15 = ldf.head(15)
        total_rows = len(top15)
        hidden_count = max(0, total_rows - FOLD_THRESHOLD)
        fold_id = _fold_id() if hidden_count > 0 else ""

        vis_rows = ""
        hid_rows = ""
        for i, (_, row) in enumerate(top15.iterrows()):
            excess = float(row["excess_momentum"])
            color = _color_for_score(max(0, excess + 5), 20)
            bar_w = _bar_width(excess + 10, 30, 0)
            row_html = f'<tr><td class="rank">{int(row["rank"])}</td><td>{row["l2_name"]}</td><td style="color:#64748b;font-size:.8rem;">{row["parent_name"]}</td><td><span class="mini-bar" style="width:{bar_w}px;background:{color};"></span>{_sign(excess)}{excess:.1f}%</td><td><span class="p-badge" style="color:{color};background:#f8fafc;">{row["label"]}</span></td></tr>\n'
            if i < FOLD_THRESHOLD:
                vis_rows += row_html
            else:
                hid_rows += row_html

        fold_html = ""
        if hidden_count > 0:
            fold_html = (
                f'<tbody id="{fold_id}" style="display:none;">\n{hid_rows}</tbody>\n'
                f'{_build_fold_button(fold_id, hidden_count)}'
            )

        strong_tags = ""
        if strong:
            tags = ["%s(%+.1f%%)" % (s["name"], s["excess"]) for s in strong[:10]]
            strong_tags = "<p style='margin-bottom:8px;'><b>🔥 强烈领先:</b> " + "、".join(tags) + "</p>"

        parts.append(f"""<div class="section">
  <div class="section-title">🔍 L2 领先信号</div>
  <p style="font-size:.85rem;color:#64748b;margin-bottom:8px;">二级行业相对其所属一级行业的超额动量排名。共 <b>{leading_count}</b> 个领先。</p>
  {strong_tags}
  <table class="ranking-table"><thead><tr><th>#</th><th>二级行业</th><th>所属一级</th><th>超额动量</th><th>强度</th></tr></thead><tbody>{vis_rows}</tbody>{fold_html}</table>
  <p style="font-size:.78rem;color:#94a3b8;margin-top:4px;">共 {len(ldf)} 个 L2 参评</p>
</div>""")

    # Persistence
    if pdf is not None and not pdf.empty:
        top = pdf.head(20)
        total_rows = len(top)
        hidden_count = max(0, total_rows - FOLD_THRESHOLD)
        fold_id2 = _fold_id() if hidden_count > 0 else ""

        vis_rows2 = ""
        hid_rows2 = ""
        for i, (_, row) in enumerate(top.iterrows()):
            ps = float(row["persistence_score"])
            color = _color_for_score(ps)
            row_html = f'<tr><td class="rank">{int(row.get("rank",0))}</td><td>{row["name"]}</td><td><span class="mini-bar" style="width:{_bar_width(ps,10)}px;background:{color};"></span>{ps:.2f}</td><td><span class="p-badge" style="color:{color};background:#f8fafc;">{row["label"]}</span></td></tr>\n'
            if i < FOLD_THRESHOLD:
                vis_rows2 += row_html
            else:
                hid_rows2 += row_html

        fold_html2 = ""
        if hidden_count > 0:
            fold_html2 = (
                f'<tbody id="{fold_id2}" style="display:none;">\n{hid_rows2}</tbody>\n'
                f'{_build_fold_button(fold_id2, hidden_count)}'
            )

        parts.append(f"""<div class="section">
  <div class="section-title">📊 L2 持续性 Top-20</div>
  <table class="ranking-table"><thead><tr><th>#</th><th>二级行业</th><th>持续性得分</th><th>判定</th></tr></thead><tbody>{vis_rows2}</tbody>{fold_html2}</table>
</div>""")

    return "\n".join(parts) if parts else ""


def _build_stock_derived_industry(result: dict) -> str:
    """构建个股推算行业指标章节。

    展示基于个股数据自下而上整合的 L2 行业动量指标。
    """
    df = result.get("df")
    if df is None or df.empty:
        return ""

    from config import STOCK_DERIVED_TOP_N

    top_n = min(STOCK_DERIVED_TOP_N, len(df))
    df = df.head(top_n)

    FOLD_THRESHOLD = 5
    total_rows = len(df)
    hidden_count = max(0, total_rows - FOLD_THRESHOLD)
    fold_id = _fold_id() if hidden_count > 0 else ""

    max_ret = max(abs(df["avg_return_20d"].max()), abs(df["avg_return_20d"].min()), 5)

    visible_rows = ""
    hidden_rows = ""
    for i, (_, row) in enumerate(df.iterrows()):
        rank = int(row.get("rank", i + 1))
        l2_name = str(row.get("l2_name", ""))
        count = int(row.get("stock_count", 0))
        avg_ret = float(row.get("avg_return_20d", 0))
        med_ret = float(row.get("median_return_20d", 0))
        pct_pos = float(row.get("pct_positive", 0))
        pct_ma = float(row.get("pct_above_ma20", 0))

        # 迷你柱状图和颜色
        bar_w = _bar_width(abs(avg_ret), max_ret, 0)
        ret_color = "#ef4444" if avg_ret > 0 else "#22c55e"
        pos_color = _color_for_score(pct_pos, 100)

        row_html = (
            f'<tr>'
            f'<td class="rank">{rank}</td>'
            f'<td class="sector-name">{l2_name}</td>'
            f'<td style="text-align:center;">{count}</td>'
            f'<td>'
            f'  <span class="mini-bar" style="width:{bar_w}px;background:{ret_color};"></span>'
            f'  <span style="color:{ret_color};font-weight:600;">{_sign(avg_ret)}{avg_ret:.1f}%</span>'
            f'</td>'
            f'<td style="color:{ret_color};">{_sign(med_ret)}{med_ret:.1f}%</td>'
            f'<td><span class="mini-bar" style="width:{int(pct_pos/100*60)}px;background:{pos_color};"></span>{pct_pos:.0f}%</td>'
            f'<td>{pct_ma:.0f}%</td>'
            f'</tr>\n'
        )

        if i < FOLD_THRESHOLD:
            visible_rows += row_html
        else:
            hidden_rows += row_html

    fold_html = ""
    if hidden_count > 0:
        fold_html = (
            f'<tbody id="{fold_id}" style="display:none;">\n'
            f'{hidden_rows}'
            f'</tbody>\n'
            f'{_build_fold_button(fold_id, hidden_count)}'
        )

    return f"""
<div class="section">
  <div class="section-title">📊 个股推算 L2 行业指标（自下而上）</div>
  <p style="font-size:0.85rem;color:#64748b;margin-bottom:12px;">
    基于 {result.get("l2_count", 0)} 个 L2 行业的成分股数据整合计算。
    个股日线数据聚合 → 行业级别指标，与个股数据时效性一致。
  </p>
  <div style="overflow-x:auto;">
  <table class="ranking-table">
    <thead>
      <tr>
        <th>#</th><th>二级行业</th><th>成分股数</th>
        <th>平均20日动量</th><th>中位数动量</th><th>上涨比例</th><th>站上MA20</th>
      </tr>
    </thead>
    <tbody>
      {visible_rows}
    </tbody>
    {fold_html}
  </table>
  </div>
</div>
"""


def _build_trend_section(trend_data: dict) -> str:
    """持续性得分趋势追踪（带折叠）。"""
    if not trend_data:
        return ""
    FOLD_THRESHOLD = 5
    all_rows = []
    for code, scores in sorted(trend_data.items(), key=lambda x: -(x[1][-1][1] if x[1] else 0)):
        if len(scores) < 2:
            continue
        name = scores[-1][2] if len(scores[-1]) > 2 else code
        latest_score = scores[-1][1]
        prev_score = scores[0][1] if len(scores) == 2 else scores[-2][1]
        delta = latest_score - prev_score
        delta_str = f"{delta:+.1f}" if delta != 0 else "—"
        delta_color = "#16a34a" if delta > 0 else ("#dc2626" if delta < 0 else "#94a3b8")
        bars = ""
        for item in scores[-10:]:
            sc = item[1]
            bars += f'<span title="{item[0]}: {sc:.1f}" style="display:inline-block;width:7px;height:14px;background:{_color_for_score(sc)};margin-right:1px;border-radius:1px;vertical-align:middle;"></span>'
        all_rows.append(f'<tr><td>{scores[-1][0]}</td><td>{name}</td><td>{bars}</td><td><b>{latest_score:.1f}</b></td><td style="color:{delta_color};font-weight:600;">{delta_str}</td></tr>\n')

    if not all_rows:
        return ""

    total_rows = len(all_rows)
    hidden_count = max(0, total_rows - FOLD_THRESHOLD)
    fold_id = _fold_id() if hidden_count > 0 else ""
    vis_rows = "".join(all_rows[:FOLD_THRESHOLD])
    hid_rows = "".join(all_rows[FOLD_THRESHOLD:]) if hidden_count > 0 else ""

    fold_html = ""
    if hidden_count > 0:
        fold_html = (
            f'<tbody id="{fold_id}" style="display:none;">\n{hid_rows}</tbody>\n'
            f'{_build_fold_button(fold_id, hidden_count)}'
        )

    return f"""
<div class="section">
  <div class="section-title">📈 持续性趋势追踪（Top-{total_rows} L1 行业）</div>
  <p style="font-size:.78rem;color:#94a3b8;margin-bottom:8px;">近10日持续性得分变化，颜色越绿越好。需要积累2天以上历史数据。</p>
  <table class="ranking-table">
    <thead><tr><th>最新日期</th><th>行业</th><th>趋势</th><th>最新分</th><th>变化</th></tr></thead>
    <tbody>{vis_rows}</tbody>
    {fold_html}
  </table>
</div>"""


def _build_module2(persistence_result: dict) -> str:
    """构建持续性排名表格（含 20日动量列，折叠长列表）。"""
    df = persistence_result.get("df")
    if df is None or df.empty:
        return """<div class="section"><div class="section-title">📊 板块持续性排名</div><p>数据不足</p></div>"""

    # 确保降序排列
    df = df.sort_values("persistence_score", ascending=False).reset_index(drop=True)

    # 子因子最大值（用于迷你柱状图）
    max_p = df["persistence_score"].max() if not df.empty else 10
    max_m = df["momentum_score"].max() if "momentum_score" in df.columns else 10

    FOLD_THRESHOLD = 5
    total_rows = len(df)
    hidden_count = max(0, total_rows - FOLD_THRESHOLD)
    fold_id = _fold_id() if hidden_count > 0 else ""

    visible_rows = ""
    hidden_rows = ""
    for i, (_, row) in enumerate(df.iterrows()):
        rank = i + 1
        name = str(row.get("name", ""))
        pscore = float(row.get("persistence_score", 0))
        mscore = float(row.get("momentum_score", 0))
        rslope = float(row.get("return_slope", 0))
        tscore = float(row.get("turnover_score", 0))
        rscore = float(row.get("relative_strength", 0))
        ret_20d = float(row.get("return_20d_pct", 0))
        label, color, bg = _persistence_class(pscore)

        p_bar_w = _bar_width(pscore, max_p)
        m_bar_w = _bar_width(mscore, max_m)
        bar_color = _color_for_score(pscore)

        # 20日动量颜色（红涨绿跌）
        mom_color = "#ef4444" if ret_20d > 0 else "#22c55e" if ret_20d < 0 else "#64748b"

        row_html = (
            f'<tr>'
            f'<td class="rank">{rank}</td>'
            f'<td class="sector-name">{name}</td>'
            f'<td>'
            f'  <span class="mini-bar" style="width:{p_bar_w}px;background:{bar_color};"></span>'
            f'  {pscore:.2f}'
            f'</td>'
            f'<td><span class="mini-bar" style="width:{m_bar_w}px;background:#6366f1;"></span>{mscore:.2f}</td>'
            f'<td>{rslope:.2f}</td>'
            f'<td>{tscore:.2f}</td>'
            f'<td>{rscore:.2f}</td>'
            f'<td style="color:{mom_color};font-weight:600;">{_sign(ret_20d)}{ret_20d:.1f}%</td>'
            f'<td><span class="p-badge" style="color:{color};background:{bg}">{label}</span></td>'
            f'</tr>\n'
        )

        if i < FOLD_THRESHOLD:
            visible_rows += row_html
        else:
            hidden_rows += row_html

    high_list = persistence_result.get("high_persistence", [])
    medium_list = persistence_result.get("medium_persistence", [])
    low_list = persistence_result.get("low_persistence", [])

    fold_html = ""
    if hidden_count > 0:
        fold_html = (
            f'<tbody id="{fold_id}" style="display:none;">\n'
            f'{hidden_rows}'
            f'</tbody>\n'
            f'{_build_fold_button(fold_id, hidden_count)}'
        )

    return f"""
<div class="section">
  <div class="section-title">📊 板块持续性排名（含20日动量）</div>
  <div style="overflow-x:auto;">
  <table class="ranking-table">
    <thead>
      <tr>
        <th>#</th><th>行业</th><th>持续性得分</th>
        <th>动量分</th><th>收益斜率</th><th>换手率</th><th>相对强度</th>
        <th>20日动量</th><th>判定</th>
      </tr>
    </thead>
    <tbody>
      {visible_rows}
    </tbody>
    {fold_html}
  </table>
  </div>
  <div class="table-summary">
    <span style="color:#16a34a;">🔥 高持续性: {len(high_list)} 个</span>
    <span style="color:#d97706;">⚡ 中等持续性: {len(medium_list)} 个</span>
    <span style="color:#dc2626;">⚠️ 低持续性: {len(low_list)} 个</span>
  </div>
</div>
"""


def _build_module3(stock_result: dict) -> str:
    """构建模块3：个股精选卡片。"""
    status = stock_result.get("status", "skipped")

    if status in ("skipped", "failed", "degraded"):
        reason = stock_result.get("reason") or stock_result.get("error") or "未知原因"
        return f"""<div class="section">
  <div class="section-title">🎯 个股精选</div>
  <p style="color:#94a3b8;">{'⏭️ 跳过' if status == 'skipped' else '⚠️ 降级' if status == 'degraded' else '❌ 失败'}: {reason}</p>
</div>"""

    by_industry = stock_result.get("by_industry", {})
    stocks = stock_result.get("stocks", [])

    if not by_industry:
        return """<div class="section">
  <div class="section-title">🎯 个股精选</div>
  <p style="color:#94a3b8;">未筛选出符合条件的个股</p>
</div>"""

    cards_html = ""
    for ind_name, picks in by_industry.items():
        # 行业卡片颜色
        first_score = picks[0]["score"] if picks else 5
        card_color = _color_for_score(first_score)

        rows_html = ""
        for i, pick in enumerate(picks):
            score = pick["score"]
            excess = pick.get("excess_return", 0)
            mom20d = pick.get("momentum_20d", 0)
            mom5d = pick.get("momentum_5d", 0)
            score_color = _color_for_score(score)
            exc_class = "positive" if excess > 0 else "negative"
            mom20d_class = "positive" if mom20d > 0 else "negative"
            mom_class = "positive" if mom5d > 0 else "negative"

            stop_html = ""
            stop_price_val = pick.get("stop_loss_price")
            if stop_price_val is not None:
                atr_pct = pick.get("atr_pct", 0)
                stop_html = f'<br><span style="font-size:.7rem;color:#94a3b8;">🛑 {stop_price_val:.2f} (ATR {atr_pct:.1f}%)</span>'

            # 基本面数据
            funda = pick.get("fundamental", {})
            funda_html = ""
            if funda:
                parts_f = []
                if "pe_pct" in funda:
                    parts_f.append(f'PE分位{funda["pe_pct"]}%{" ✅" if funda["pe_pct"]<30 else ""}')
                if "pb_pct" in funda:
                    parts_f.append(f'PB分位{funda["pb_pct"]}%{" ✅" if funda["pb_pct"]<30 else ""}')
                if "roe" in funda:
                    trend = funda.get("roe_trend", "")
                    parts_f.append(f'ROE {funda["roe"]}%{trend}')
                if "funda_bonus" in funda and funda["funda_bonus"] > 0:
                    parts_f.append(f'+{funda["funda_bonus"]:.1f}')
                if parts_f:
                    funda_html = '<br><span style="font-size:.7rem;color:#6366f1;">' + " · ".join(parts_f) + '</span>'

            rows_html += (
                f'<div class="stock-row">'
                f'<div class="stock-score" style="background:{score_color};">{score:.1f}</div>'
                f'<div class="stock-info">'
                f'  <div class="code">{pick["ts_code"]}</div>'
                f'  <div class="name">{pick["name"]}</div>'
                f'</div>'
                f'<div class="stock-metrics">'
                f'  <span>超额 </span><span class="{exc_class}">{_sign(excess)}{excess:.1f}%</span><br>'
                f'  <span>20日 </span><span class="{mom20d_class}">{_sign(mom20d)}{mom20d:.1f}%</span> '
                f'  <span>5日 </span><span class="{mom_class}">{_sign(mom5d)}{mom5d:.1f}%</span>'
                f'  {stop_html}'
                f'  {funda_html}'
                f'</div>'
                f'</div>\n'
            )

        cards_html += (
            f'<div class="stock-card">'
            f'<div class="stock-card-header" style="border-bottom-color:{card_color};">'
            f'📌 {ind_name} ({len(picks)}只)'
            f'</div>'
            f'<div class="stock-card-body">{rows_html}</div>'
            f'</div>\n'
        )

    return f"""
<div class="section">
  <div class="section-title">🎯 个股精选</div>
  <p style="font-size:0.85rem;color:#64748b;margin-bottom:16px;">
    从持续性 Top-{len(by_industry)} 个 L2 行业中精选 {len(stocks)} 只个股（按持续性得分降序排列）
    <br>评分由 ML LightGBM LambdaRank 模型综合多因子计算（含超额收益、20日动量、5日动量、MA偏离、基本面等）
    （已排除银行/证券/保险类）
  </p>
  <div class="stock-grid">
    {cards_html}
  </div>
</div>
"""


def _build_risk_section(regime_result: dict | None, crowding_result: dict | None, portfolio_result: dict | None = None, anomaly_result: dict | None = None) -> str:
    """构建风控汇总章节。"""
    regime_html = ""
    if regime_result and regime_result.get("regime"):
        r = regime_result
        regime_colors = {"BULL": "#16a34a", "RANGE": "#d97706", "BEAR": "#dc2626"}
        regime_bg = {"BULL": "#f0fdf4", "RANGE": "#fffbeb", "BEAR": "#fef2f2"}
        color = regime_colors.get(r["regime"], "#64748b")
        bg = regime_bg.get(r["regime"], "#f8fafc")
        regime_html = f"""
        <div class="summary-card" style="border-left-color:{color};background:{bg};">
          <h3>🌐 宏观状态: {r['regime']}</h3>
          <div style="font-size:1.2rem;font-weight:700;color:{color};">{r.get('position_advice', '')}</div>
          <div style="font-size:.8rem;color:#64748b;margin-top:6px;">{r.get('details', '')}</div>
          <div style="font-size:.78rem;color:#94a3b8;">ADX={r.get('adx','?')} | MA50={r.get('ma50','?')} | MA200={r.get('ma200','?')}</div>
        </div>"""

    crowding_html = ""
    if crowding_result:
        crowded = crowding_result.get("crowded_industries", [])
        if crowded:
            crowding_html = f"""
        <div class="summary-card" style="border-left-color:#ef4444;background:#fef2f2;">
          <h3>⚠️ 拥挤度预警</h3>
          <div style="font-size:.9rem;color:#dc2626;">{'、'.join(crowded[:6])}</div>
          <div style="font-size:.78rem;color:#64748b;margin-top:4px;">{len(crowded)} 个行业拥挤（成交额>90%分位+动量回落）</div>
        </div>"""
        else:
            crowding_html = """
        <div class="summary-card" style="border-left-color:#22c55e;background:#f0fdf4;">
          <h3>✅ 拥挤度正常</h3>
          <div style="font-size:.9rem;color:#16a34a;">无行业触发拥挤预警</div>
        </div>"""

    # Virtual portfolio card
    portfolio_html = ""
    if portfolio_result:
        nav = portfolio_result.get("nav", 1.0)
        positions = portfolio_result.get("total_positions", 0)
        pret = (nav - 1) * 100
        portfolio_html = f"""
        <div class="summary-card" style="border-left-color:#6366f1;">
          <h3>📊 虚拟持仓</h3>
          <div style="font-size:1.2rem;font-weight:700;">NAV: {nav:.4f} ({pret:+.2f}%)</div>
          <div style="font-size:.8rem;color:#64748b;">{positions} 只持仓 | {portfolio_result.get('new_positions',0)} 新开 {portfolio_result.get('closed',0)} 平仓</div>
        </div>"""

    # Anomaly alerts
    anomaly_html = ""
    if anomaly_result and anomaly_result.get("alerts"):
        alerts = anomaly_result["alerts"]
        alert_items = ""
        for a in alerts:
            alert_items += f'<div style="font-size:.82rem;margin-top:4px;">• {a["type"]}: {a["detail"]}</div>'
        anomaly_html = f"""
        <div class="summary-card" style="border-left-color:#f59e0b;background:#fffbeb;">
          <h3>⚠️ 异常归因: {anomaly_result["summary"]}</h3>
          {alert_items}
        </div>"""
    elif anomaly_result:
        anomaly_html = """
        <div class="summary-card" style="border-left-color:#22c55e;background:#f0fdf4;">
          <h3>✅ 无异常</h3>
          <div style="font-size:.9rem;color:#16a34a;">今日结果与近期均值一致</div>
        </div>"""

    if not regime_html and not crowding_html and not portfolio_html:
        return ""

    return f"""
<div class="section">
  <div class="section-title">🛡️ 风控与监控</div>
  <div class="summary">
    {regime_html}
    {crowding_html}
    {portfolio_html}
    {anomaly_html}
  </div>
</div>
"""


def _build_footer(module_status: dict, data_summary: dict) -> str:
    """构建页脚：运行状态和数据摘要。"""
    status_icons = {
        "success": ("dot-success", "成功"),
        "degraded": ("dot-warning", "降级"),
        "failed": ("dot-failed", "失败"),
        "skipped": ("dot-skipped", "跳过"),
        "pending": ("dot-skipped", "等待"),
    }

    status_items = ""
    for module, status in module_status.items():
        dot, label = status_icons.get(status, ("dot-skipped", status))
        status_items += (
            f'<div class="status-item">'
            f'<span class="status-dot {dot}"></span> {module}: {label}'
            f'</div>'
        )

    latest_date = data_summary.get("latest_date", "N/A")
    updated = data_summary.get("industries_updated", 0)
    total_rows = data_summary.get("total_rows", 0)
    stocks_fetched = data_summary.get("stocks_fetched", 0)
    stocks_updated = data_summary.get("stocks_updated", 0)
    stocks_new = data_summary.get("stocks_new_rows", 0)

    now = _now_beijing()
    gen_time = now.strftime("%Y-%m-%d %H:%M:%S CST")

    return f"""
<div class="footer">
  <div class="status-grid">
    {status_items}
  </div>
  <div style="font-size:0.78rem;">
    📦 行业数据: {updated}个行业 · {total_rows}条记录 · 最新 {latest_date}<br>
    📦 个股数据: {stocks_fetched}只检测 · {stocks_updated}只更新 (+{stocks_new}条)
  </div>
  <div class="disclaimer">
    ⚠️ 本报告由 A股趋势交易系统 自动生成，仅供研究参考，不构成投资建议。<br>
    数据来源: Tushare / akshare · 生成时间: {gen_time}
  </div>
</div>
"""
