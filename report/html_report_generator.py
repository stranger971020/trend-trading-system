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
    parts.append(f"""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#6366f1;border-top:2px solid #6366f1;border-bottom:2px solid #6366f1;">一、大盘研判 — 申万一级行业 · 整体情绪与趋势 <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(数据: {l1_date})</span></div>""")
    parts.append(_build_module1(sentiment_result, persistence_result))
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

    # === 三、个股精选 ===
    stock_date = data_summary.get("stock_latest_date", "N/A")
    mf_date = data_summary.get("moneyflow_latest_date", "N/A")
    parts.append(f"""<div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#dc2626;border-top:2px solid #dc2626;border-bottom:2px solid #dc2626;">三、个股精选 — 从强势三级行业中优选（已排除银行/非银金融） <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(个股: {stock_date} | 资金流: {mf_date})</span></div>""")
    parts.append(_build_module3(stock_result))

    # === 风控汇总 ===
    if regime_result or crowding_result:
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


def _build_module1(sentiment_result: dict, persistence_result: dict) -> str:
    """构建模块1：市场情绪详览 + 动量分布图。"""
    # 从 persistence df 获取各行业的动量数据
    df = persistence_result.get("df")
    if df is None or df.empty:
        return """<div class="section"><div class="section-title">📋 模块1: 市场情绪详览</div><p>数据不足</p></div>"""

    # 按 return_20d_pct 排序用于柱状图
    if "return_20d_pct" in df.columns:
        chart_df = df.sort_values("return_20d_pct", ascending=True).copy()
        max_ret = max(abs(chart_df["return_20d_pct"].max()), abs(chart_df["return_20d_pct"].min()), 5)
    else:
        chart_df = df.copy()
        max_ret = 10

    # 动量分布柱状图
    bars_html = ""
    for _, row in chart_df.iterrows():
        name = str(row.get("name", ""))
        ret = float(row.get("return_20d_pct", 0))
        pct = _bar_width(abs(ret), max_ret, 0)
        color = "#ef4444" if ret > 0 else "#22c55e"  # A股红涨绿跌
        bar_style = f"width:{pct}%;background:{color};"
        bars_html += (
            f'<div class="bar-row">'
            f'<span class="name">{name}</span>'
            f'<span class="bar-wrap"><span class="bar-fill" style="{bar_style}"></span></span>'
            f'<span class="val" style="color:{color}">{_sign(ret)}{ret:.1f}%</span>'
            f'</div>\n'
        )

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

    bullish = sentiment_result.get("bullish_count", 0)
    bearish = sentiment_result.get("bearish_count", 0)
    neutral = sentiment_result.get("neutral_count", 0)
    total = bullish + bearish + neutral

    return f"""
<div class="section">
  <div class="section-title">📋 模块1: 市场情绪与动量分布</div>

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

  <p style="font-size:0.85rem;color:#64748b;margin-bottom:8px;">
    📊 31个申万一级行业 20日动量分布（红涨绿跌）
  </p>
  <div class="bar-chart">
    {bars_html}
  </div>

  {warn_html}
</div>
"""


def _build_module0(l3_leading_result: dict) -> str:
    """构建模块0：三级行业领先信号。"""
    status = l3_leading_result.get("status")
    if status != "success":
        return ""

    df = l3_leading_result.get("df")
    leading_count = l3_leading_result.get("leading_count", 0)
    strong = l3_leading_result.get("strong_leading", [])

    if df is None or df.empty:
        return ""

    # 取前15和后5
    top15 = df.head(15)
    bottom5 = df.tail(5)

    rows = ""
    for _, row in top15.iterrows():
        excess = float(row["excess_momentum"])
        color = _color_for_score(max(0, excess + 5), 20)
        bar_w = _bar_width(excess + 10, 30, 0)
        rows += (
            f'<tr>'
            f'<td class="rank">{int(row["rank"])}</td>'
            f'<td>{row["l3_name"]}</td>'
            f'<td style="color:#64748b;font-size:0.8rem;">{row["parent_name"]}</td>'
            f'<td><span class="mini-bar" style="width:{bar_w}px;background:{color};"></span>'
            f'{_sign(excess)}{excess:.1f}%</td>'
            f'<td><span class="p-badge" style="color:{color};background:#f8fafc;">{row["label"]}</span></td>'
            f'</tr>\n'
        )

    # 强烈领先标签（避免 f-string 内引号转义问题）
    strong_tags = ""
    if strong:
        tags = []
        for s in strong[:10]:
            tags.append(f"{s['name']}({_sign(s['excess'])}{s['excess']:.1f}%)")
        strong_tags = "<p style='margin-bottom:8px;'><b>🔥 强烈领先:</b> " + "、".join(tags) + "</p>"

    # 领先统计摘要
    leading_summary = ""
    if leading_count > 0:
        leading_summary = (
            f'共 <b style="color:#6366f1;">{leading_count}</b> 个三级行业领先（超额≥2%），'
            f'其中 <b style="color:#dc2626;">{len(strong)}</b> 个强烈领先（≥5%）'
        )

    return f"""
<div class="section">
  <div class="section-title">🔍 模块0: 三级行业领先信号</div>
  <p style="font-size:0.85rem;color:#64748b;margin-bottom:12px;">
    三级行业相对其所属一级行业的超额动量排名。
    {leading_summary}
  </p>

  {strong_tags}

  <div style="overflow-x:auto;">
  <table class="ranking-table">
    <thead>
      <tr><th>#</th><th>三级行业</th><th>所属一级</th><th>超额动量</th><th>强度</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p style="font-size:0.78rem;color:#94a3b8;margin-top:8px;">
    📊 共 {len(df)} 个三级行业参评。超额动量 = L3_20日收益 − 所属L1_20日收益
  </p>
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
    """L2 投资方向：领先信号 + 持续性排名。"""
    ldf = l2_leading.get("df") if l2_leading else None
    pdf = l2_persistence.get("df") if l2_persistence else None

    parts = []
    # Leading signals
    if ldf is not None and not ldf.empty:
        leading_count = l2_leading.get("leading_count", 0)
        strong = l2_leading.get("strong_leading", [])
        top15 = ldf.head(15)
        rows = ""
        for _, row in top15.iterrows():
            excess = float(row["excess_momentum"])
            color = _color_for_score(max(0, excess + 5), 20)
            bar_w = _bar_width(excess + 10, 30, 0)
            rows += f'<tr><td class="rank">{int(row["rank"])}</td><td>{row["l2_name"]}</td><td style="color:#64748b;font-size:.8rem;">{row["parent_name"]}</td><td><span class="mini-bar" style="width:{bar_w}px;background:{color};"></span>{_sign(excess)}{excess:.1f}%</td><td><span class="p-badge" style="color:{color};background:#f8fafc;">{row["label"]}</span></td></tr>\n'

        strong_tags = ""
        if strong:
            tags = ["%s(%+.1f%%)" % (s["name"], s["excess"]) for s in strong[:10]]
            strong_tags = "<p style='margin-bottom:8px;'><b>🔥 强烈领先:</b> " + "、".join(tags) + "</p>"

        parts.append(f"""<div class="section">
  <div class="section-title">🔍 L2 领先信号</div>
  <p style="font-size:.85rem;color:#64748b;margin-bottom:8px;">二级行业相对其所属一级行业的超额动量排名。共 <b>{leading_count}</b> 个领先。</p>
  {strong_tags}
  <table class="ranking-table"><thead><tr><th>#</th><th>二级行业</th><th>所属一级</th><th>超额动量</th><th>强度</th></tr></thead><tbody>{rows}</tbody></table>
  <p style="font-size:.78rem;color:#94a3b8;margin-top:4px;">共 {len(ldf)} 个 L2 参评</p>
</div>""")

    # Persistence
    if pdf is not None and not pdf.empty:
        top = pdf.head(20)
        rows2 = ""
        for _, row in top.iterrows():
            ps = float(row["persistence_score"])
            color = _color_for_score(ps)
            rows2 += f'<tr><td class="rank">{int(row.get("rank",0))}</td><td>{row["name"]}</td><td><span class="mini-bar" style="width:{_bar_width(ps,10)}px;background:{color};"></span>{ps:.2f}</td><td><span class="p-badge" style="color:{color};background:#f8fafc;">{row["label"]}</span></td></tr>\n'
        parts.append(f"""<div class="section">
  <div class="section-title">📊 L2 持续性 Top-20</div>
  <table class="ranking-table"><thead><tr><th>#</th><th>二级行业</th><th>持续性得分</th><th>判定</th></tr></thead><tbody>{rows2}</tbody></table>
</div>""")

    return "\n".join(parts) if parts else ""


def _build_l3_direction(l3_leading: dict, l3_persistence: dict) -> str:
    """构建 L3 投资方向：按 L1 父级分组展示强势 L3。"""
    pdf = l3_persistence.get("df")
    ldf = l3_leading.get("df")

    if pdf is None or pdf.empty:
        return ""

    # 只展示中等持续性以上的 L3
    from config import MEDIUM_PERSISTENCE
    strong_l3 = pdf[pdf["persistence_score"] >= MEDIUM_PERSISTENCE].head(30)

    # 按 L1 父级分组 (从 L3 leading 数据中查找 parent)
    l3_to_parent = {}
    if ldf is not None and not ldf.empty:
        for _, row in ldf.iterrows():
            l3_to_parent[row["l3_code"]] = row.get("parent_name", "")

    # 手工分组
    groups: dict[str, list] = {}
    for _, row in strong_l3.iterrows():
        code = row["ts_code"]
        name = row["name"]
        score = row["persistence_score"]
        parent = l3_to_parent.get(code, "其他")
        if parent not in groups:
            groups[parent] = []
        groups[parent].append((name, score, code))

    if not groups:
        return ""

    cards = ""
    for parent_name, items in sorted(groups.items(), key=lambda x: -max(s for _, s, _ in x[1])):
        if not parent_name:
            continue
        item_html = ""
        for name, score, code in items[:8]:
            bar_w = int(score / 10 * 120)
            color = _color_for_score(score)
            item_html += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.8rem;">'
                f'<span style="min-width:80px;text-align:right;color:#475569;">{name}</span>'
                f'<span style="flex:1;height:6px;background:#f1f5f9;border-radius:3px;">'
                f'<span style="display:block;height:100%;width:{bar_w}px;background:{color};border-radius:3px;"></span></span>'
                f'<span style="font-weight:600;min-width:36px;">{score:.1f}</span>'
                f'</div>'
            )
        cards += (
            f'<div style="background:#fff;border-radius:8px;padding:12px;border-left:4px solid {_color_for_score(max(s for _,s,_ in items))};">'
            f'<b style="font-size:.9rem;">📌 {parent_name}</b>'
            f'{item_html}'
            f'</div>'
        )

    return f"""
<div class="section">
  <div class="section-title">🎯 L3 投资方向（按一级行业分组）</div>
  <p style="font-size:.82rem;color:#64748b;margin-bottom:12px;">
    展示各一级行业下属的三级子行业持续性得分。高分三级行业 = ETF/赛道投资方向。
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;">
    {cards}
  </div>
</div>
"""


def _build_trend_section(trend_data: dict) -> str:
    """持续性得分趋势追踪。"""
    if not trend_data:
        return ""
    rows = ""
    # 从 persistence_result 获取 name 映射
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
        rows += f'<tr><td>{scores[-1][0]}</td><td>{name}</td><td>{bars}</td><td><b>{latest_score:.1f}</b></td><td style="color:{delta_color};font-weight:600;">{delta_str}</td></tr>\n'

    if not rows:
        return ""
    return f"""
<div class="section">
  <div class="section-title">📈 持续性趋势追踪（Top-10 L1 行业）</div>
  <p style="font-size:.78rem;color:#94a3b8;margin-bottom:8px;">近10日持续性得分变化，颜色越绿越好。需要积累2天以上历史数据。</p>
  <table class="ranking-table">
    <thead><tr><th>最新日期</th><th>行业</th><th>趋势</th><th>最新分</th><th>变化</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>""" if rows else ""


def _build_module2(persistence_result: dict) -> str:
    """构建模块2：板块持续性排名表格。"""
    df = persistence_result.get("df")
    if df is None or df.empty:
        return """<div class="section"><div class="section-title">📊 模块2: 板块持续性排名</div><p>数据不足</p></div>"""

    # 计算用户看到的子因子最大值（用于迷你柱状图）
    max_p = df["persistence_score"].max() if not df.empty else 10
    max_m = df["momentum_score"].max() if "momentum_score" in df.columns else 10

    rows_html = ""
    for _, row in df.iterrows():
        rank = int(row.get("rank", 0))
        name = str(row.get("name", ""))
        pscore = float(row.get("persistence_score", 0))
        mscore = float(row.get("momentum_score", 0))
        rslope = float(row.get("return_slope", 0))
        tscore = float(row.get("turnover_score", 0))
        rscore = float(row.get("relative_strength", 0))
        label, color, bg = _persistence_class(pscore)

        # 迷你柱状图
        p_bar_w = _bar_width(pscore, max_p)
        m_bar_w = _bar_width(mscore, max_m)
        bar_color = _color_for_score(pscore)

        rows_html += (
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
            f'<td><span class="p-badge" style="color:{color};background:{bg}">{label}</span></td>'
            f'</tr>\n'
        )

    high_list = persistence_result.get("high_persistence", [])
    medium_list = persistence_result.get("medium_persistence", [])
    low_list = persistence_result.get("low_persistence", [])

    return f"""
<div class="section">
  <div class="section-title">📊 模块2: 板块持续性排名</div>
  <div style="overflow-x:auto;">
  <table class="ranking-table">
    <thead>
      <tr>
        <th>#</th><th>行业</th><th>持续性得分</th>
        <th>动量分</th><th>收益斜率</th><th>换手率</th><th>相对强度</th><th>判定</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
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
  <div class="section-title">🎯 模块3: 个股精选</div>
  <p style="color:#94a3b8;">{'⏭️ 跳过' if status == 'skipped' else '⚠️ 降级' if status == 'degraded' else '❌ 失败'}: {reason}</p>
</div>"""

    by_industry = stock_result.get("by_industry", {})
    stocks = stock_result.get("stocks", [])

    if not by_industry:
        return """<div class="section">
  <div class="section-title">🎯 模块3: 个股精选</div>
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
            mom5d = pick.get("momentum_5d", 0)
            score_color = _color_for_score(score)
            exc_class = "positive" if excess > 0 else "negative"
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
                f'  <span>5日动量 </span><span class="{mom_class}">{_sign(mom5d)}{mom5d:.1f}%</span>'
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
  <div class="section-title">🎯 模块3: 个股精选</div>
  <p style="font-size:0.85rem;color:#64748b;margin-bottom:16px;">
    从持续性 Top-{min(7, len(by_industry))} 行业中精选 {len(stocks)} 只个股
    （已排除银行/证券/保险类，评分 = 超额收益×50% + 5日动量×30% + MA20偏离×20%）
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
  <div class="section-title">🛡️ 风控与监控 (第四+五阶段)</div>
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
