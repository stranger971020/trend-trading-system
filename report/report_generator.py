"""
报告生成器
- 组装模块1/2/3的结果为格式化的文本报告
- Telegram 兼容的 HTML 格式
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import BEIJING_TZ_OFFSET

logger = logging.getLogger(__name__)

# 北京时区
_BEIJING_TZ = timezone(timedelta(hours=BEIJING_TZ_OFFSET))


def _now_beijing() -> datetime:
    return datetime.now(_BEIJING_TZ)


def _format_pct(val: float) -> str:
    """格式化百分比。"""
    if val > 0:
        return f"+{val:.2f}%"
    return f"{val:.2f}%"


def _sentiment_emoji(sentiment: str) -> str:
    return {"Bullish": "📈", "Neutral": "📊", "Bearish": "📉"}.get(sentiment, "❓")


def generate_report(
    sentiment_result: dict,
    persistence_result: dict,
    stock_result: dict,
    module_status: dict,
    data_summary: dict,
    stock_derived_industry_result: dict = None,
    stock_picks_text: str = None,
    regime_result: dict = None,
) -> str:
    """生成完整的分析报告。

    Args:
        sentiment_result: 模块1输出
        persistence_result: 模块2输出（含 df）
        stock_result: 模块3输出
        module_status: {"module1": "success", "module2": "success", ...}
        data_summary: {"latest_date": "20260613", "industries_updated": 31, "total_rows": 155}
        stock_derived_industry_result: 个股推算行业指标输出（可选）

    Returns:
        格式化的 HTML 报告文本
    """
    now = _now_beijing()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

    lines = []

    # ==================== 标题 ====================
    lines.append(f"<b>A股趋势交易系统 - 日报</b>")
    lines.append(f"日期: {date_str} ({weekday})")
    lines.append(f"生成时间: {time_str} CST")
    lines.append("")

    # ==================== 一、市场情绪与板块持续性（合并） ====================
    lines.append("━" * 20)
    lines.append("<b>一、市场情绪与板块持续性</b>")
    lines.append("━" * 20)

    if sentiment_result.get("status") == "success":
        sent = sentiment_result.get("sentiment", "N/A")
        emoji = _sentiment_emoji(sent)
        avg_mom = sentiment_result.get("avg_momentum", 0)

        lines.append(f"情绪判定: <b>{sent}</b> {emoji}")
        lines.append(f"行业平均动量: {_format_pct(avg_mom)}")
        lines.append(
            f"上涨行业: {sentiment_result.get('bullish_count', 0)}个 | "
            f"下跌行业: {sentiment_result.get('bearish_count', 0)}个 | "
            f"持平: {sentiment_result.get('neutral_count', 0)}个"
        )

        # 背离预警
        warnings = sentiment_result.get("divergence_warnings", [])
        if warnings:
            lines.append("")
            lines.append(f"⚠️ <b>顶背离预警</b>: {', '.join(warnings)}")
        else:
            lines.append("")
            lines.append("✅ 无顶背离预警信号")

        # 仓位建议（显示中期+短期两个口径）
        regime_label = regime_result.get("regime", "N/A") if regime_result else "N/A"
        regime_pos = regime_result.get("position_advice", "N/A") if regime_result else "N/A"
        adx_v = regime_result.get("adx", 0) if regime_result else 0
        sent_pos = sentiment_result.get("position_advice", "N/A")
        lines.append("")
        lines.append("━" * 10)
        lines.append("<b>仓位建议（两口径）</b>")
        lines.append(f"  📈 中期趋势: <b>{regime_pos}</b>  |  {regime_label} · MA200+ADX={adx_v:.1f}")
        lines.append(f"  📊 短期情绪: <b>{sent_pos}</b>  |  {sent} · 20日动量 {_format_pct(avg_mom)}")
        lines.append(f"  💡 两口径不一致时，以中期趋势为主，短期情绪为辅")
    elif sentiment_result.get("status") == "degraded":
        lines.append(f"⚠️ 情绪判定: 数据不足，无法完整分析")
        lines.append(f"原因: {sentiment_result.get('error', '未知')}")
    else:
        lines.append(f"❌ 市场情绪分析失败: {sentiment_result.get('error', '未知错误')}")

    lines.append("")

    # 板块持续性排名
    if persistence_result.get("status") == "success":
        df = persistence_result.get("df")
        if df is not None and not df.empty:
            df = df.sort_values("persistence_score", ascending=False)
            lines.append(
                f"<code>{'排名':<4} {'板块':<10} {'持续性':<8} {'动量':<8} {'收益':<8} {'换手':<8} {'强度':<8} {'20日动量':<10}</code>"
            )

            for i, (_, row) in enumerate(df.iterrows()):
                rank = i + 1
                name = str(row.get("name", ""))[:10]
                pscore = row.get("persistence_score", 0)
                mscore = row.get("momentum_score", 0)
                rslope = row.get("return_slope", 0)
                tscore = row.get("turnover_score", 0)
                rscore = row.get("relative_strength", 0)
                ret20 = row.get("return_20d_pct", 0)
                label = str(row.get("label", ""))

                lines.append(
                    f"<code>{rank:>3}  {name:<10} {pscore:<8.2f} {mscore:<8.2f} "
                    f"{rslope:<8.2f} {tscore:<8.2f} {rscore:<8.2f} {ret20:<+9.1f}%</code>  {label}"
                )

            # 统计
            high_list = persistence_result.get("high_persistence", [])
            low_list = persistence_result.get("low_persistence", [])
            lines.append("")
            lines.append(f"<b>高持续性 ({len(high_list)}个)</b>: {', '.join(high_list[:8])}")
            if low_list:
                lines.append(f"<b>低持续性/预警 ({len(low_list)}个)</b>: {', '.join(low_list[:8])}")
        else:
            lines.append("⚠️ 无行业数据可供分析")
    elif persistence_result.get("status") == "degraded":
        lines.append(f"⚠️ 板块分析降级: {persistence_result.get('error', '')}")
    else:
        lines.append(f"❌ 板块持续性分析失败: {persistence_result.get('error', '未知错误')}")

    lines.append("")

    # ==================== 二、个股推算行业指标 ====================
    if stock_derived_industry_result is not None and stock_derived_industry_result.get("status") == "success":
        lines.append("━" * 20)
        lines.append("<b>二、个股推算行业指标（自下而上）</b>")
        lines.append("━" * 20)

        sdf = stock_derived_industry_result.get("df")
        if sdf is not None and not sdf.empty:
            from config import STOCK_DERIVED_TOP_N
            top_n = min(STOCK_DERIVED_TOP_N, len(sdf))
            sdf = sdf.head(top_n)
            lines.append(f"基于个股数据整合的 L2 行业指标（Top-{top_n}）:")
            lines.append("")
            lines.append(f"<code>{'排名':<4} {'二级行业':<12} {'成分股':<6} {'平均动量':<10} {'中位动量':<10} {'上涨比':<6} {'站上MA20':<8}</code>")
            for i, (_, row) in enumerate(sdf.iterrows()):
                lines.append(
                    f"<code>{i+1:>3}  {str(row.get('l2_name','')):<12} "
                    f"{int(row.get('stock_count',0)):<6} "
                    f"{float(row.get('avg_return_20d',0)):>+9.1f}% "
                    f"{float(row.get('median_return_20d',0)):>+9.1f}% "
                    f"{float(row.get('pct_positive',0)):>5.0f}% "
                    f"{float(row.get('pct_above_ma20',0)):>7.0f}%</code>"
                )
        else:
            lines.append("⚠️ 无足够数据推算行业指标")

        # 反转候选
        rev_df = stock_derived_industry_result.get("reversal_df")
        if rev_df is not None and not rev_df.empty:
            lines.append("")
            lines.append("━" * 10)
            lines.append("<b>🔁 反转候选 — 弱势行业中捕捉反弹信号</b>")
            lines.append("20日动量为负，但短期明显走强（反转强度 = 5日 - 20日），可能处于弱转强早期阶段:")
            lines.append("")
            lines.append(f"<code>{'行业':<12} {'个股':<6} {'5日动量':<10} {'20日动量':<10} {'反转强度':<10} {'站上MA5':<8}</code>")
            for _, row in rev_df.iterrows():
                lines.append(
                    f"<code>{str(row.get('l2_name','')):<12} "
                    f"{int(row.get('stock_count',0)):<6} "
                    f"{float(row.get('avg_return_5d',0)):>+9.1f}% "
                    f"{float(row.get('avg_return_20d',0)):>+9.1f}% "
                    f"{float(row.get('reversal_strength',0)):>+8.1f}% "
                    f"{float(row.get('pct_above_ma5',0)):>6.0f}%</code>"
                )

        lines.append("")

    # ==================== 三、个股挖掘 ====================
    lines.append("━" * 20)
    lines.append("<b>三、个股精选（从强势二级行业中优选）</b>")
    lines.append("━" * 20)

    if stock_result.get("status") == "success":
        stocks = stock_result.get("stocks", [])
        by_industry = stock_result.get("by_industry", {})

        if stocks:
            lines.append(f"从持续性 Top-{len(by_industry)} 个 L2 行业（按持续性降序）中精选 {len(stocks)} 只个股:")
            lines.append("")

            # 按行业分组展示（行业已按持续性得分降序）
            for ind_name, picks in by_industry.items():
                lines.append(f"<b>▸ {ind_name}</b>")
                for pick in picks:
                    excess_str = _format_pct(pick.get("excess_return", 0))
                    mom20d_str = _format_pct(pick.get("momentum_20d", 0))
                    mom5d_str = _format_pct(pick.get("momentum_5d", 0))
                    lines.append(
                        f"  <code>{pick['ts_code']:<12} {pick['name']:<8} "
                        f"评分:{pick['score']:<5.2f} "
                        f"超额:{excess_str:<8} "
                        f"20日:{mom20d_str:<8} "
                        f"5日:{mom5d_str}</code>"
                    )
                lines.append("")
        else:
            lines.append("未筛选出符合条件的个股")
    elif stock_result.get("status") == "skipped":
        reason = stock_result.get("reason", "MVP阶段待实现")
        # 如果是因数据不足跳过，给出具体原因
        if "数据为空" in reason or "不可用" in reason:
            lines.append(f"⏭️ 个股数据暂缺 — {reason}")
        else:
            lines.append(f"⏭️ {reason}")
    elif stock_result.get("status") == "degraded":
        lines.append(f"⚠️ 个股筛选降级: {stock_result.get('reason', stock_result.get('error', ''))}")
    else:
        lines.append(f"❌ 模块3失败: {stock_result.get('error', '未知错误')}")

    lines.append("")

    # ==================== 四、个股推荐（含止损与概率） ====================
    if stock_picks_text:
        lines.append(stock_picks_text)
        lines.append("")

    # ==================== 运行状态 ====================
    lines.append("━" * 20)
    lines.append("<b>运行状态</b>")
    lines.append("━" * 20)

    status_icons = {
        "success": "✅ 成功",
        "degraded": "⚠️ 降级",
        "failed": "❌ 失败",
        "skipped": "⏭️ 跳过",
    }

    for module, status in module_status.items():
        icon = status_icons.get(status, f"❓ {status}")
        lines.append(f"• {module}: {icon}")

    # 数据摘要
    latest_date = data_summary.get("latest_date", "N/A")
    updated = data_summary.get("industries_updated", 0)
    total_rows = data_summary.get("total_rows", 0)
    new_rows = data_summary.get("new_rows", 0)
    stocks_fetched = data_summary.get("stocks_fetched", 0)
    stocks_updated = data_summary.get("stocks_updated", 0)
    stocks_new = data_summary.get("stocks_new_rows", 0)

    lines.append(f"• 数据日期: {latest_date}")
    lines.append(f"• 行业: {updated} 个 | DB总记录: {total_rows} 条")
    if new_rows > 0:
        lines.append(f"• 行业新增: {new_rows} 条记录")
    if stocks_fetched > 0:
        lines.append(f"• 个股: {stocks_fetched} 只检测 | {stocks_updated} 只更新 (+{stocks_new}条)")

    lines.append("")
    lines.append("<i>=== END REPORT ===</i>")

    return "\n".join(lines)
