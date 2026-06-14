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
) -> str:
    """生成完整的分析报告。

    Args:
        sentiment_result: 模块1输出
        persistence_result: 模块2输出（含 df）
        stock_result: 模块3输出
        module_status: {"module1": "success", "module2": "success", ...}
        data_summary: {"latest_date": "20260613", "industries_updated": 31, "total_rows": 155}

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

    # ==================== 模块1：市场情绪 ====================
    lines.append("━" * 20)
    lines.append("<b>模块1: 市场情绪与择时</b>")
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

        # 仓位建议
        lines.append("")
        lines.append(f"<b>仓位建议:</b> {sentiment_result.get('position_advice', 'N/A')}")
    elif sentiment_result.get("status") == "degraded":
        lines.append(f"⚠️ 情绪判定: 数据不足，无法完整分析")
        lines.append(f"原因: {sentiment_result.get('error', '未知')}")
    else:
        lines.append(f"❌ 模块1失败: {sentiment_result.get('error', '未知错误')}")

    lines.append("")

    # ==================== 模块2：板块持续性 ====================
    lines.append("━" * 20)
    lines.append("<b>模块2: 板块持续性排名</b>")
    lines.append("━" * 20)

    if persistence_result.get("status") == "success":
        df = persistence_result.get("df")
        if df is not None and not df.empty:
            lines.append(
                f"<code>{'排名':<4} {'板块':<10} {'持续性':<8} {'动量':<8} {'收益':<8} {'换手':<8} {'强度':<8}</code>"
            )

            for _, row in df.iterrows():
                rank = int(row.get("rank", 0))
                name = str(row.get("name", ""))[:10]
                pscore = row.get("persistence_score", 0)
                mscore = row.get("momentum_score", 0)
                rslope = row.get("return_slope", 0)
                tscore = row.get("turnover_score", 0)
                rscore = row.get("relative_strength", 0)
                label = str(row.get("label", ""))

                lines.append(
                    f"<code>{rank:>3}  {name:<10} {pscore:<8.2f} {mscore:<8.2f} "
                    f"{rslope:<8.2f} {tscore:<8.2f} {rscore:<8.2f}</code>  {label}"
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
        lines.append(f"❌ 模块2失败: {persistence_result.get('error', '未知错误')}")

    lines.append("")

    # ==================== 模块3：个股挖掘 ====================
    lines.append("━" * 20)
    lines.append("<b>模块3: 个股精选</b>")
    lines.append("━" * 20)

    if stock_result.get("status") == "success":
        stocks = stock_result.get("stocks", [])
        by_industry = stock_result.get("by_industry", {})

        if stocks:
            lines.append(f"从 {len(by_industry)} 个行业中精选 {len(stocks)} 只个股:")
            lines.append("")

            # 按行业分组展示
            for ind_name, picks in by_industry.items():
                lines.append(f"<b>▸ {ind_name}</b>")
                for pick in picks:
                    excess_str = _format_pct(pick.get("excess_return", 0))
                    mom_str = _format_pct(pick.get("momentum_5d", 0))
                    lines.append(
                        f"  <code>{pick['ts_code']:<12} {pick['name']:<8} "
                        f"评分:{pick['score']:<5.2f} "
                        f"超额:{excess_str:<8} "
                        f"5日动量:{mom_str}</code>"
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
