"""
报告生成器
- 组装模块1/2/3的结果为格式化的文本报告
- Telegram 兼容的 HTML 格式
- iPhone 紧凑排版适配（每行不超过 50 字符）
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


def _stock_link(ts_code: str, label: str = None) -> str:
    """生成同花顺个股超链接。"""
    if not ts_code:
        return label or ""
    parts = ts_code.split(".")
    if len(parts) != 2:
        return label or ts_code
    num, ex = parts
    prefix = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}.get(ex.upper(), "SH")
    url = f"https://stockpage.10jqka.com.cn/{prefix}{num}/"
    return f'<a href="{url}" target="_blank">{label or ts_code}</a>'


def _format_pct(val: float) -> str:
    """格式化百分比。"""
    if val > 0:
        return f"+{val:.2f}%"
    return f"{val:.2f}%"


def _sentiment_emoji(sentiment: str) -> str:
    return {"Bullish": "📈", "Neutral": "📊", "Bearish": "📉"}.get(sentiment, "❓")


# ── 关键点位格式化 ──

def _format_key_levels(kl_result: dict) -> list:
    """将 key_levels_calculator 的结果格式化为 Telegram HTML 报告行。"""
    lines = []
    if not kl_result or not kl_result.get("success"):
        lines.append("⚠️ 关键点位数据不可用")
        return lines

    levels = kl_result["levels"]
    status = kl_result["status"]

    lines.append("━" * 20)
    lines.append("<b>【大盘关键点位】</b>")
    lines.append("━" * 20)
    lines.append(f"当前点位: <b>{levels['close_price']}</b>")
    lines.append("")
    lines.append(f"  🛡️ 强支撑: {levels['strong_support']} | 极强支撑: {levels['ultra_support']}")
    lines.append(f"  🔴 中档阻力: {levels['resistance_mid']} | 强阻力: {levels['resistance_high']}")
    lines.append(f"  🚩 突破确认: {levels['breakout_confirm']}")
    lines.append("")
    lines.append(f"  VWAP: {levels['vwap']} | 日内: {levels['day_low']}~{levels['day_high']}")
    lines.append(f"  MA5/10/20/60: {levels['ma_5']}/{levels['ma_10']}/{levels['ma_20']}/{levels['ma_60']}")
    lines.append(f"  量比: {levels['volume_ratio']}x")
    lines.append("")
    lines.append(f"  🎯 状态: <b>{status['status']}</b> | 建议: {status['action_cn']} ({status['confidence']})")
    lines.append(f"  {status['details']}")
    lines.append("")
    return lines


def generate_report(
    sentiment_result: dict,
    persistence_result: dict,
    stock_result: dict,
    module_status: dict,
    data_summary: dict,
    stock_derived_industry_result: dict = None,
    stock_picks_text: str = None,
    regime_result: dict = None,
    key_levels_result: dict = None,
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

        # 仓位建议（显示中期+短期+策略参数）
        regime_label = regime_result.get("regime", "N/A") if regime_result else "N/A"
        regime_pos = regime_result.get("position_advice", "N/A") if regime_result else "N/A"
        adx_v = regime_result.get("adx", 0) if regime_result else 0
        sent_pos = sentiment_result.get("position_advice", "N/A")
        lines.append("")
        lines.append("━" * 10)
        lines.append("<b>仓位建议 + 策略参数</b>")
        lines.append(f"  📈 中期趋势: <b>{regime_pos}</b>  |  {regime_label} · MA200+ADX={adx_v:.1f}")
        lines.append(f"  📊 短期情绪: <b>{sent_pos}</b>  |  {sent} · 20日动量 {_format_pct(avg_mom)}")

        # 新增: 7状态细粒度市场状态 + 策略参数
        if regime_result and "regime_v2" in regime_result:
            rv2 = regime_result["regime_v2_label"]
            rdesc = regime_result["regime_v2_desc"]
            sp = regime_result.get("strategy_params", {})
            ra20 = regime_result.get("returns_20d", 0)
            ra60 = regime_result.get("returns_60d", 0)
            m20 = regime_result.get("ma20_short", 0)
            am20 = "✅" if regime_result.get("above_ma20") else "❌"
            lines.append(f"  🏷️ 细粒度状态: <b>{rv2}</b> | {rdesc}")
            lines.append(f"     │ MA20={m20:.0f} {am20} | 20日{ra20:+.2f}% | 60日{ra60:+.2f}%")
            lines.append(f"     │ 💡 策略: 选股{sp.get('n','?')} 持有{sp.get('hold','?')} 止损{sp.get('sl','?')} 止盈{sp.get('tp','?')}")
        lines.append(f"  💡 两口径不一致时，以中期趋势为主，短期情绪为辅")
    elif sentiment_result.get("status") == "degraded":
        lines.append(f"⚠️ 情绪判定: 数据不足，无法完整分析")
        lines.append(f"原因: {sentiment_result.get('error', '未知')}")
    else:
        lines.append(f"❌ 市场情绪分析失败: {sentiment_result.get('error', '未知错误')}")

    lines.append("")

    # ==================== 关键点位（R5 整合） ====================
    if key_levels_result:
        lines.extend(_format_key_levels(key_levels_result))

    # 板块持续性排名（仅展示前 5 条，手机紧凑格式 ~45 字符/行）
    if persistence_result.get("status") == "success":
        df = persistence_result.get("df")
        if df is not None and not df.empty:
            df = df.sort_values("persistence_score", ascending=False)
            total_inds = len(df)
            MAX_VISIBLE_IND = 5

            for i, (_, row) in enumerate(df.iterrows()):
                if i >= MAX_VISIBLE_IND:
                    break
                rank = i + 1
                name = str(row.get("name", ""))[:8]
                pscore = row.get("persistence_score", 0)
                ret20 = row.get("return_20d_pct", 0)
                label = str(row.get("label", ""))
                lines.append(
                    f"<code>{rank}. {name:<8} {pscore:<5.2f} {ret20:<+7.1f}%</code> {label}"
                )

            hidden_inds = total_inds - MAX_VISIBLE_IND
            if hidden_inds > 0:
                lines.append(f"<i>…… 还有 {hidden_inds} 个行业的详细评分，详情见完整 HTML 报告</i>")
                lines.append("")

            # 统计
            high_list = persistence_result.get("high_persistence", [])
            low_list = persistence_result.get("low_persistence", [])
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
            MAX_VISIBLE_DERIVED = 5
            total_derived = len(stock_derived_industry_result.get("df"))
            lines.append(f"基于个股数据整合的 L2 行业指标（展示前 {MAX_VISIBLE_DERIVED}，共 {total_derived}）:")
            lines.append("")
            for i, (_, row) in enumerate(sdf.iterrows()):
                if i >= MAX_VISIBLE_DERIVED:
                    break
                lines.append(
                    f"<code>{i+1}. {str(row.get('l2_name',''))[:8]:<8} "
                    f"动量:{float(row.get('avg_return_20d',0)):<+7.1f}% "
                    f"涨比:{float(row.get('pct_positive',0)):<4.0f}%</code>"
                )
            hidden_derived = total_derived - MAX_VISIBLE_DERIVED
            if hidden_derived > 0:
                lines.append(f"<i>…… 还有 {hidden_derived} 个行业指标，详情见完整 HTML 报告</i>")
        else:
            lines.append("⚠️ 无足够数据推算行业指标")

        # 反转候选（仅展示前 5 条，手机紧凑格式）
        rev_df = stock_derived_industry_result.get("reversal_df")
        MAX_VISIBLE_REV = 5
        if rev_df is not None and not rev_df.empty:
            lines.append("")
            lines.append("━" * 10)
            total = len(rev_df)
            lines.append(f"<b>🔁 反转候选 — 弱势反弹信号（共 {total}，前 {MAX_VISIBLE_REV}）</b>")
            lines.append("")
            for idx, (_, row) in enumerate(rev_df.iterrows()):
                if idx >= MAX_VISIBLE_REV:
                    break
                lines.append(
                    f"<code>{str(row.get('l2_name',''))[:8]:<8} "
                    f"5日:{float(row.get('avg_return_5d',0)):<+7.1f}% "
                    f"强度:{float(row.get('reversal_strength',0)):<+5.1f}%</code>"
                )
            hidden = total - MAX_VISIBLE_REV
            if hidden > 0:
                lines.append(f"<i>…… 还有 {hidden} 个反转候选，详情见完整 HTML 报告</i>")

        lines.append("")

    # ==================== 三、个股挖掘 ====================
    lines.append("━" * 20)
    lines.append("<b>三、个股精选（从强势二级行业中优选）</b>")
    lines.append("━" * 20)

    if stock_result.get("status") == "success":
        stocks = stock_result.get("stocks", [])
        by_industry = stock_result.get("by_industry", {})

        if stocks:
            lines.append(f"从持续性 Top-{len(by_industry)} 个 L2 行业中精选 {len(stocks)} 只个股:")
            lines.append("")

            # 按行业分组展示（行业已按持续性得分降序）
            for ind_name, picks in by_industry.items():
                lines.append(f"<b>▸ {ind_name}</b>")
                for pick in picks:
                    excess_str = _format_pct(pick.get("excess_return", 0))
                    mom20d_str = _format_pct(pick.get("momentum_20d", 0))
                    code_link = _stock_link(pick.get("ts_code", ""), pick.get("ts_code", ""))
                    lines.append(
                        f"  {code_link} "
                        f"{pick['name']:<6} "
                        f"评分:{pick['score']:<4.2f} "
                        f"超额:{excess_str:<6}"
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

    # 只列出非成功模块
    abnormal = [(m, s) for m, s in module_status.items() if s != "success"]
    if abnormal:
        for module, status in abnormal:
            icon = status_icons.get(status, f"❓ {status}")
            lines.append(f"• {module}: {icon}")
    else:
        lines.append("✅ 所有模块正常运行")

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
