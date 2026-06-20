"""
个股精选推荐展示模块 (独立预览版)
=====================================
生成 Top-5 个股推荐卡片，包含：
  - 趋势指标：5日/20日动量、MA20偏离、量比
  - 波动与止损：ATR、静态止损 5%、ATR 动态止损 1.5x
  - 个股级概率：基于回测历史按评分邻近匹配
  - 风控提醒：超买/异常放量/极端波动提示

用法（独立运行）:
  python3 analysis/stock_picks.py              # 文本 + HTML 同时输出
  python3 analysis/stock_picks.py --text        # 仅文本
  python3 analysis/stock_picks.py --html        # 仅 HTML
  python3 analysis/stock_picks.py --date 20260616

集成调用:
  from analysis.stock_picks import generate_picks_text, generate_picks_html
  text = generate_picks_text()
  html  = generate_picks_html()
"""

import argparse
import csv
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DB_PATH,
    DATA_DIR,
    COL_TS_CODE,
    COL_TRADE_DATE,
    COL_CLOSE,
    LOG_LEVEL,
)
from analysis.backtest_predictor import (
    load_stock_data,
    get_trading_dates,
    precompute_features_vectorized,
    _score_day,
    TOP_N_DEFAULT,
    run_backtest,
)

logger = logging.getLogger(__name__)

FIXED_STOP = 5.0         # 静态止损 5%
ATR_STOP_MULT = 1.5      # ATR 动态止损乘数
PROB_NEAREST_N = 80      # 概率估算取最近 N 条历史记录


# ============================================================
# 数据加载
# ============================================================

def load_stock_names() -> dict:
    """从 stock_industry_mapping.csv 加载个股名称。"""
    csv_path = os.path.join(DATA_DIR, "stock_industry_mapping.csv")
    names = {}
    if not os.path.exists(csv_path):
        return names
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[row["ts_code"]] = row["stock_name"]
    return names


def _safe_float(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return round(float(val), 2)


# ============================================================
# 个股级概率估算
# ============================================================

def _compute_individual_probability(
    score: float,
    pred_df: pd.DataFrame,
    n: int = PROB_NEAREST_N,
):
    """基于回测历史，按评分欧氏距离匹配最近 N 条记录。

    对每个评分值，在回测历史中找到评分最接近的 N 个历史预测，
    用它们的实际收益分布来估算该评分档位的期望收益和胜率。

    Args:
        score: 目标评分
        pred_df: 回测预测记录 DataFrame（含 score, forward_return）
        n: 取最近多少条

    Returns:
        {"avg_return": 1.06, "prob_gt0": 50.0, "prob_gt10": 26.3,
         "prob_gt20": 12.9, "samples": 80} 或 None
    """
    if pred_df is None or pred_df.empty:
        return None

    df = pred_df.copy()
    df["_dist"] = (df["score"] - score).abs()
    df = df.sort_values("_dist")

    similar = df.head(min(n, len(df)))
    fwd = similar["forward_return"].values

    if len(fwd) < 10:
        return None

    return {
        "avg_return": round(float(np.mean(fwd)), 2),
        "median_return": round(float(np.median(fwd)), 2),
        "prob_gt0": round(float(np.mean(fwd > 0) * 100), 1),
        "prob_gt10": round(float(np.mean(fwd > 10) * 100), 1),
        "prob_gt20": round(float(np.mean(fwd > 20) * 100), 1),
        "samples": len(fwd),
    }


_BACKTEST_PROB_CACHE = None


def _get_probability_cache():
    """获取回测概率缓存（全局缓存，仅计算一次）。"""
    global _BACKTEST_PROB_CACHE
    if _BACKTEST_PROB_CACHE is None:
        report = run_backtest(lookback_days=120, top_n=TOP_N_DEFAULT, verbose=False)
        _BACKTEST_PROB_CACHE = report.get("_raw_predictions")
    return _BACKTEST_PROB_CACHE


# ============================================================
# 风控提醒
# ============================================================

def _get_risk_warnings(pick: dict) -> list[str]:
    """检查个股风险，返回风险提示列表。"""
    warnings_list = []
    ma20 = pick.get("ma20_deviation")
    mom5 = pick.get("momentum_5d")
    vr = pick.get("volume_ratio")
    atr = pick.get("atr_ratio")

    if ma20 is not None and ma20 > 50:
        warnings_list.append(("🛑 严重超买", f"MA20偏离 {ma20:+.1f}%，追高风险极大"))
    elif ma20 is not None and ma20 > 30:
        warnings_list.append(("⚠️ 超买", f"MA20偏离 {ma20:+.1f}%，短线回调风险较高"))

    if mom5 is not None and mom5 > 50:
        warnings_list.append(("⚠️ 短期暴涨", f"5日涨幅 {mom5:+.1f}%，获利盘巨大"))

    if vr is not None and vr < 0.3:
        warnings_list.append(("⚠️ 严重缩量", f"量比 {vr:.2f}，上涨动能不足"))

    if atr is not None and atr > 10:
        warnings_list.append(("⚠️ 极端波动", f"日波幅(ATR) {atr:.1f}%，止损需加宽"))

    return warnings_list


# ============================================================
# 获取推荐数据
# ============================================================

def get_picks_data(
    db_path: str = DB_PATH,
    date: str = "latest",
    top_n: int = 5,
) -> list[dict]:
    """获取个股推荐数据（含所有指标、止损价、概率、风控）。

    Returns:
        list of dict, each containing:
          ts_code, name, score, close,
          momentum_5d, momentum_20d, ma20_deviation, volume_ratio,
          atr_ratio, volatility_20d,
          static_stop_pct, static_stop_price,
          atr_stop_pct, atr_stop_price,
          probability (dict), warnings (list)
    """
    df = load_stock_data(db_path)
    all_dates = get_trading_dates(df)

    if date == "latest":
        date = all_dates[-1]

    df = precompute_features_vectorized(df)
    df_today = df[df[COL_TRADE_DATE] == date].copy()

    if df_today.empty:
        logger.warning("日期 %s 无数据", date)
        return []

    candidates = _score_day(df_today, top_n)
    if candidates.empty:
        return []

    stock_names = load_stock_names()

    # 加载概率缓存
    prob_df = _get_probability_cache()

    results = []
    for _, row in candidates.iterrows():
        ts_code = row[COL_TS_CODE]
        close_price = float(row[COL_CLOSE])

        atr_ratio = row.get("atr_ratio")
        if pd.isna(atr_ratio):
            atr_ratio = None

        score = round(float(row["score"]), 1)

        # 止损
        static_stop_price = round(close_price * (1 - FIXED_STOP / 100), 2)
        if atr_ratio is not None:
            atr_stop_pct = round(ATR_STOP_MULT * atr_ratio, 2)
            atr_stop_price = round(close_price * (1 - atr_stop_pct / 100), 2)
        else:
            atr_stop_pct = None
            atr_stop_price = None

        pick = {
            "ts_code": ts_code,
            "name": stock_names.get(ts_code, ts_code),
            "score": score,
            "close": close_price,
            "momentum_5d": _safe_float(row.get("momentum_5d")),
            "momentum_20d": _safe_float(row.get("momentum_20d")),
            "ma20_deviation": _safe_float(row.get("ma20_deviation")),
            "volume_ratio": _safe_float(row.get("volume_ratio")),
            "atr_ratio": _safe_float(row.get("atr_ratio")),
            "volatility_20d": _safe_float(row.get("volatility_20d")),
            "static_stop_pct": FIXED_STOP,
            "static_stop_price": static_stop_price,
            "atr_stop_pct": atr_stop_pct,
            "atr_stop_price": atr_stop_price,
            "probability": _compute_individual_probability(score, prob_df) if prob_df is not None else None,
            "warnings": [],
        }
        pick["warnings"] = _get_risk_warnings(pick)
        results.append(pick)

    return results


# ============================================================
# 文本输出 (Telegram / Console)
# ============================================================

def generate_picks_text(
    db_path: str = DB_PATH,
    date: str = "latest",
    top_n: int = 5,
) -> str:
    """生成个股推荐纯文本。"""
    picks = get_picks_data(db_path, date, top_n)
    if not picks:
        return "⚠️ 个股推荐暂无数据"

    dt = date if date != "latest" else "最新交易日"
    lines = []
    lines.append("━" * 48)
    lines.append("  📌 A股趋势交易系统 · 个股精选 TOP{}".format(len(picks)))
    lines.append("  基准日: {}  |  概率基于 120 日回测".format(dt))
    lines.append("━" * 48)

    emoji_map = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, p in enumerate(picks, 1):
        emoji = emoji_map.get(i, f"{i}.")
        lines.append("")
        lines.append(f"  {emoji} {p['name']} ({p['ts_code']})  ·  评分 {p['score']}")

        # 风险警告（如果有，放最前面）
        for w_type, w_msg in p.get("warnings", []):
            lines.append(f"     {w_type}: {w_msg}")

        lines.append(f"  {'─'*44}")
        mom5 = _fmt_pct(p["momentum_5d"])
        mom20 = _fmt_pct(p["momentum_20d"])
        ma20 = _fmt_pct(p["ma20_deviation"])
        vr = _fmt_num(p["volume_ratio"], 2)
        lines.append(f"  📈 趋势:  5日 {mom5}  |  20日 {mom20}  |  MA20偏离 {ma20}")
        lines.append(f"  📊 量价:  量比 {vr}  |  现价 {p['close']:.2f}")

        atr = _fmt_pct(p["atr_ratio"])
        lines.append(f"  🌊 波动:  ATR {atr}  |  静态止损 -{FIXED_STOP:.0f}%  |  ATR×{ATR_STOP_MULT} 自适应")

        ss = f"{p['static_stop_price']:.2f}"
        if p["atr_stop_price"] is not None:
            ds = f"{p['atr_stop_price']:.2f}"
            lines.append(f"  🛑 止损:  静态 {p['static_stop_pct']:.0f}% ({ss})  │  ATR×{ATR_STOP_MULT} {p['atr_stop_pct']:.1f}% ({ds})")
        else:
            lines.append(f"  🛑 止损:  静态 {p['static_stop_pct']:.0f}% ({ss})  │  ATR N/A")

        # 个股概率
        prob = p.get("probability")
        if prob:
            lines.append(f"  🎯 期望:  预期 {prob['avg_return']:+.2f}%  |  >0%: {prob['prob_gt0']:.0f}%"
                         f"  |  >10%: {prob['prob_gt10']:.0f}%  |  >20%: {prob['prob_gt20']:.0f}%"
                         f"  (n={prob['samples']})")

    lines.append("")
    lines.append("━" * 48)
    lines.append("  💡 止损说明")
    lines.append(f"     静态止损: 固定 {FIXED_STOP:.0f}%，适合大盘平稳时")
    lines.append(f"     ATR止损:  {ATR_STOP_MULT}×ATR，根据波动自适应，适合震荡市")
    lines.append("     实盘建议: 结合开盘价判断，跳空低开时手动离场")
    lines.append("━" * 48)

    return "\n".join(lines)


# ============================================================
# HTML 输出
# ============================================================

def generate_picks_html(
    db_path: str = DB_PATH,
    date: str = "latest",
    top_n: int = 5,
) -> str:
    """生成个股推荐的 HTML 区块（可嵌入主报告）。

    样式与主报告兼容，使用 inline CSS。
    """
    picks = get_picks_data(db_path, date, top_n)
    if not picks:
        return "<!-- 个股推荐暂无数据 -->"

    dt = date if date != "latest" else "最新交易日"

    def _score_color(s: float) -> str:
        if s >= 70: return "#8b5cf6"
        if s >= 55: return "#22c55e"
        if s >= 40: return "#f59e0b"
        return "#ef4444"

    def _pct_style(val, good_positive=True):
        if val is None:
            return '<span style="color:#94a3b8;">N/A</span>'
        if good_positive:
            color = "#ef4444" if val >= 0 else "#22c55e"
        else:
            color = "#22c55e" if val >= 0 else "#ef4444"
        return f'<span style="color:{color};font-weight:600;">{val:+.2f}%</span>'

    def _risk_badge(w_type: str) -> str:
        if "严重" in w_type or "极端" in w_type:
            return f'<span style="display:inline-block;background:#fef2f2;color:#dc2626;border:1px solid #fecaca;border-radius:4px;padding:2px 8px;font-size:.72rem;font-weight:600;">{w_type}</span>'
        return f'<span style="display:inline-block;background:#fffbeb;color:#d97706;border:1px solid #fde68a;border-radius:4px;padding:2px 8px;font-size:.72rem;font-weight:600;">{w_type}</span>'

    cards_html = ""
    emoji_map = {1: "🥇", 2: "🥈", 3: "🥉"}

    for i, p in enumerate(picks, 1):
        emoji = emoji_map.get(i, f"{i}.")
        score_color = _score_color(p["score"])

        # 风控标签
        risk_badges = ""
        for w_type, _ in p.get("warnings", []):
            risk_badges += _risk_badge(w_type) + " "

        # 概率
        prob_html = ""
        prob = p.get("probability")
        if prob:
            prob_html = f'''
            <div style="margin-top:10px;padding:10px;background:#f8fafc;border-radius:6px;">
              <span style="font-size:.82rem;color:#64748b;">🎯 预期收益 <b style="color:#1e293b;">{prob["avg_return"]:+.2f}%</b>
              · 胜率>0% <b>{prob["prob_gt0"]:.0f}%</b>
              · >10% <b>{prob["prob_gt10"]:.0f}%</b>
              · >20% <b>{prob["prob_gt20"]:.0f}%</b>
              · n={prob["samples"]}</span>
            </div>'''

        # 止损
        ss = f"{p['static_stop_price']:.2f}"
        if p["atr_stop_price"] is not None:
            ds = f"{p['atr_stop_price']:.2f}"
            stop_html = f'静态 {p["static_stop_pct"]:.0f}% (<b>{ss}</b>) &nbsp;│&nbsp; ATR×{ATR_STOP_MULT} {p["atr_stop_pct"]:.1f}% (<b>{ds}</b>)'
        else:
            stop_html = f'静态 {p["static_stop_pct"]:.0f}% (<b>{ss}</b>) &nbsp;│&nbsp; ATR N/A'

        cards_html += f'''
        <div style="background:white;border-radius:10px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:14px;">
          <div style="padding:14px 18px 10px;border-bottom:2px solid {score_color};display:flex;justify-content:space-between;align-items:center;">
            <div>
              <span style="font-size:.95rem;margin-right:6px;">{emoji}</span>
              <span style="font-weight:700;font-size:1rem;">{p['name']}</span>
              <span style="color:#94a3b8;font-size:.85rem;">{p['ts_code']}</span>
            </div>
            <span style="background:{score_color};color:white;padding:4px 12px;border-radius:12px;font-weight:700;font-size:.85rem;">{p['score']}</span>
          </div>
          <div style="padding:12px 18px;">
            {risk_badges}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;font-size:.85rem;">
              <div>📈 5日动量: {_pct_style(p["momentum_5d"])}</div>
              <div>📈 20日动量: {_pct_style(p["momentum_20d"])}</div>
              <div>📏 MA20偏离: {_pct_style(p["ma20_deviation"])}</div>
              <div>📊 量比: <b>{_fmt_num(p["volume_ratio"], 2)}</b></div>
              <div>🌊 ATR: {_pct_style(p["atr_ratio"])}</div>
              <div>💵 现价: <b>{p['close']:.2f}</b></div>
            </div>
            <div style="margin-top:8px;padding:8px 12px;background:#fff7ed;border-left:3px solid #f97316;border-radius:4px;font-size:.82rem;">
              🛑 止损: {stop_html}
            </div>
            {prob_html}
          </div>
        </div>'''

    return f'''
    <div style="text-align:center;padding:12px;margin:16px 0 8px;font-size:1.1rem;font-weight:700;color:#8b5cf6;border-top:2px solid #8b5cf6;border-bottom:2px solid #8b5cf6;">六、个股精选 TOP{top_n}  <span style="font-size:.75rem;font-weight:400;color:#94a3b8;">(数据: {dt} · 概率基于120日回测)</span></div>
    <div class="section">
      <div class="section-title">🎯 个股推荐（含止损与概率）</div>
      <p style="font-size:0.85rem;color:#64748b;margin-bottom:12px;">
        基于多因子评分排序，历史概率通过 120 日回测按评分邻近匹配估算。
        静态止损 {FIXED_STOP:.0f}%，ATR 动态止损 {ATR_STOP_MULT}x。
      </p>
      {cards_html}
    </div>'''


# ============================================================
# 格式化工具
# ============================================================

def _fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val:+.2f}%" if val >= 0 else f"{val:.2f}%"


def _fmt_num(val, decimals: int = 2):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


# ============================================================
# 独立运行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="个股精选推荐展示")
    parser.add_argument("--date", type=str, default="latest", help="基准日期 YYYYMMDD")
    parser.add_argument("--top", type=int, default=5, help="推荐数量")
    parser.add_argument("--text", action="store_true", help="仅输出文本")
    parser.add_argument("--html", action="store_true", help="仅输出 HTML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.html:
        print(generate_picks_html(date=args.date, top_n=args.top))
        return

    if args.text:
        print(generate_picks_text(date=args.date, top_n=args.top))
        return

    # 默认：文本 + HTML
    print(generate_picks_text(date=args.date, top_n=args.top))
    print()
    print("=" * 48)
    print("  HTML 输出预览")
    print("=" * 48)
    print(generate_picks_html(date=args.date, top_n=args.top))


if __name__ == "__main__":
    main()
