"""
回测 HTML 报告生成器
- 纯 CSS 净值曲线图
- 绩效指标卡片
"""

import os

import numpy as np
import pandas as pd

from backtest.backtest_engine import BacktestResult

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#f1f5f9;color:#1e293b;padding:20px;max-width:1000px;margin:0 auto}
h1{font-size:1.4rem;margin-bottom:8px}
.meta{color:#64748b;font-size:.85rem;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}
.card{background:#fff;border-radius:10px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}
.card .label{font-size:.75rem;color:#64748b;text-transform:uppercase;margin-bottom:4px}
.card .value{font-size:1.5rem;font-weight:700}
.green{color:#16a34a}
.red{color:#dc2626}
.amber{color:#d97706}

/* NAV chart — pure CSS */
.chart-container{background:#fff;border-radius:10px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.chart-container h2{font-size:1rem;margin-bottom:16px}
.chart{position:relative;height:300px;border-left:2px solid #e2e8f0;border-bottom:2px solid #e2e8f0;margin:0 0 10px 60px}
.chart-svg{width:100%;height:100%}
.chart-legend{display:flex;gap:20px;font-size:.8rem;margin-top:8px}
.chart-legend span{display:flex;align-items:center;gap:6px}
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}

/* Rankings table */
.ranking-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:20px;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.ranking-table th{background:#f8fafc;padding:10px 12px;text-align:left;font-weight:600;color:#64748b;border-bottom:2px solid #e2e8f0}
.ranking-table td{padding:8px 12px;border-bottom:1px solid #f1f5f9}
.ranking-table tr:hover{background:#f8fafc}
"""


def generate_backtest_report(
    result: BacktestResult,
    all_results: list | None = None,
    output_path: str | None = None,
) -> str:
    """生成回测 HTML 报告。

    Args:
        result: 最优参数的回测结果
        all_results: 网格搜索结果列表
        output_path: 输出文件路径

    Returns:
        HTML 字符串
    """
    # 净值曲线 SVG
    nav_svg = _nav_to_svg(result.nav, result.benchmark_nav)

    # 绩效指标卡片
    cards = [
        ("年化收益", f"{result.annual_return*100:+.1f}%", "green" if result.annual_return > 0 else "red"),
        ("超额收益", f"{result.excess_return*100:+.1f}%", "green" if result.excess_return > 0 else "red"),
        ("夏普比率", f"{result.sharpe_ratio:.2f}", "green" if result.sharpe_ratio > 0.5 else "amber"),
        ("卡玛比率", f"{result.calmar_ratio:.2f}", "green" if result.calmar_ratio > 0.5 else "amber"),
        ("最大回撤", f"{result.max_drawdown*100:.1f}%", "red"),
        ("日胜率", f"{result.win_rate_daily*100:.1f}%", "green" if result.win_rate_daily > 0.5 else "amber"),
        ("月胜率", f"{result.win_rate_monthly*100:.1f}%", "green" if result.win_rate_monthly > 0.5 else "amber"),
        ("盈亏比", f"{result.profit_loss_ratio:.2f}", ""),
    ]

    cards_html = ""
    for label, value, cls in cards:
        cards_html += f'<div class="card"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>\n'

    # 参数信息
    params = result.params
    params_html = ""
    if params:
        params_html = "<p style='font-size:.85rem;color:#64748b;margin-bottom:12px;'>"
        for k, v in params.items():
            if isinstance(v, dict):
                params_html += f"<b>{k}</b>: {v}<br>"
            else:
                params_html += f"<b>{k}</b>: {v} | "
        params_html += "</p>"

    # 网格搜索排名
    grid_html = ""
    if all_results:
        grid_html = """
        <table class="ranking-table">
        <thead><tr><th>#</th><th>夏普</th><th>年化收益</th><th>最大回撤</th><th>Lookback</th><th>TopN</th><th>w_mom</th><th>w_slope</th></tr></thead><tbody>
        """
        for i, r in enumerate(all_results[:20]):
            # Support both dict formats: with 'params'/'weights' or flat keys
            p = r.get("params", r)
            w = r.get("weights", {})
            grid_html += (
                f"<tr><td>{i+1}</td><td><b>{r['sharpe']:.3f}</b></td>"
                f"<td>{r['annual'] if 'annual' in r else r.get('annual_return',0):.1f}%</td>"
                f"<td>{r['mdd'] if 'mdd' in r else r.get('max_drawdown',0):.1f}%</td>"
                f"<td>{p.get('lookback','?')}</td><td>{p.get('top_n','?')}</td>"
                f"<td>{w.get('momentum_score','?')}</td><td>{w.get('return_slope','?')}</td></tr>\n"
            )
        grid_html += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>回测报告 · L1行业轮动策略</title>
<style>{CSS}</style></head>
<body>
<h1>📊 回测报告 · L1行业轮动策略</h1>
<div class="meta">回测区间: {result.nav.index[0].strftime('%Y-%m-%d') if hasattr(result.nav.index[0],'strftime') else str(result.nav.index[0])} ~ {result.nav.index[-1].strftime('%Y-%m-%d') if hasattr(result.nav.index[-1],'strftime') else str(result.nav.index[-1])}</div>
{params_html}

<h2 style="margin-bottom:12px;">绩效概览</h2>
<div class="cards">{cards_html}</div>

<h2 style="margin-bottom:12px;">净值曲线</h2>
<div class="chart-container">
{nav_svg}
<div class="chart-legend">
  <span><span class="legend-dot" style="background:#6366f1"></span> 策略净值</span>
  <span><span class="legend-dot" style="background:#94a3b8"></span> 等权基准</span>
</div>
</div>

<h2 style="margin-bottom:12px;">参数搜索 Top 20</h2>
{grid_html}

<p style="font-size:.78rem;color:#94a3b8;margin-top:24px;">
⚠️ 回测结果不代表未来表现。本报告由 A股趋势交易系统 自动生成。
</p>
</body></html>"""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


def _nav_to_svg(strat_nav: pd.Series, bench_nav: pd.Series, width: int = 800, height: int = 280) -> str:
    """将净值曲线渲染为 SVG polyline。"""
    # 下采样到 ~400 个点
    n = min(len(strat_nav), 400)
    idx = np.linspace(0, len(strat_nav) - 1, n, dtype=int)
    s_nav = strat_nav.iloc[idx].values
    b_nav = bench_nav.iloc[idx].values

    y_min = min(s_nav.min(), b_nav.min()) * 0.95
    y_max = max(s_nav.max(), b_nav.max()) * 1.05
    if y_max - y_min < 0.01:
        y_max = y_min + 0.01

    def scale_y(y):
        return height - (y - y_min) / (y_max - y_min) * height

    def scale_x(i):
        return i / (n - 1) * width

    s_points = " ".join(f"{scale_x(i):.1f},{scale_y(v):.1f}" for i, v in enumerate(s_nav))
    b_points = " ".join(f"{scale_x(i):.1f},{scale_y(v):.1f}" for i, v in enumerate(b_nav))

    return f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg">
      <polyline points="{s_points}" fill="none" stroke="#6366f1" stroke-width="2" vector-effect="non-scaling-stroke"/>
      <polyline points="{b_points}" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,3" vector-effect="non-scaling-stroke"/>
    </svg>
    """
