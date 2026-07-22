#!/usr/bin/env python3
"""
三段式风险预警系统 — 5年历史回测
=================================
信号体系:
  🚩 Phase 1 (广度崩塌): 近5天≥3天超过70%行业下跌
  🔴 Phase 2 (趋势破坏): 指数收盘跌破MA20
  💀 Phase 3 (死猫反弹): 指数涨 >60%行业跌 + 缩量

目标事件:
  A. 短期暴跌：5日内最大回撤 ≥ -3%
  B. 极端事件：行业90%下跌 + 指数跌≥2.5%
  C. 连续下跌：10日累计跌幅 ≥ -5%

评估指标:
  - 各阶段对事件的预警覆盖率和提前天数
  - 综合信号的精确率 / 召回率 / F1
  - 误报率
"""
import os, sys, json
from datetime import datetime
import sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# ── 回测参数 ──
LOOKAHEAD_SCOPE = 15  # 信号后多少天内有效
STUDY_START = "20200101"
STUDY_END = "20260722"


def load_data():
    import tushare as ts
    from config import TUSHARE_TOKEN
    pro = ts.pro_api(TUSHARE_TOKEN)

    print("加载指数数据...")
    idx = pro.index_daily(ts_code='000001.SH', start_date='20190101', end_date=STUDY_END)
    idx['trade_date'] = idx['trade_date'].astype(str)
    idx = idx.sort_values('trade_date').reset_index(drop=True)
    for c in ['close','pct_chg','vol']:
        idx[c] = pd.to_numeric(idx[c], errors='coerce')

    # 技术指标
    idx['ma20'] = idx['close'].rolling(20).mean()
    idx['below_ma20'] = idx['close'] < idx['ma20']
    idx['high_20'] = idx['close'].rolling(20).max()
    idx['dist_high'] = (idx['close'] - idx['high_20']) / idx['high_20']
    idx['vol_ma20'] = idx['vol'].rolling(20).mean()
    idx['vol_below_avg'] = idx['vol'] < idx['vol_ma20']
    idx['ret_5d'] = idx['close'].pct_change(5) * 100
    idx['ret_10d'] = idx['close'].pct_change(10) * 100

    print("加载行业数据...")
    conn = sqlite3.connect(DB_PATH)
    sector = pd.read_sql_query(
        f"SELECT trade_date, ts_code, close FROM sw_l2_index_daily "
        f"WHERE trade_date >= '20190101' ORDER BY trade_date", conn
    )
    sector['trade_date'] = sector['trade_date'].astype(str)
    sector['close'] = pd.to_numeric(sector['close'], errors='coerce')
    sector['prev_close'] = sector.groupby('ts_code')['close'].shift(1)
    sector['ret'] = (sector['close'] - sector['prev_close']) / sector['prev_close']

    daily = sector.groupby('trade_date').agg(
        down_pct=('ret', lambda x: (x<0).sum()/len(x)*100),
    ).reset_index()
    conn.close()

    df = daily.merge(idx[['trade_date','close','pct_chg','below_ma20','vol_below_avg','dist_high','ret_5d','ret_10d']],
                     on='trade_date', how='inner')
    df = df[df['trade_date'] >= STUDY_START].reset_index(drop=True)
    return df


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """计算三段式信号"""
    n = len(df)
    df['heavy_day'] = df['down_pct'] > 70
    df['broad_collapse_3of5'] = df['heavy_day'].rolling(5).sum() >= 3

    # Phase 1: 广度崩塌（首次满足3/5条件）
    df['phase1'] = False
    for i in range(5, n):
        if df['broad_collapse_3of5'].iloc[i] and not df['broad_collapse_3of5'].iloc[i-1]:
            df.loc[df.index[i], 'phase1'] = True

    # Phase 2: 首次跌破MA20
    df['phase2'] = False
    for i in range(20, n):
        if df['below_ma20'].iloc[i] and not df['below_ma20'].iloc[i-1]:
            df.loc[df.index[i], 'phase2'] = True

    # Phase 3: 死猫反弹（指数涨 + >60%行业跌 + 缩量）
    df['phase3'] = (df['pct_chg'] > 0) & (df['down_pct'] > 60) & (df['vol_below_avg'])

    # 综合风险等级
    df['alert_level'] = 'normal'
    df.loc[df['phase1'], 'alert_level'] = 'caution'
    df.loc[df['phase2'], 'alert_level'] = 'warning'
    df.loc[df['phase3'], 'alert_level'] = 'danger'

    return df


def evaluate_events(df: pd.DataFrame):
    """评估三段式信号对三类目标事件的预警效果"""
    n = len(df)

    # 定义目标事件
    events = {
        '短期暴跌 (>-3% in 5d)': lambda fd: fd.ret_5d.min() <= -3,
        '极端事件 (90%+行业跌 + 指数>-2.5%)': lambda fd: ((fd.down_pct >= 90) & (fd.pct_chg <= -2.5)).any(),
        '持续下跌 (>-5% in 10d)': lambda fd: fd.ret_10d.min() <= -5,
        '90%+行业暴跌日': lambda fd: (fd.down_pct >= 90).any(),
    }

    print(f"\n{'='*85}")
    print("  全局统计")
    print(f"{'='*85}")
    print(f"  交易日: {len(df)} 天 ({df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]})")

    print(f"\n  各阶段触发次数:")
    print(f"    🚩 Phase 1 (广度崩塌): {df['phase1'].sum()} 次")
    print(f"    🔴 Phase 2 (趋势破坏): {df['phase2'].sum()} 次")
    print(f"    💀 Phase 3 (死猫反弹): {df['phase3'].sum()} 次")

    print(f"\n  目标事件发生次数:")
    for ename, efunc in events.items():
        cnt = 0
        for i in range(n):
            future = df.iloc[i+1:i+LOOKAHEAD_SCOPE]
            if len(future) >= 3 and efunc(future):
                cnt += 1
        print(f"    {ename}: {cnt} 次")

    # 评估每种信号对每种目标事件
    print(f"\n{'='*85}")
    print("  信号预警评估 (信号后15天内事件命中率)")
    print(f"{'='*85}")

    all_results = []
    for sig_name, sig_col in [('🚩 Phase1 广度崩塌', 'phase1'),
                               ('🔴 Phase2 趋势破坏', 'phase2'),
                               ('💀 Phase3 死猫反弹', 'phase3'),
                               ('🚩+🔴 Phase1→2', 'phase1_and_2'),
                               ('🚩+🔴+💀 全阶段', 'all_three')]:

        if sig_name == '🚩+🔴 Phase1→2':
            df['phase1_and_2'] = df['phase1'] | df['phase2']
        elif sig_name == '🚩+🔴+💀 全阶段':
            df['all_three'] = df['phase1'] | df['phase2'] | df['phase3']

        sig_dates = df[df[sig_col]]['trade_date'].tolist()
        print(f"\n  {sig_name} ({len(sig_dates)}次触发):")

        for ename, efunc in events.items():
            true_pos = 0
            false_pos = 0
            lead_times = []

            for _, row in df[df[sig_col]].iterrows():
                idx = row.name
                future = df.iloc[idx+1:idx+LOOKAHEAD_SCOPE]
                if len(future) < 3:
                    continue
                if efunc(future):
                    true_pos += 1
                    # 首次事件日期
                    for j in range(len(future)):
                        if efunc(future.iloc[j:j+1]):
                            lead_times.append(j + 1)
                            break
                else:
                    false_pos += 1

            total = true_pos + false_pos
            precision = true_pos / total if total > 0 else 0
            avg_lead = np.mean(lead_times) if lead_times else 0

            print(f"    {ename:<30} 命中{true_pos:>2}/{total:<3} "
                  f"精确率{precision:>7.1%} 提前{avg_lead:>4.1f}天")

    return df


def find_optimal_thresholds(df: pd.DataFrame):
    """扫描Phase 1和Phase 3的阈值参数，寻找最优组合"""
    print(f"\n{'='*85}")
    print("  阈值优化扫描")
    print(f"{'='*85}")

    # 测试不同的down_pct和day_count组合
    params = []
    for down_thresh in [60, 65, 70, 75]:
        for days_req in [2, 3, 4]:
            col_name = f'custom_p1_{down_thresh}_{days_req}'
            df['heavy_custom'] = df['down_pct'] > down_thresh
            df[col_name] = df['heavy_custom'].rolling(5).sum() >= days_req

            # 首次触发
            first_triggers = 0
            for i in range(5, len(df)):
                if df[col_name].iloc[i] and not df[col_name].iloc[i-1]:
                    first_triggers += 1

            # 评估对未来极端事件(短期暴跌)的预警能力
            tp = 0
            total_triggers = 0
            for i in range(5, len(df)):
                if df[col_name].iloc[i] and not df[col_name].iloc[i-1]:
                    future = df.iloc[i+1:i+LOOKAHEAD_SCOPE]
                    if len(future) >= 3:
                        total_triggers += 1
                        if future['ret_5d'].min() <= -3:
                            tp += 1

            prec = tp / total_triggers if total_triggers > 0 else 0
            params.append({
                'type': 'P1',
                'down_thresh': down_thresh,
                'days_req': days_req,
                'triggers': first_triggers,
                'evaluated': total_triggers,
                'precision': prec,
                'score': prec * (1 - total_triggers/len(df)*50)  # 平衡精确率和触发率
            })

    # Phase 3 阈值优化
    for down_thresh in [55, 60, 65, 70]:
        for vol_cond in ['below_avg', 'any']:
            col_name = f'custom_p3_{down_thresh}_{vol_cond}'
            if vol_cond == 'below_avg':
                df[col_name] = (df['pct_chg'] > 0) & (df['down_pct'] > down_thresh) & (df['vol_below_avg'])
            else:
                df[col_name] = (df['pct_chg'] > 0) & (df['down_pct'] > down_thresh)

            triggers = df[col_name].sum()

            tp = 0
            total_triggers = 0
            for i in range(len(df)):
                if df[col_name].iloc[i]:
                    future = df.iloc[i+1:i+LOOKAHEAD_SCOPE]
                    if len(future) >= 3:
                        total_triggers += 1
                        if future['ret_5d'].min() <= -3:
                            tp += 1

            prec = tp / total_triggers if total_triggers > 0 else 0
            params.append({
                'type': 'P3',
                'down_thresh': down_thresh,
                'vol_cond': vol_cond,
                'triggers': triggers,
                'evaluated': total_triggers,
                'precision': prec,
                'score': prec * (1 - total_triggers/len(df)*30)
            })

    # Top 10
    params.sort(key=lambda x: -x['score'])
    print(f"\n  Phase 1 最佳参数组合 (按综合评分):")
    print(f"  {'行业跌>%':>8} {'天数要求':>8} {'触发次数':>8} {'精确率':>8} {'评分':>8}")
    for p in [p for p in params[:10] if p['type'] == 'P1']:
        print(f"  {p['down_thresh']:>7}% {p['days_req']:>7}天 {p['triggers']:>8} {p['precision']:>7.1%} {p['score']:>7.3f}")

    print(f"\n  Phase 3 最佳参数组合:")
    print(f"  {'行业跌>%':>8} {'缩量条件':>8} {'触发次数':>8} {'精确率':>8} {'评分':>8}")
    for p in [p for p in params[:10] if p['type'] == 'P3']:
        print(f"  {p['down_thresh']:>7}% {p['vol_cond']:>8} {p['triggers']:>8} {p['precision']:>7.1%} {p['score']:>7.3f}")


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def main():
    print("=" * 85)
    print("  三段式风险预警系统 — 5年历史回测")
    print(f"  信号窗口: {LOOKAHEAD_SCOPE}天")
    print(f"  回测区间: {STUDY_START} ~ {STUDY_END}")
    print("=" * 85)

    df = load_data()
    df = compute_signals(df)
    df = evaluate_events(df)
    find_optimal_thresholds(df)

    # 年度明细
    print(f"\n{'='*85}")
    print("  年度信号明细")
    print(f"{'='*85}")
    print(f"{'年份':>6} {'P1':>5} {'P2':>5} {'P3':>5} {'暴跌>3%':>8} {'极端事件':>8} {'持续跌>5%':>8}")
    for yr in range(2020, 2027):
        yr_df = df[df['trade_date'].str.startswith(str(yr))]
        if yr_df.empty:
            continue
        p1 = yr_df['phase1'].sum()
        p2 = yr_df['phase2'].sum()
        p3 = yr_df['phase3'].sum()

        n = len(yr_df)
        crash_events = 0
        extreme_events = 0
        sustained_events = 0
        for i in range(n):
            future = yr_df.iloc[i+1:i+LOOKAHEAD_SCOPE]
            if len(future) < 3: continue
            if future['ret_5d'].min() <= -3: crash_events += 1
            if ((future['down_pct'] >= 90) & (future['pct_chg'] <= -2.5)).any(): extreme_events += 1
            if future['ret_10d'].min() <= -5: sustained_events += 1

        print(f"  {yr:>4}年 {p1:>4} {p2:>4} {p3:>4} {crash_events:>7} {extreme_events:>7} {sustained_events:>7}")

    # 输出2026年关键日期明细
    print(f"\n{'='*85}")
    print("  2026年信号触发明细")
    print(f"{'='*85}")
    yr26 = df[df['trade_date'].str.startswith('2026')]
    print(f"{'日期':>10} {'指数':>7} {'行业跌%':>7} {'P1':>5} {'P2':>5} {'P3':>5} {'后续5日最低%':>12}")
    for _, r in yr26.iterrows():
        idx_pos = r.name
        future = df.iloc[idx_pos+1:idx_pos+6]
        fut_min = future['ret_5d'].min() if len(future) >= 3 else 0

        # 只在触发信号或大跌时显示
        if r['phase1'] or r['phase2'] or r['phase3'] or r['pct_chg'] <= -2:
            p1m = '🚩' if r['phase1'] else ''
            p2m = '🔴' if r['phase2'] else ''
            p3m = '💀' if r['phase3'] else ''
            emoji = '⚠️' if fut_min <= -5 else ('🔻' if fut_min <= -3 else '')
            print(f"  {emoji} {r['trade_date']} {r['close']:>7.1f} {r['down_pct']:>6.1f}% "
                  f"{p1m:>3} {p2m:>3} {p3m:>3} {fut_min:>10.1f}%")

    print(f"\n{'='*85}")
    print("  ✅ 回测完成")
    print(f"{'='*85}")


if __name__ == '__main__':
    main()
