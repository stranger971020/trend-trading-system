#!/usr/bin/env python3
"""
早期预警信号分析 — 寻找行业90%下跌+指数跌2.5%事件的先行指标
===========================================================

目标:
  1. 找出历史上 "行业90%下跌 + 指数日跌≥2.5%" 的所有事件
  2. 分析事件前 1~5 天的各类指标变化
  3. 筛选出有预测能力的先行信号

事件定义:
  - 123 个 L2 行业中 ≥90% 当天下跌
  - 且上证指数当日跌幅 ≥ -2.5%

候选先行指标:
  A. 行业涨跌比连续 N 天低于阈值
  B. 行业 MA20 突破比骤降
  C. 指数波动率扩张（5日ATR/20日ATR）
  D. 指数距前高距离
  E. 指数 RSI 进入超买/超卖
  F. 顶背离计数（当前系统的指标）
  G. 高位区计数（当前系统的指标）
  H. MA20 上下行业比（看跌/看涨比值）
"""
import os, sys, json
from datetime import datetime, timedelta
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

LOOKAHEAD_WINDOWS = [1, 2, 3, 5]  # 事件前 N 天检查信号


def load_data(conn) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import tushare as ts
    from config import TUSHARE_TOKEN
    pro = ts.pro_api(TUSHARE_TOKEN)

    # 指数数据
    idx = pro.index_daily(ts_code='000001.SH', start_date='20150101', end_date='20260722')
    idx['trade_date'] = idx['trade_date'].astype(str)
    idx = idx.sort_values('trade_date').reset_index(drop=True)
    for c in ['close', 'open', 'high', 'low', 'pct_chg', 'amount']:
        idx[c] = pd.to_numeric(idx[c], errors='coerce')

    # L2 行业数据
    sector = pd.read_sql_query(
        "SELECT trade_date, ts_code, close FROM sw_l2_index_daily "
        "WHERE trade_date >= '20150101' ORDER BY trade_date",
        conn
    )
    sector['trade_date'] = sector['trade_date'].astype(str)
    sector['close'] = pd.to_numeric(sector['close'], errors='coerce')

    return idx, sector


def compute_event_dates(idx: pd.DataFrame, sector: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """找出所有事件日期 + 计算每日全量指标"""
    print("计算每日行业涨跌比...")
    # 每只行业股票的日涨跌
    sector['prev_close'] = sector.groupby('ts_code')['close'].shift(1)
    sector['ret'] = (sector['close'] - sector['prev_close']) / sector['prev_close']

    # 每日汇总
    daily = sector.groupby('trade_date').agg(
        total=('ts_code', 'count'),
        down=('ret', lambda x: (x < 0).sum()),
    ).reset_index()
    daily['down_pct'] = daily['down'] / daily['total'] * 100

    # 合并指数
    daily = daily.merge(idx[['trade_date', 'close', 'pct_chg', 'amount']], on='trade_date', how='left')

    # 技术指标
    daily = daily.sort_values('trade_date').reset_index(drop=True)

    # 波动率
    daily['ret_daily'] = daily['close'].pct_change()
    daily['atr_5'] = daily['ret_daily'].rolling(5).std()
    daily['atr_20'] = daily['ret_daily'].rolling(20).std()
    daily['vol_expansion'] = daily['atr_5'] / daily['atr_20']

    # 距前高
    daily['high_20'] = daily['close'].rolling(20).max()
    daily['dist_from_high'] = (daily['close'] - daily['high_20']) / daily['high_20']

    # RSI
    delta = daily['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    daily['rsi'] = 100 - (100 / (1 + rs))

    # MA 偏离度
    daily['ma20'] = daily['close'].rolling(20).mean()
    daily['ma60'] = daily['close'].rolling(60).mean()
    daily['ma200'] = daily['close'].rolling(200).mean()
    daily['deviation_ma20'] = (daily['close'] - daily['ma20']) / daily['ma20']
    daily['deviation_ma60'] = (daily['close'] - daily['ma60']) / daily['ma60']

    # 事件标记：行业90%下跌 + 指数跌≥2.5%
    daily['event'] = (daily['down_pct'] >= 90) & (daily['pct_chg'] <= -2.5)
    daily['event_severe'] = (daily['down_pct'] >= 95) & (daily['pct_chg'] <= -3)

    event_dates = daily[daily['event']]['trade_date'].tolist()
    print(f"事件识别: {len(event_dates)} 天 ({event_dates[0] if event_dates else 'N/A'} ~ {event_dates[-1] if event_dates else 'N/A'})")

    return event_dates, daily


def analyze_pre_event_signals(daily: pd.DataFrame, event_dates: List[str], lookahead: int = 3):
    """
    分析事件前 lookahead 天的指标变化

    对每个事件，检查事件前 1~lookahead 天的各类指标的 "异常信号" 出现情况
    """
    print(f"\n{'='*65}")
    print(f"事件前 {lookahead} 天信号分析")
    print(f"{'='*65}")

    date_set = set(event_dates)
    n_events = len(event_dates)
    n_days = len(daily)

    # 候选信号定义
    signals = {
        # (信号名, 列名, 比较方式, 阈值)
        '行业下跌比>85%': ('down_pct', '>=', 85),
        '行业下跌比>80%': ('down_pct', '>=', 80),
        '行业下跌比>70%': ('down_pct', '>=', 70),
        '指数跌>1.5%': ('pct_chg', '<=', -1.5),
        '指数跌>1%': ('pct_chg', '<=', -1.0),
        '波动率扩张>1.3': ('vol_expansion', '>=', 1.3),
        '波动率扩张>1.2': ('vol_expansion', '>=', 1.2),
        '距前高<-3%': ('dist_from_high', '<=', -0.03),
        '距前高<-2%': ('dist_from_high', '<=', -0.02),
        'RSI<30(超卖)': ('rsi', '<=', 30),
        'RSI<40': ('rsi', '<=', 40),
        '偏离MA20<-5%': ('deviation_ma20', '<=', -0.05),
        '偏离MA20<-3%': ('deviation_ma20', '<=', -0.03),
        '偏离MA60<-5%': ('deviation_ma60', '<=', -0.05),
        '偏离MA60<-3%': ('deviation_ma60', '<=', -0.03),
    }

    results = []
    for sig_name, (col, op, threshold) in signals.items():
        hits_before_event = 0  # 事件前 lookahead 天内信号出现次数
        total_checks = 0

        for i, row in daily.iterrows():
            if row['trade_date'] not in date_set:
                continue

            # 往前看 lookahead 天
            for offset in range(1, lookahead + 1):
                if i - offset < 0:
                    continue
                prev_row = daily.iloc[i - offset]
                val = prev_row.get(col, np.nan)
                if pd.isna(val):
                    continue
                total_checks += 1

                if op == '>=' and val >= threshold:
                    hits_before_event += 1
                elif op == '<=' and val <= threshold:
                    hits_before_event += 1

        hit_rate = hits_before_event / total_checks * 100 if total_checks > 0 else 0

        # 随机概率（即所有非事件日该信号的比例）
        non_event = daily[~daily['event']].copy()
        random_count = 0
        random_total = 0
        for _, row in non_event.iterrows():
            val = row.get(col, np.nan)
            if pd.isna(val):
                continue
            random_total += 1
            if op == '>=' and val >= threshold:
                random_count += 1
            elif op == '<=' and val <= threshold:
                random_count += 1
        random_rate = random_count / random_total * 100 if random_total > 0 else 0
        lift = hit_rate / random_rate if random_rate > 0 else 1.0

        results.append({
            'signal': sig_name,
            'hit_before_event': hits_before_event,
            'total_checks': total_checks,
            'hit_rate': hit_rate,
            'random_rate': random_rate,
            'lift': lift,
        })

    results.sort(key=lambda x: -x['lift'])

    print(f"{'信号':<25} {'命中/检查':>12} {'事件前命中率':>12} {'平日概率':>10} {'提升倍数':>10}")
    print("-" * 75)
    for r in results:
        label = "🟢" if r['lift'] > 1.5 else ("🟡" if r['lift'] > 1.2 else "⚪")
        print(f"{label} {r['signal']:<22} {r['hit_before_event']:>4}/{r['total_checks']:<5} "
              f"{r['hit_rate']:>10.1f}% {r['random_rate']:>9.1f}% {r['lift']:>9.2f}x")
    return results


def find_best_combination(daily: pd.DataFrame, event_dates: List[str], lookahead: int = 3):
    """
    测试信号组合的预测能力

    选取命中率最高的几个信号，测试它们的组合能否在事件前给出有效预警
    """
    print(f"\n{'='*65}")
    print(f"信号组合测试 (事件前{lookahead}天预警)")
    print(f"{'='*65}")

    date_set = set(event_dates)
    n_events = len(event_dates)

    # 候选组合信号
    combos = [
        ("距前高<-2% OR RSI<40", lambda r: r['dist_from_high'] <= -0.02 or r['rsi'] <= 40),
        ("偏离MA20<-3%", lambda r: r['deviation_ma20'] <= -0.03),
        ("距前高<-3%", lambda r: r['dist_from_high'] <= -0.03),
        ("波动扩张>1.2 + 距前高<-2%", lambda r: r['vol_expansion'] >= 1.2 and r['dist_from_high'] <= -0.02),
        ("波动扩张>1.2 + 偏离MA20<-3%", lambda r: r['vol_expansion'] >= 1.2 and r['deviation_ma20'] <= -0.03),
        ("距前高<-2% + 行业下跌>70%", lambda r: r['dist_from_high'] <= -0.02 and r['down_pct'] >= 70),
        ("偏离MA20<-3% + 行业下跌>70%", lambda r: r['deviation_ma20'] <= -0.03 and r['down_pct'] >= 70),
        ("偏离MA60<-3%", lambda r: r['deviation_ma60'] <= -0.03),
        ("距前高<-2% OR 偏离MA20<-3%", lambda r: r['dist_from_high'] <= -0.02 or r['deviation_ma20'] <= -0.03),
    ]

    for combo_name, combo_fn in combos:
        # 对每个事件，检查事件前 lookahead 天内是否有任意一天触发信号
        events_with_warning = 0
        false_alarms = 0
        false_alarm_days = 0
        total_days = len(daily)

        for i, row in daily.iterrows():
            if row['trade_date'] in date_set:
                # 事件日：往前检查
                warned = False
                for offset in range(1, lookahead + 1):
                    if i - offset >= 0:
                        prev_row = daily.iloc[i - offset]
                        try:
                            if combo_fn(prev_row):
                                warned = True
                                break
                        except:
                            pass
                if warned:
                    events_with_warning += 1

        # 误报：非事件日前 lookahead 天触发信号但没有后续事件
        false_alarm_count = 0
        for i in range(len(daily)):
            if i < lookahead:
                continue
            if daily.iloc[i]['trade_date'] in date_set:
                continue
            # 检查是否触发信号
            try:
                if not combo_fn(daily.iloc[i]):
                    continue
            except:
                continue
            # 检查后面 lookahead 天是否有事件
            has_event = False
            for offset in range(1, lookahead + 1):
                if i + offset < len(daily):
                    if daily.iloc[i + offset]['trade_date'] in date_set:
                        has_event = True
                        break
            if not has_event:
                false_alarm_count += 1

        warning_rate = events_with_warning / n_events * 100

        # 计算总触发天数
        trigger_count = 0
        for i in range(len(daily)):
            try:
                if combo_fn(daily.iloc[i]) and i >= lookahead:
                    trigger_count += 1
            except:
                pass

        precision = events_with_warning / (events_with_warning + false_alarm_count) * 100 if (events_with_warning + false_alarm_count) > 0 else 0

        print(f"\n{combo_name}")
        print(f"  事件预警: {events_with_warning}/{n_events} ({warning_rate:.0f}%)")
        print(f"  误报: {false_alarm_count} 次")
        print(f"  信号精确率: {precision:.0f}%")
        print(f"  信号触发率: {trigger_count}/{total_days} ({trigger_count/total_days*100:.1f}%)")


def main():
    print("=" * 65)
    print("  早期预警信号分析")
    print("  目标事件: 行业90%下跌 + 指数日跌≥2.5%")
    print("=" * 65)

    conn = sqlite3.connect(DB_PATH)

    print("\n[1/3] 加载数据...")
    idx, sector = load_data(conn)
    print(f"  指数: {idx['trade_date'].min()} ~ {idx['trade_date'].max()} ({len(idx)}天)")
    print(f"  行业数据: {len(sector)} 行")

    print("\n[2/3] 识别事件 & 计算指标...")
    event_dates, daily = compute_event_dates(idx, sector)

    # 显示事件明细
    print(f"\n  事件列表 ({len(event_dates)} 次):")
    for d in event_dates:
        row = daily[daily['trade_date'] == d].iloc[0]
        print(f"    {d} 行业跌{row['down_pct']:.1f}% 指数{row['pct_chg']:.2f}%")

    print("\n[3/3] 信号分析...")

    all_results = {}
    for lookahead in [1, 2, 3, 5]:
        results = analyze_pre_event_signals(daily, event_dates, lookahead)
        all_results[lookahead] = results

    # 最佳 lookahead 综合
    print(f"\n{'='*65}")
    print(f"最佳信号汇总 (综合 1~5 天前)")
    print(f"{'='*65}")

    for lookahead in [2, 3, 5]:
        print(f"\n--- 事件前 {lookahead} 天最佳信号 Top 5 ---")
        results = sorted(all_results[lookahead], key=lambda x: -x['lift'])
        for r in results[:5]:
            print(f"  {r['signal']:<25} 命中率{r['hit_rate']:>6.1f}% (随机{r['random_rate']:.1f}%) 提升{r['lift']:.2f}x")

    # 最佳组合
    for lookahead in [2, 3]:
        print(f"\n--- 信号组合 (事件前{lookahead}天预警) ---")
        find_best_combination(daily, event_dates, lookahead)

    conn.close()
    print("\n✅ 分析完成")


if __name__ == '__main__':
    main()
