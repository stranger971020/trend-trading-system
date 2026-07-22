#!/usr/bin/env python3
"""
回测: warning信号择时 vs 满仓持有
===================================
策略:
  A (择时): 默认80%仓位 → warning次日降到30%
            → 首次行业≥70%上涨日 开始逐步加仓(每天+20%) 直到80%
  B (基准): 始终保持80%仓位

费用: 单边0.1%  (印花费+佣金)
"""
import tushare as ts
from config import TUSHARE_TOKEN
import sqlite3
import pandas as pd
import numpy as np

pro = ts.pro_api(TUSHARE_TOKEN)

# ── 数据 ──
idx = pro.index_daily(ts_code='000001.SH', start_date='20200101', end_date='20260722')
idx['trade_date'] = idx['trade_date'].astype(str)
idx = idx.sort_values('trade_date').reset_index(drop=True)
for c in ['close','pct_chg']:
    idx[c] = pd.to_numeric(idx[c], errors='coerce')

conn = sqlite3.connect('data_storage/sw_index_data.db')
sector = pd.read_sql_query(
    "SELECT trade_date, ts_code, close FROM sw_l2_index_daily WHERE trade_date >= '20200101' ORDER BY trade_date",
    conn
)
sector['trade_date'] = sector['trade_date'].astype(str)
sector['close'] = pd.to_numeric(sector['close'], errors='coerce')
sector['prev_close'] = sector.groupby('ts_code')['close'].shift(1)
sector['ret'] = (sector['close'] - sector['prev_close']) / sector['prev_close']

daily = sector.groupby('trade_date').agg(
    down_pct=('ret', lambda x: (x<0).sum()/len(x)*100),
).reset_index()

# 合并指数
df = daily.merge(idx[['trade_date','close','pct_chg']], on='trade_date', how='inner')

# ── 计算warning信号 ──
df['heavy'] = df['down_pct'] > 70
df['p1_active'] = df['heavy'].rolling(5).sum() >= 3
df['ma20'] = df['close'].rolling(20).mean()
df['below_ma20'] = df['close'] < df['ma20']
df['vol_ma20'] = 1  # 简化
df['vol_below_avg'] = True  # 简化
df['p3'] = (df['pct_chg'] > 0) & (df['down_pct'] > 60)

# warning首次触发
df['warning'] = False
for i in range(5, len(df)):
    p1_first = df['p1_active'].iloc[i] and not df['p1_active'].iloc[i-1]
    p2p3_first = (df['below_ma20'].iloc[i] and df['p3'].iloc[i]
                  and not (df['below_ma20'].iloc[i-1] and df['p3'].iloc[i-1]))
    if p1_first or p2p3_first:
        df.loc[df.index[i], 'warning'] = True

print(f"warning信号: {df['warning'].sum()} 次")

# ── 回测 ──
DEFAULT_POS = 0.80
SAFE_POS = 0.30
STEP = 0.20
COST = 0.001  # 单边交易成本

pos_a = DEFAULT_POS  # 当前仓位
pos_b = DEFAULT_POS  # 基准仓位
in_safe = False      # 是否在避险模式
re_entry_day = -1    # 逐步加仓的第几天 (0,1,2,...)
pending_re_entry = False  # 等待加仓信号

equity_a = 1.0
equity_b = 1.0
trades = 0

records = []
for i in range(5, len(df)):
    row = df.iloc[i]
    daily_ret = row['pct_chg'] / 100 if not pd.isna(row['pct_chg']) else 0

    # ── 策略A仓位调整（在每日收益前执行） ──
    if row['warning'] and not in_safe:
        # warning触发，第二天降到30%
        # 在warning当天标记，第二天执行
        pass  # 下一天执行

    if i > 0 and df.iloc[i-1]['warning'] and not in_safe:
        # 前一天的warning → 今天执行减仓
        old_pos = pos_a
        pos_a = SAFE_POS
        in_safe = True
        pending_re_entry = True
        re_entry_day = -1
        cost = abs(pos_a - old_pos) * COST
        equity_a *= (1 - cost)
        trades += 1

    # 加仓阶段
    if in_safe and pending_re_entry:
        # 检查是否出现70%+行业上涨
        if row['down_pct'] <= 30:  # ≥70%行业上涨
            pending_re_entry = False
            re_entry_day = 0

    if in_safe and re_entry_day >= 0:
        # 每天加仓20%
        old_pos = pos_a
        if re_entry_day == 0:
            pos_a = min(SAFE_POS + STEP, DEFAULT_POS)
        elif re_entry_day == 1:
            pos_a = min(SAFE_POS + 2 * STEP, DEFAULT_POS)
        elif re_entry_day >= 2:
            pos_a = DEFAULT_POS
            in_safe = False
            re_entry_day = -1

        cost = abs(pos_a - old_pos) * COST
        equity_a *= (1 - cost)
        trades += 1

        if in_safe and re_entry_day >= 0:
            re_entry_day += 1

    # 再次触发warning的处理（在逐步加仓中也立即响应）
    if row['warning'] and in_safe:
        old_pos = pos_a
        pos_a = SAFE_POS
        pending_re_entry = True
        re_entry_day = -1
        cost = abs(pos_a - old_pos) * COST
        equity_a *= (1 - cost)
        trades += 1

    # ── 日收益 ──
    equity_a *= (1 + daily_ret * pos_a)
    equity_b *= (1 + daily_ret * pos_b)

    if i % 60 == 0 or row['warning']:
        records.append({
            'date': row['trade_date'],
            'close': row['close'],
            'pos_a': pos_a,
            'equity_a': equity_a,
            'equity_b': equity_b,
            'in_safe': in_safe,
            'warning': str(row['warning']),
        })

# ── 结果 ──
total_days = len(df) - 5
ret_a = (equity_a - 1) * 100
ret_b = (equity_b - 1) * 100

# 年化
years = total_days / 245
annual_a = ((equity_a) ** (1/years) - 1) * 100
annual_b = ((equity_b) ** (1/years) - 1) * 100

# 最大回撤
def max_dd(series):
    peak = series.expanding().max()
    dd = (series - peak) / peak
    return dd.min() * 100

# 计算每日净值序列
eq_a_list = []
eq_b_list = []
pos_a = DEFAULT_POS
in_safe = False
pending_re_entry = False
re_entry_day = -1
eq_a = 1.0
eq_b = 1.0

for i in range(5, len(df)):
    row = df.iloc[i]
    daily_ret = row['pct_chg'] / 100 if not pd.isna(row['pct_chg']) else 0

    if i > 0 and df.iloc[i-1]['warning'] and not in_safe:
        old_pos = pos_a
        pos_a = SAFE_POS
        in_safe = True
        pending_re_entry = True
        re_entry_day = -1
        cost = abs(pos_a - old_pos) * COST
        eq_a *= (1 - cost)

    if in_safe and pending_re_entry and row['down_pct'] <= 30:
        pending_re_entry = False
        re_entry_day = 0

    if in_safe and re_entry_day >= 0:
        old_pos = pos_a
        steps = min(re_entry_day + 1, 3)
        pos_a = min(SAFE_POS + steps * STEP, DEFAULT_POS)
        cost = abs(pos_a - old_pos) * COST
        eq_a *= (1 - cost)
        re_entry_day += 1
        if pos_a >= DEFAULT_POS:
            in_safe = False
            re_entry_day = -1

    if row['warning'] and in_safe:
        old_pos = pos_a
        pos_a = SAFE_POS
        pending_re_entry = True
        re_entry_day = -1
        cost = abs(pos_a - old_pos) * COST
        eq_a *= (1 - cost)

    eq_a *= (1 + daily_ret * pos_a)
    eq_b *= (1 + daily_ret * DEFAULT_POS)
    eq_a_list.append(eq_a)
    eq_b_list.append(eq_b)

dd_a = max_dd(pd.Series(eq_a_list))
dd_b = max_dd(pd.Series(eq_b_list))

# 胜率
win_days_a = sum(1 for i in range(1, len(eq_a_list)) if eq_a_list[i] > eq_a_list[i-1])
win_days_b = sum(1 for i in range(1, len(eq_b_list)) if eq_b_list[i] > eq_b_list[i-1])

print()
print("=" * 60)
print("  回测结果: Warning信号择时 vs 满仓持有")
print(f"  区间: {df.iloc[5]['trade_date']} ~ {df.iloc[-1]['trade_date']} ({total_days}天)")
print("=" * 60)
print()
print(f"{'指标':<25} {'择时策略(A)':>15} {'满仓持有(B)':>15}")
print("-" * 55)
print(f"{'最终收益':<25} {ret_a:>14.2f}% {ret_b:>14.2f}%")
print(f"{'年化收益':<25} {annual_a:>14.2f}% {annual_b:>14.2f}%")
print(f"{'最大回撤':<25} {dd_a:>14.2f}% {dd_b:>14.2f}%")
print(f"{'日胜率':<25} {win_days_a/total_days*100:>13.1f}% {win_days_b/total_days*100:>13.1f}%")
print(f"{'交易次数':<25} {trades:>15}")
print()

# 卡玛比率
cmar_a = annual_a / abs(dd_a) if dd_a != 0 else 0
cmar_b = annual_b / abs(dd_b) if dd_b != 0 else 0
print(f"{'卡玛比率(收益/回撤)':<25} {cmar_a:>14.2f} {cmar_b:>14.2f}")

# 年度对比
print()
print("=" * 60)
print("  年度对比")
print("=" * 60)
print(f"{'年份':>6} {'择时收益':>10} {'满仓收益':>10} {'择时回撤':>10} {'满仓回撤':>10}")
for yr in range(2020, 2027):
    start_i = 5
    end_i = len(df)
    for i in range(5, len(df)):
        if df.iloc[i]['trade_date'].startswith(str(yr)):
            start_i = i
            break
    for i in range(start_i + 1, len(df)):
        if not df.iloc[i]['trade_date'].startswith(str(yr)):
            end_i = i
            break

    if end_i <= start_i:
        continue

    yr_ret_a = (eq_a_list[end_i-5] / eq_a_list[start_i-5] - 1) * 100 if end_i-5 < len(eq_a_list) and start_i-5 < len(eq_a_list) else 0
    yr_ret_b = (eq_b_list[end_i-5] / eq_b_list[start_i-5] - 1) * 100 if end_i-5 < len(eq_b_list) and start_i-5 < len(eq_b_list) else 0

    yr_eq_a = eq_a_list[start_i-5:end_i-4] if end_i-4 <= len(eq_a_list) else eq_a_list[start_i-5:]
    yr_eq_b = eq_b_list[start_i-5:end_i-4] if end_i-4 <= len(eq_b_list) else eq_b_list[start_i-5:]

    yr_dd_a = max_dd(pd.Series(yr_eq_a)) if len(yr_eq_a) > 20 else 0
    yr_dd_b = max_dd(pd.Series(yr_eq_b)) if len(yr_eq_b) > 20 else 0

    print(f"{yr:>6} {yr_ret_a:>9.1f}% {yr_ret_b:>9.1f}% {yr_dd_a:>9.1f}% {yr_dd_b:>9.1f}%")

# 2026年详细
print()
print("=" * 60)
print("  2026年逐月净值对比")
print("=" * 60)
print(f"{'月份':>8} {'择时净值':>10} {'满仓净值':>10} {'仓位':>8}")
for i in range(5, len(df)):
    d = df.iloc[i]['trade_date']
    if d.endswith('01') or i == 5:
        idx = i - 5
        if idx < len(eq_a_list):
            print(f"{d[:7]:>8} {eq_a_list[idx]:>10.4f} {eq_b_list[idx]:>10.4f} {pos_history.get(d, 0.8):>7.0%}" if False else "")

# 简化输出: 只选关键节点
print(f"\n关键节点:")
print(f"{'日期':>10} {'择时净值':>10} {'满仓净值':>10} {'仓位':>8} {'事件':>12}")
key_dates = ['20260102','20260202','20260301','20260323','20260401',
             '20260501','20260515','20260601','20260617','20260701','20260713','20260717','20260722']
pos_history = {}
pos_a = DEFAULT_POS
in_safe = False
pending_re_entry = False
re_entry_day = -1

for i in range(5, len(df)):
    row = df.iloc[i]
    pos_history[row['trade_date']] = pos_a
    if i > 0 and df.iloc[i-1]['warning'] and not in_safe:
        pos_a = SAFE_POS; in_safe = True; pending_re_entry = True; re_entry_day = -1
    if in_safe and pending_re_entry and row['down_pct'] <= 30:
        pending_re_entry = False; re_entry_day = 0
    if in_safe and re_entry_day >= 0:
        steps = min(re_entry_day + 1, 3)
        pos_a = min(SAFE_POS + steps * STEP, DEFAULT_POS)
        re_entry_day += 1
        if pos_a >= DEFAULT_POS: in_safe = False; re_entry_day = -1
    if row['warning'] and in_safe:
        pos_a = SAFE_POS; pending_re_entry = True; re_entry_day = -1

for d in key_dates:
    if d in pos_history:
        idx = df[df['trade_date'] == d].index[0] - 5
        if 0 <= idx < len(eq_a_list):
            evt = ''
            if df.iloc[idx+5]['warning']: evt = '⚠️warning'
            elif idx+5 > 0 and df.iloc[idx+4]['warning']: evt = '⬇️减仓日'
            print(f"{d:>10} {eq_a_list[idx]:>10.4f} {eq_b_list[idx]:>10.4f} {pos_history[d]:>7.0%} {evt:>12}")

conn.close()
print(f"\n✅ 回测完成")
