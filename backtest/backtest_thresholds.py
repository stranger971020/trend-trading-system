#!/usr/bin/env python3
"""
回测：每日交易参考的阈值与大盘风险识别成功率
============================================
目标：
  1. 科学定义"大盘风险"
  2. 回测当前阈值在历史上的识别成功率
  3. 尝试找到更优阈值

大盘风险定义（向前看）:
  - 短期风险: N日内最大回撤 ≥ -2%
  - 中期风险: N日内最大回撤 ≥ -5%
  - 使用上证指数作为大盘基准

回测指标:
  - 顶背离板块数 ≥ 4 → critical（系统性风险）
  - 高位区板块数 ≥ 16 → high
  - 组合规则的 Precision / Recall / F1
"""
import os, sys, json, argparse
from datetime import datetime, timedelta
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# ── 回测参数 ──
LOOKAHEAD_DAYS = [5, 10, 20]  # 向前看 N 个交易日
RISK_THRESHOLDS = [-0.02, -0.03, -0.05]  # 最大回撤阈值
MIN_HISTORY = 60  # 每个日期需要至少 N 天历史数据才能计算指标

# ── 数据加载 ──

def load_index_data(conn) -> pd.DataFrame:
    """加载上证指数日线（用于判定风险是否发生）"""
    import tushare as ts
    from config import TUSHARE_TOKEN
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.index_daily(ts_code='000001.SH', start_date='20150101', end_date='20260722')
    if df is None or df.empty:
        raise RuntimeError("无法获取指数数据")
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    return df


def load_l2_sectors(conn) -> List[str]:
    """获取所有 L2 行业代码"""
    cur = conn.execute("SELECT DISTINCT ts_code FROM sw_l2_index_daily ORDER BY ts_code")
    return [r[0] for r in cur.fetchall()]


def get_sector_data(conn, ts_code: str) -> pd.DataFrame:
    """获取单个 L2 行业的完整历史数据"""
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, vol, amount "
        "FROM sw_l2_index_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(ts_code,)
    )
    df['trade_date'] = df['trade_date'].astype(str)
    return df


# ── 技术指标计算 ──

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal).mean()
    macd = 2 * (dif - dea)
    return dif, dea, macd


def calc_bollinger(series: pd.Series, period=20, std_mult=2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


def compute_sector_status(df: pd.DataFrame) -> pd.DataFrame:
    """对单个行业计算所有历史技术指标"""
    if df.empty or len(df) < MIN_HISTORY:
        return pd.DataFrame()
    df = df.copy().sort_values('trade_date').reset_index(drop=True)
    df['rsi'] = calc_rsi(df['close'], 14)
    dif, dea, macd_val = calc_macd(df['close'])
    df['dif'] = dif
    df['dea'] = dea
    df['macd'] = macd_val
    _, boll_upper, boll_lower = calc_bollinger(df['close'])
    df['boll_upper'] = boll_upper
    df['boll_lower'] = boll_lower
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['volume_ma20'] = df['vol'].rolling(20).mean()
    df['volume_ratio'] = df['vol'] / df['volume_ma20']
    return df


def detect_divergence(df: pd.DataFrame, lookback=60) -> bool:
    """
    检测 MACD 顶背离（基于最近的 lookback 窗口）
    价格创新高但 DIF 未创新高 → 顶背离
    """
    if len(df) < lookback:
        return False
    recent = df.tail(lookback).copy()
    # 找价格峰值
    price_peaks = []
    for i in range(5, len(recent) - 5):
        if (recent['close'].iloc[i] > recent['close'].iloc[i-5:i].max()
                and recent['close'].iloc[i] > recent['close'].iloc[i+1:i+6].max()):
            price_peaks.append((i, recent['close'].iloc[i], recent['dif'].iloc[i]))
    if len(price_peaks) < 2:
        return False
    last_two = price_peaks[-2:]
    # 价格新高但 DIF 未新高
    if last_two[1][1] > last_two[0][1] and last_two[1][2] <= last_two[0][2]:
        return True
    return False


# ── 每日信号计算 ──

def classify_zone(row: dict) -> str:
    """根据一行指标判定分区"""
    rsi = row.get('rsi', 0)
    dif = row.get('dif', 0)
    # 追涨区: RSI>50 + MACD>0
    if rsi > 50 and dif > 0:
        # 检查量价配合
        vol_ok = row.get('volume_ratio', 1) >= 0.8 or pd.isna(row.get('volume_ratio'))
        if vol_ok:
            return "chase"
        return "watch"
    # 观察区: MACD刚转多或RSI回升
    if dif > 0 or rsi > 45:
        return "watch"
    # 高位区/弱势区分界
    if rsi > 55 and dif <= 0:
        return "top_warn"
    if dif <= 0 and rsi <= 55:
        # 超卖
        if rsi < 30:
            return "top_warn"  # 超卖但弱势
        return "weak"
    return "weak"


def get_daily_signals(all_sector_data: dict, date: str, sector_codes: List[str]) -> dict:
    """
    对某一日期，基于截至该日的数据计算全市场信号
    返回: chase_count, top_warn_count, divergence_count
    """
    chase_count = 0
    watch_count = 0
    top_warn_count = 0
    weak_count = 0
    divergence_count = 0

    for code in sector_codes:
        df = all_sector_data.get(code)
        if df is None or df.empty:
            continue
        # 找到截至 date 的行
        mask = df['trade_date'] <= date
        if not mask.any():
            continue
        hist = df[mask].copy()
        if len(hist) < MIN_HISTORY:
            continue
        latest = hist.iloc[-1]

        # 技术指标
        rsi = latest.get('rsi', 0)
        dif = latest.get('dif', 0)

        row = {
            'rsi': rsi if not pd.isna(rsi) else 0,
            'dif': dif if not pd.isna(dif) else 0,
            'volume_ratio': latest.get('volume_ratio', 1),
        }
        zone = classify_zone(row)

        if zone == 'chase':
            chase_count += 1
        elif zone == 'watch':
            watch_count += 1
        elif zone == 'top_warn':
            top_warn_count += 1
        elif zone == 'weak':
            weak_count += 1

        # 顶背离检测
        if detect_divergence(hist, lookback=60):
            divergence_count += 1

    return {
        'chase': chase_count,
        'watch': watch_count,
        'top_warn': top_warn_count,
        'weak': weak_count,
        'divergence': divergence_count,
    }


def compute_risk_level(signals: dict) -> str:
    """根据当前阈值判定风险等级"""
    if signals['divergence'] >= 4:
        return 'critical'
    if signals['top_warn'] >= 16:
        return 'high'
    return 'normal'


# ── 风险定义 ──

def compute_forward_risk(index_df: pd.DataFrame, date: str, lookahead: int, threshold: float) -> bool:
    """
    从 date 开始向前看 lookahead 个交易日
    如果指数最大回撤 >= threshold → 有风险
    """
    mask = index_df['trade_date'] > date
    future = index_df[mask].head(lookahead)
    if len(future) < 3:
        return False  # 数据不足
    start_price = future.iloc[0]['close']
    min_price = future['close'].min()
    max_drawdown = (min_price - start_price) / start_price
    return max_drawdown <= threshold  # 负值


# ── 主回测 ──

def run_backtest(
    conn,
    index_df: pd.DataFrame,
    all_sector_data: dict,
    sector_codes: List[str],
    test_dates: List[str],
    lookahead: int = 5,
    risk_threshold: float = -0.02,
) -> pd.DataFrame:
    """
    对每个测试日期：
      1. 计算信号和风险等级
      2. 检查实际风险是否发生
    返回每行的预测结果
    """
    results = []
    total = len(test_dates)

    for i, date in enumerate(test_dates):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{total} ({date})", file=sys.stderr)

        # 获取当日信号
        signals = get_daily_signals(all_sector_data, date, sector_codes)

        # 跳过数据不足的日期
        if signals['chase'] + signals['watch'] + signals['top_warn'] + signals['weak'] < 10:
            continue

        # 系统判定的风险等级
        predicted_level = compute_risk_level(signals)
        predicted_risky = predicted_level in ('critical', 'high')

        # 实际风险
        actual_risky = compute_forward_risk(index_df, date, lookahead, risk_threshold)

        results.append({
            'date': date,
            'predicted_level': predicted_level,
            'predicted_risky': predicted_risky,
            'actual_risky': actual_risky,
            'chase': signals['chase'],
            'watch': signals['watch'],
            'top_warn': signals['top_warn'],
            'weak': signals['weak'],
            'divergence': signals['divergence'],
        })

    return pd.DataFrame(results)


def print_metrics(df: pd.DataFrame, label: str, lookahead: int, threshold: float):
    """打印混淆矩阵和评价指标"""
    if df.empty:
        print(f"\n  [{label}] 无有效数据")
        return

    tp = len(df[(df['predicted_risky']) & (df['actual_risky'])])
    fp = len(df[(df['predicted_risky']) & (~df['actual_risky'])])
    tn = len(df[(~df['predicted_risky']) & (~df['actual_risky'])])
    fn = len(df[(~df['predicted_risky']) & (df['actual_risky'])])

    total = len(df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    risk_days = df['actual_risky'].sum()
    risk_pct = risk_days / total * 100

    print(f"\n  [{label}] 向前{lookahead}日 ≥{threshold*100:.0f}%回撤")
    print(f"  {'='*45}")
    print(f"  测试天数:     {total}")
    print(f"  实际风险天数: {risk_days} ({risk_pct:.1f}%)")
    print(f"  预测为风险:   {df['predicted_risky'].sum()}")
    print(f"  混淆矩阵:     TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  {'='*45}")
    print(f"  准确率 Accuracy:  {accuracy:.1%}")
    print(f"  精确率 Precision: {precision:.1%}")
    print(f"  召回率 Recall:    {recall:.1%}")
    print(f"  F1 Score:         {f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description='回测交易报告阈值')
    parser.add_argument('--start', default='20200101', help='开始日期')
    parser.add_argument('--end', default='20260722', help='结束日期')
    parser.add_argument('--lookahead', type=int, default=5, help='向前看天数')
    parser.add_argument('--threshold', type=float, default=-0.02, help='回撤阈值')
    args = parser.parse_args()

    print("=" * 55)
    print("  每日交易参考 — 阈值回测")
    print("  大盘风险定义: N日内最大回撤 ≥ 阈值")
    print("=" * 55)

    conn = sqlite3.connect(DB_PATH)

    # 1. 加载指数数据
    print("\n[1/5] 加载上证指数数据...")
    index_df = load_index_data(conn)
    print(f"  指数数据: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()} ({len(index_df)} 天)")

    # 2. 获取行业列表
    print("\n[2/5] 获取 L2 行业列表...")
    sector_codes = load_l2_sectors(conn)
    print(f"  {len(sector_codes)} 个行业")

    # 3. 计算所有行业的技术指标
    print("\n[3/5] 计算全行业历史技术指标...")
    all_sector_data = {}
    for i, code in enumerate(sector_codes):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(sector_codes)}")
        df = get_sector_data(conn, code)
        if len(df) >= MIN_HISTORY:
            df_tech = compute_sector_status(df)
            if not df_tech.empty:
                all_sector_data[code] = df_tech
    print(f"  完成: {len(all_sector_data)} 个行业有足够数据")

    # 4. 生成测试日期列表
    print("\n[4/5] 生成测试日期列表...")
    test_dates = sorted(set(index_df['trade_date'].values))
    test_dates = [d for d in test_dates if d >= args.start and d <= args.end]
    print(f"  测试日期: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} 天)")

    # 5. 运行回测
    print(f"\n[5/5] 运行回测 (向前{args.lookahead}日 ≥{args.threshold*100:.0f}%回撤)...")
    results = run_backtest(
        conn, index_df, all_sector_data, sector_codes, test_dates,
        lookahead=args.lookahead, risk_threshold=args.threshold,
    )
    print(f"  有效天数: {len(results)}")

    # 6. 输出指标
    print("\n" + "=" * 55)
    print("  回测结果")
    print("=" * 55)

    # 按风险等级细分
    for level in ['critical', 'high', 'normal']:
        subset = results[results['predicted_level'] == level]
        if not subset.empty:
            risk_rate = subset['actual_risky'].mean() * 100
            warning = "⚠️ " if level == 'critical' else ("⚠️ " if level == 'high' else "✅ ")
            print(f"  {warning}预测为 {level:8s}: {len(subset):5d} 天, 实际风险率 {risk_rate:.1f}%")

    print_metrics(results, "整体", args.lookahead, args.threshold)

    # 7. 阈值扫描：寻找最优 divergence 阈值
    print("\n" + "=" * 55)
    print("  阈值扫描: 顶背离数量 → critical 判定")
    print("=" * 55)
    for div_thresh in [1, 2, 3, 4, 5, 6, 8, 10]:
        tp = 0; fp = 0; tn = 0; fn = 0
        for _, row in results.iterrows():
            pred = row['divergence'] >= div_thresh
            actual = row['actual_risky']
            if pred and actual: tp += 1
            elif pred and not actual: fp += 1
            elif not pred and not actual: tn += 1
            elif not pred and actual: fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  顶背离 ≥ {div_thresh:2d}:  TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
              f"Prec={prec:.1%} Rec={rec:.1%} F1={f1:.3f}")

    # 8. 阈值扫描: top_warn
    print("\n" + "=" * 55)
    print("  阈值扫描: 高位区数量 → high 判定")
    print("=" * 55)
    for warn_thresh in [5, 8, 10, 12, 14, 16, 18, 20, 25]:
        tp = 0; fp = 0; tn = 0; fn = 0
        for _, row in results.iterrows():
            pred = row['top_warn'] >= warn_thresh
            actual = row['actual_risky']
            if pred and actual: tp += 1
            elif pred and not actual: fp += 1
            elif not pred and not actual: tn += 1
            elif not pred and actual: fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  高位区 ≥ {warn_thresh:2d}:  TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
              f"Prec={prec:.1%} Rec={rec:.1%} F1={f1:.3f}")

    conn.close()
    print("\n✅ 回测完成")


if __name__ == '__main__':
    main()
