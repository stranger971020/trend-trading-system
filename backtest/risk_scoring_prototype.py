#!/usr/bin/env python3
"""
大盘风险评分模型 — 原型验证
=============================
基于历史可用的行业级数据构建复合风险评分，验证对大盘回撤的预测能力。

数据源:
  - L2 行业日线 (2015至今, 123行业): 行业涨跌比、行业MA20突破比
  - 上证指数日线 (2005至今): 波动率扩张、指数本身的技术指标
  - L2 行业成交量: 市场成交量冲击

风险评分因子:
  1. **行业涨跌比**: 当日上涨行业占比（强势市场=低风险）
  2. **MA20突破比**: 站上MA20的行业占比（趋势市场=低风险）
  3. **波动率扩张**: 5日ATR / 20日ATR（扩张=高风险）
  4. **成交量冲击**: 全市场金额 / 20日均值（异常放量=高风险）
  5. **指数自身**: 指数距MA20偏离度（过远=回调风险）
  6. **RSI市场均值**: 123行业RSI中位数（过热=高风险）

输出:
  - 各因子单独预测能力
  - 复合评分 vs 当前阈值系统的对比
  - 最优因子权重建议
"""
import os, sys, json
from datetime import datetime, timedelta
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

np.seterr(all='ignore')

# ── 参数 ──
LOOKAHEAD = 10          # 向前看 N 个交易日
DRAWDOWN_THRESHOLD = -0.03  # 最大回撤 ≥ -3% 视为有风险
TRAIN_END = "20231231"  # 训练集截止
TEST_START = "20240101"  # 测试集开始

L2_L1_SECTORS = 31     # 一级行业数
L2_L2_SECTORS = 123    # 二级行业数


# ═══════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════

def load_index(conn) -> pd.DataFrame:
    """加载上证指数"""
    import tushare as ts
    from config import TUSHARE_TOKEN
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.index_daily(ts_code='000001.SH', start_date='20100101', end_date='20260722')
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values('trade_date').reset_index(drop=True)
    for c in ['close', 'open', 'high', 'low', 'pre_close', 'vol', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_l2_data(conn) -> pd.DataFrame:
    """加载全部 L2 行业数据（close 用于涨跌比和 MA20 突破比）"""
    df = pd.read_sql_query(
        "SELECT trade_date, ts_code, close, vol, amount FROM sw_l2_index_daily "
        "WHERE trade_date >= '20100101' ORDER BY trade_date",
        conn
    )
    df['trade_date'] = df['trade_date'].astype(str)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    return df


def load_l1_data(conn) -> pd.DataFrame:
    """加载 L1 行业数据"""
    df = pd.read_sql_query(
        "SELECT trade_date, ts_code, close FROM sw_index_daily "
        "WHERE trade_date >= '20100101' ORDER BY trade_date",
        conn
    )
    df['trade_date'] = df['trade_date'].astype(str)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df


# ═══════════════════════════════════════════════
#  特征计算
# ═══════════════════════════════════════════════

def compute_features(index_df: pd.DataFrame, l2_df: pd.DataFrame, l1_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个交易日计算全部风险因子

    Returns:
        DataFrame with columns: trade_date, close, breadth_ratio, ma20_ratio,
        vol_expansion, volume_shock, index_deviation, rsi_median, risk_label
    """
    print("计算特征...")
    all_dates = sorted(index_df['trade_date'].values)

    rows = []
    total = len(all_dates)

    # 预计算 L2 行业的技术指标（逐行业计算，避免重复）
    print("  预计算行业技术指标...")
    sector_tech = {}  # ts_code → DataFrame with date, close, ma20, rsi, return
    for code in l2_df['ts_code'].unique():
        sdf = l2_df[l2_df['ts_code'] == code].sort_values('trade_date').copy()
        if len(sdf) < 60:
            continue
        sdf['return'] = sdf['close'].pct_change()
        sdf['ma20'] = sdf['close'].rolling(20).mean()
        # RSI
        delta = sdf['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        sdf['rsi'] = 100 - (100 / (1 + rs))
        sector_tech[code] = sdf

    print(f"  已处理 {len(sector_tech)} 个行业的技术指标")

    for i, date in enumerate(all_dates):
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{total} ({date})")

        # 当前指数数据
        idx_row = index_df[index_df['trade_date'] == date]
        if idx_row.empty:
            continue
        idx = idx_row.iloc[0]
        close = idx['close']

        # ── 1. 行业涨跌比 (Breadth) ──
        up_count = 0
        total_count = 0
        ma20_above = 0
        all_rsi = []

        for code, sdf in sector_tech.items():
            srow = sdf[sdf['trade_date'] == date]
            if srow.empty:
                continue
            sr = srow.iloc[0]

            # 涨跌
            ret = sr.get('return')
            if ret is not None and not pd.isna(ret):
                total_count += 1
                if ret > 0:
                    up_count += 1

            # MA20 突破
            ma20 = sr.get('ma20')
            if ma20 is not None and not pd.isna(ma20) and ma20 > 0:
                if sr['close'] > ma20:
                    ma20_above += 1

            # RSI
            rsi = sr.get('rsi')
            if rsi is not None and not pd.isna(rsi):
                all_rsi.append(rsi)

        breadth_ratio = up_count / total_count if total_count > 0 else 0.5
        ma20_ratio = ma20_above / total_count if total_count > 0 else 0.5
        rsi_median = float(np.median(all_rsi)) if len(all_rsi) > 0 else 50

        # ── 2. 指数波动率扩张 ──
        hist_idx = index_df[index_df['trade_date'] <= date].tail(25)
        if len(hist_idx) >= 20:
            returns = hist_idx['close'].pct_change().dropna()
            atr_5 = returns.tail(5).std()
            atr_20 = returns.tail(20).std()
            vol_expansion = atr_5 / atr_20 if atr_20 > 0 else 1.0
        else:
            vol_expansion = 1.0

        # ── 3. 成交量冲击 ──
        # 使用指数成交额
        amount = idx.get('amount', 0)
        hist_amount = index_df[index_df['trade_date'] <= date].tail(21)['amount']
        if len(hist_amount) >= 20:
            ma_amount = hist_amount.head(20).mean()
            volume_shock = amount / ma_amount if ma_amount > 0 else 1.0
        else:
            volume_shock = 1.0

        # ── 4. 指数距 MA20 偏离度 ──
        hist_close = index_df[index_df['trade_date'] <= date].tail(21)['close']
        if len(hist_close) >= 20:
            ma20_idx = hist_close.mean()
            deviation = (close - ma20_idx) / ma20_idx if ma20_idx > 0 else 0
        else:
            deviation = 0

        rows.append({
            'trade_date': date,
            'close': close,
            'breadth_ratio': breadth_ratio,       # 0-1, 越高越强
            'ma20_ratio': ma20_ratio,             # 0-1, 越高越强
            'vol_expansion': vol_expansion,        # >1 = 波动扩张
            'volume_shock': volume_shock,          # >1.5 = 异常放量
            'index_deviation': deviation,           # 百分比, >5% = 偏离过大
            'rsi_median': rsi_median,               # 0-100
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════
#  风险标签 & 评估
# ═══════════════════════════════════════════════

def add_risk_label(df: pd.DataFrame, index_df: pd.DataFrame, lookahead=10, threshold=-0.03):
    """添加向前看的风险标签"""
    print(f"添加风险标签 (向前{lookahead}日, 回撤≥{threshold*100:.0f}%)...")
    labels = []
    for _, row in df.iterrows():
        date = row['trade_date']
        future = index_df[index_df['trade_date'] > date].head(lookahead)
        if len(future) < 3:
            labels.append(False)
            continue
        start = future.iloc[0]['close']
        min_c = future['close'].min()
        dd = (min_c - start) / start
        labels.append(dd <= threshold)
    df['risk_label'] = labels
    return df


def evaluate_single_factor(df: pd.DataFrame, factor: str, direction: str = 'reverse',
                           bins: int = 10) -> dict:
    """
    评估单因子预测能力

    Args:
        factor: 因子列名
        direction: 'reverse' = 值越高风险越高, 'normal' = 值越低风险越高
        bins: 分桶数
    """
    from sklearn.metrics import roc_auc_score

    valid = df[['trade_date', factor, 'risk_label']].dropna()
    if len(valid) < 100:
        return {'factor': factor, 'error': '数据不足'}

    # 分桶
    valid['bucket'] = pd.qcut(valid[factor], bins, labels=False, duplicates='drop')

    # 每个桶的风险率
    bucket_risk = valid.groupby('bucket')['risk_label'].agg(['mean', 'count']).reset_index()
    bucket_risk.columns = ['bucket', 'risk_rate', 'count']

    # 单调性
    if direction == 'reverse':
        expected = bucket_risk['risk_rate'].is_monotonic_decreasing
    else:
        expected = bucket_risk['risk_rate'].is_monotonic_increasing

    # AUC (用因子值直接预测)
    try:
        if direction == 'reverse':
            auc = roc_auc_score(valid['risk_label'], -valid[factor])
        else:
            auc = roc_auc_score(valid['risk_label'], valid[factor])
    except:
        auc = 0.5

    # 极端桶对比：最高风险桶 vs 最低风险桶
    top_bucket = bucket_risk.loc[bucket_risk['bucket'].idxmax()]
    bot_bucket = bucket_risk.loc[bucket_risk['bucket'].idxmin()]
    if direction == 'reverse':
        high_risk_bucket, low_risk_bucket = bot_bucket, top_bucket
    else:
        high_risk_bucket, low_risk_bucket = top_bucket, bot_bucket

    lift = high_risk_bucket['risk_rate'] / low_risk_bucket['risk_rate'] if low_risk_bucket['risk_rate'] > 0 else 1.0

    return {
        'factor': factor,
        'direction': direction,
        'auc': round(auc, 4),
        'lift': round(lift, 2),
        'monotonic': bool(expected),
        'high_bucket_risk': round(high_risk_bucket['risk_rate'] * 100, 1),
        'low_bucket_risk': round(low_risk_bucket['risk_rate'] * 100, 1),
        'buckets': len(bucket_risk),
        'samples': len(valid),
    }


def build_composite_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    构建复合评分

    weights: {factor: weight}, weight 正数=值越高越危险
    """
    score = pd.Series(0.0, index=df.index)

    for factor, weight in weights.items():
        if factor not in df.columns:
            continue
        col = df[factor]
        # Min-max 归一化
        mn, mx = col.min(), col.max()
        if mx > mn:
            normalized = (col - mn) / (mx - mn)
        else:
            normalized = col * 0
        score += normalized * weight

    return score


def confusion_matrix(df: pd.DataFrame, pred_col: str, threshold_percentile: float):
    """使用分数百分位作为阈值计算混淆矩阵"""
    thresh = df[pred_col].quantile(threshold_percentile)
    pred = df[pred_col] >= thresh
    actual = df['risk_label']

    tp = ((pred) & (actual)).sum()
    fp = ((pred) & (~actual)).sum()
    tn = ((~pred) & (~actual)).sum()
    fn = ((~pred) & (actual)).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    return {
        'threshold_percentile': threshold_percentile,
        'threshold_value': round(thresh, 3),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'f1': round(f1, 4), 'accuracy': round(acc, 4),
        'alarm_pct': round(pred.mean() * 100, 1),
    }


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  大盘风险评分模型 — 原型验证")
    print(f"  风险定义: {LOOKAHEAD}日内最大回撤 ≥ {DRAWDOWN_THRESHOLD*100:.0f}%")
    print(f"  训练集: ~{TRAIN_END}, 测试集: {TEST_START}~")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    index_df = load_index(conn)
    l2_df = load_l2_data(conn)
    l1_df = load_l1_data(conn)
    print(f"  指数: {len(index_df)} 天")
    print(f"  L2行业: {len(l2_df)} 行")

    # 2. 计算特征
    print("\n[2/5] 计算风险因子...")
    df = compute_features(index_df, l2_df, l1_df)
    print(f"  有效交易日: {len(df)}")

    # 3. 添加风险标签
    print("\n[3/5] 添加风险标签...")
    df = add_risk_label(df, index_df, lookahead=LOOKAHEAD, threshold=DRAWDOWN_THRESHOLD)
    risk_rate = df['risk_label'].mean() * 100
    print(f"  有风险天数: {df['risk_label'].sum()} / {len(df)} ({risk_rate:.1f}%)")

    # 4. 单因子评估
    print("\n[4/5] 单因子预测能力评估...")
    factors = [
        ('breadth_ratio', 'reverse'),     # 涨跌比越低 → 风险越高
        ('ma20_ratio', 'reverse'),         # MA20突破比越低 → 风险越高
        ('vol_expansion', 'normal'),        # 波动扩张越高 → 风险越高
        ('volume_shock', 'normal'),         # 成交量异常 → 风险越高
        ('index_deviation', 'normal'),      # 偏离MA20越远 → 风险越高
        ('rsi_median', 'normal'),           # RSI中位数过高 → 风险越高
    ]

    results = []
    for factor, direction in factors:
        r = evaluate_single_factor(df, factor, direction)
        results.append(r)
        mark = "✅" if r.get('auc', 0) > 0.55 else "❌" if r.get('auc', 0) < 0.52 else "➖"
        print(f"  {mark} {factor:20s}: AUC={r.get('auc', 'N/A'):>6}  "
              f"Lift={r.get('lift', 'N/A')}  "
              f"单调性={'✓' if r.get('monotonic') else '✗'}")

    # 5. 构建复合评分
    print("\n[5/5] 复合评分模型...")

    # 等权组合（方向统一：值越高=风险越高）
    weights = {
        'breadth_ratio': -1.0,    # 负号：涨跌比越低风险越高
        'ma20_ratio': -1.0,       # 负号：突破比越低风险越高
        'vol_expansion': 1.0,     # 正号：波动扩张越高风险越高
        'volume_shock': 0.5,      # 正号：成交异常放量
        'index_deviation': 1.0,   # 正号：偏离越远风险越高
        'rsi_median': 0.5,        # 正号：RSI过高风险
    }

    df['composite_score'] = build_composite_score(df, weights)

    # 训练集/测试集
    train_df = df[df['trade_date'] <= TRAIN_END].copy()
    test_df = df[df['trade_date'] >= TEST_START].copy()
    print(f"  训练集: {len(train_df)} 天 | 测试集: {len(test_df)} 天")

    # 在测试集上评估
    print(f"\n  ── 测试集结果 ──")
    print(f"  {'报警比例':>8}  {'精确率':>8}  {'召回率':>8}  {'F1':>8}  {'准确率':>8}")
    for pctl in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        cm = confusion_matrix(test_df, 'composite_score', pctl)
        print(f"  {cm['alarm_pct']:>7.1f}%  {cm['precision']:>7.1%}  {cm['recall']:>7.1%}  "
              f"{cm['f1']:>7.3f}  {cm['accuracy']:>7.1%}")

    # 对比当前阈值系统（复用之前的回测结果）
    print(f"\n  ── 对比: 当前阈值系统（同时间段） ──")
    current_test = test_df.copy()
    # 模拟当前阈值: divergence >= 4 → critical
    # 用 ma20_ratio 低 + vol_expansion 高作为当前系统的近似
    # 实际对比用 backtest_thresholds.py 的结果更准确
    print(f"  当前系统: Precision ~17-22%, F1 ~0.27-0.33")
    print(f"  复合模型最优 F1 见上表")

    # 6. 输出最佳因子权重建议
    print(f"\n  ── 因子重要性排序（基于 AUC） ──")
    sorted_results = sorted(results, key=lambda x: x.get('auc', 0), reverse=True)
    for i, r in enumerate(sorted_results):
        print(f"  {i+1}. {r['factor']:20s} AUC={r.get('auc', 0):.4f}")

    conn.close()
    print("\n✅ 原型验证完成")


if __name__ == '__main__':
    main()
