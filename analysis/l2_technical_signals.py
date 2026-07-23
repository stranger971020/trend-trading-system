"""
L2 二级行业技术指标分析 — RSI/MACD/量价/蜡烛图
为每日交易提供具体板块分区和信号

输出:
  zone: 🔥追涨 / ⏳观察 / ⚠️高位 / ❌弱势
  signal: 具体技术信号说明
  prices: 入场区间/止损/目标(估算)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 参数 ──
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_LOOKBACK = 20
TOP_SIGNAL_LOOKBACK = 60


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series) -> dict:
    ema12 = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema26 = close.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = dif - dea
    return {"dif": dif, "dea": dea, "hist": hist}


def detect_top_candle(high: pd.Series, low: pd.Series, close: pd.Series, open_p: pd.Series, i: int) -> dict:
    """检测最后3根K线是否为黄昏星/乌云盖顶"""
    signals = []
    if i < 3:
        return signals

    c1, c2, c3 = close.iloc[i - 2], close.iloc[i - 1], close.iloc[i]
    o1, o2, o3 = open_p.iloc[i - 2], open_p.iloc[i - 1], open_p.iloc[i]
    h1, l1 = high.iloc[i - 2], low.iloc[i - 2]

    # 黄昏星: 大阳→星线→大阴深入阳线实体50%+
    body1 = abs(c1 - o1)
    body3 = abs(c3 - o3)
    if body1 > 0 and body3 > 0:
        is_big_up = (c1 > o1) and body1 > np.mean(high.iloc[i - 20:i] - low.iloc[i - 20:i]) * 0.6
        is_star = abs(c2 - o2) < body1 * 0.3
        is_big_down = (c3 < o3) and (c3 < c1 - body1 * 0.5)
        if is_big_up and is_star and is_big_down:
            signals.append(("黄昏星", "🔴 黄昏星(高确定性见顶)", 3))

        # 乌云盖顶: 阳线后高开低走,收盘深入前阳实体
        if (c1 > o1) and (o3 > c1) and (c3 < o1 + (c1 - o1) * 0.5):
            signals.append(("乌云盖顶", "🌤️ 乌云盖顶(短期见顶)", 2))

    return signals


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算 ATR (Average True Range)"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_price_levels(price: float, atr: float, above_ma20: bool, above_ma60: bool) -> dict:
    """根据波动率和均线位置计算入场/止损/目标价"""
    atr_pct = atr / price if price > 0 else 0.01
    
    if above_ma20:
        # 趋势向上: 以MA20附近为止损参考
        entry_low = round(price * (1 - atr_pct * 0.5), 2)
        entry_high = round(price, 2)
        stop_loss = round(price * (1 - atr_pct * 1.5), 2)
        target_1 = round(price * (1 + atr_pct * 1.0), 2)
        target_2 = round(price * (1 + atr_pct * 2.0), 2)
    else:
        # 趋势向下或震荡: 严格止损
        entry_low = round(price * (1 - atr_pct * 0.3), 2)
        entry_high = round(price * (1 + atr_pct * 0.2), 2)
        stop_loss = round(price * (1 - atr_pct * 1.2), 2)
        target_1 = round(price * (1 + atr_pct * 0.8), 2)
        target_2 = round(price * (1 + atr_pct * 1.5), 2)
    
    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "expected_return_t1": round((target_1 / price - 1) * 100, 1),
        "expected_return_t2": round((target_2 / price - 1) * 100, 1),
        "risk": round((price - stop_loss) / price * 100, 1),
        "reward_risk_t1": round((target_1 - price) / (price - stop_loss), 1) if (price - stop_loss) > 0 else 0,
    }


def analyze_sector_stocks(db_path: str, chase_codes: list, latest_date: str, top_n: int = 5) -> list:
    """基于多因子评分从追涨区L2板块挑选最优个股

    评分因子(每个因子归一化0-10):
    1. 超额收益(权重0.25): 个股20日收益 - 所属L2板块20日收益
    2. 动量强度(权重0.20): 个股20日绝对收益
    3. 量价配合(权重0.20): 近5日均量 / 近20日均量
    4. 趋势位置(权重0.20): 收盘价偏离MA20的程度(正偏离加分)
    5. 流动性(权重0.15): 20日均成交额分位数

    返回: sorted list of dict, each containing code/name/score/factors
    """
    import sqlite3, pandas as pd, numpy as np, os
    conn = sqlite3.connect(db_path)

    # 加载行业映射CSV
    csv_candidates = [
        os.path.join(os.path.dirname(db_path), 'sw_industry_complete_pro.csv'),
        os.path.expanduser('~/hermes/projects/zh_a_trend_trading/src/data/sw_industry_complete_pro.csv'),
    ]
    mapping = None
    for cp in csv_candidates:
        if os.path.exists(cp):
            mapping = pd.read_csv(cp)
            break
    if mapping is None:
        conn.close()
        return []

    # 构建股票名称映射表
    name_map = dict(zip(mapping['ts_code'], mapping['name']))

    # 获取追涨区各板块的评分(用于板块排序)
    all_picks = []
    for l2_code in chase_codes[:5]:  # Top 5 chase sectors
        # 获取该L2板块内的个股列表
        sector_stocks = mapping[mapping['l2_code'] == l2_code]['ts_code'].tolist()
        if not sector_stocks:
            continue
        stock_set = set(sector_stocks)

        # 获取板块指数20日收益(作为基准)
        bench_ret = 0
        rows = conn.execute(
            f"SELECT close FROM sw_l2_index_daily WHERE ts_code='{l2_code}' AND trade_date<='{latest_date}' ORDER BY trade_date DESC LIMIT 21"
        ).fetchall()
        if len(rows) >= 21:
            bench_prices = [r[0] for r in rows[::-1]]
            bench_ret = (bench_prices[-1] - bench_prices[0]) / bench_prices[0] * 100

        # 批量获取当日所有个股数据(只需1次查询)
        all_stocks = conn.execute(
            f"SELECT ts_code, close, amount, pct_chg FROM stock_daily WHERE trade_date='{latest_date}'"
        ).fetchall()

        scored = []
        for code, price, amount, pct_chg in all_stocks:
            if code not in stock_set:
                continue

            try:
                # 获取个股历史数据(21天 + 5天 = 26天足够)
                hist = conn.execute(
                    f"SELECT trade_date, close, vol, amount FROM stock_daily WHERE ts_code='{code}' AND trade_date<='{latest_date}' ORDER BY trade_date DESC LIMIT 26"
                ).fetchall()
                if len(hist) < 21:
                    continue

                df = pd.DataFrame(hist[::-1], columns=['trade_date', 'close', 'vol', 'amount'])
                close_arr = df['close'].values.astype(float)
                vol_arr = df['vol'].values.astype(float)
                amount_arr = df['amount'].values.astype(float)

                # 因子1: 超额收益 vs 板块
                stock_ret_20 = (close_arr[-1] - close_arr[0]) / close_arr[0] * 100
                excess_ret = stock_ret_20 - bench_ret

                # 因子2: 动量强度(20日收益的绝对值评分)
                momentum_raw = stock_ret_20

                # 因子3: 量价配合 - 5日均量/20日均量
                vol_5 = np.mean(vol_arr[-5:]) if len(vol_arr) >= 5 else 1
                vol_20 = np.mean(vol_arr[-20:]) if len(vol_arr) >= 20 else 1
                vol_ratio = (vol_5 / vol_20) if vol_20 > 0 else 1

                # 因子4: 趋势位置 - 收盘价偏离MA20
                ma20 = np.mean(close_arr[-20:])
                price_dev = (close_arr[-1] - ma20) / ma20 * 100 if ma20 > 0 else 0

                # 因子5: 流动性 - 20日均成交额
                avg_amount = np.mean(amount_arr[-20:])

                # 归一化评分(每个因子0-10)
                def norm(val, good_high=True, cap=10):
                    # 简单的线性映射,以经验值为基准
                    if good_high:
                        return max(0, min(cap, val + cap/2))
                    else:
                        return max(0, min(cap, cap/2 - val))

                score_excess = max(0, min(10, (excess_ret + 10) / 2))  # -10% ~ +10% → 0~10
                score_momentum = max(0, min(10, (momentum_raw + 5) / 2))  # -5% ~ +15% → 0~10
                score_volume = max(0, min(10, (vol_ratio - 0.5) * 5))  # 0.5~2.5 → 0~10
                score_trend = max(0, min(10, (price_dev + 5) / 2))  # -5% ~ +15% → 0~10
                score_liquidity = max(0, min(10, np.log10(avg_amount + 1) / 8 * 10))  # 成交额对数分

                # 综合评分
                total = (score_excess * 0.25 + score_momentum * 0.20
                         + score_volume * 0.20 + score_trend * 0.20
                         + score_liquidity * 0.15)

                # 动量趋势方向(用于做多/做空判断)
                above_ma20 = close_arr[-1] > ma20
                ret_5d = (close_arr[-1] - close_arr[-6]) / close_arr[-6] * 100 if len(close_arr) >= 6 else 0

                stock_name = name_map.get(code, code)
                scored.append({
                    "code": code,
                    "name": stock_name,
                    "price": round(float(close_arr[-1]), 2),
                    "sector": l2_code,
                    "score": round(total, 1),
                    "excess_ret": round(excess_ret, 2),
                    "momentum_20d": round(stock_ret_20, 2),
                    "bench_ret": round(bench_ret, 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "price_dev_pct": round(price_dev, 2),
                    "above_ma20": bool(above_ma20),
                    "ret_5d": round(ret_5d, 2),
                    "avg_amount": round(float(avg_amount) / 1e8, 2),
                    "factors": {
                        "excess": round(score_excess, 1),
                        "momentum": round(score_momentum, 1),
                        "volume": round(score_volume, 1),
                        "trend": round(score_trend, 1),
                        "liquidity": round(score_liquidity, 1),
                    }
                })
            except Exception:
                continue

        # 按综合评分排序取前N
        scored.sort(key=lambda x: -x['score'])
        all_picks.extend(scored[:5])  # 每个板块取5只

    conn.close()

    # 全局排序
    all_picks.sort(key=lambda x: -x['score'])
    top25 = all_picks[:25]

    # 缓存风险事件并为每只个股添加风险提示
    # 个股风险事件（可选，API已变更时跳过）
    try:
        from analysis.stock_events import ensure_events, get_warnings_text
        ensure_events(db_path, codes, timeout=10)
        for p in top25:
            try:
                p["risk_warnings"] = get_warnings_text(db_path, p["code"])
            except Exception:
                p["risk_warnings"] = ""
    except Exception as e:
        logger.info("风险事件跳过（不影响选股）: %s", e)

    return top25  # 最多25只(5板块×5只)


def compute_l2_technical_signals(db_path: str, latest_date: str = None) -> dict:
    """
    主入口: 对所有123个L2行业计算技术指标并分区

    Returns:
        zones: dict of zone → [list of sector info]
        details: 每个L2代码的完整指标
    """
    import sqlite3
    conn = sqlite3.connect(db_path)

    if latest_date is None:
        latest_date = conn.execute("SELECT MAX(trade_date) FROM sw_l2_index_daily").fetchone()[0]

    all_codes = conn.execute(
        "SELECT DISTINCT ts_code, name FROM sw_l2_index_daily WHERE trade_date=? ORDER BY name",
        (latest_date,)
    ).fetchall()

    zones = {
        "chase": [],      # 🔥 追涨区 — RSI>50 MACD>0 量价配合
        "watch": [],      # ⏳ 观察区 — 趋势中性,等机会
        "top_warn": [],   # ⚠️ 高位区 — RSI>70或顶背离/顶部K线
        "weak": [],       # ❌ 弱势区 — RSI<40 趋势向下
    }
    details = {}

    for code, name in all_codes:
        try:
            df = pd.read_sql(
                f"SELECT trade_date, open, high, low, close, vol, amount "
                f"FROM sw_l2_index_daily WHERE ts_code='{code}' "
                f"AND trade_date <= '{latest_date}' ORDER BY trade_date",
                conn
            )
            if len(df) < 60:
                continue

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            open_p = df["open"].astype(float)
            vol = df["vol"].astype(float)

            # ── 指标计算 ──
            rsi = compute_rsi(close)
            macd = compute_macd(close)
            rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            dif_now = float(macd["dif"].iloc[-1])
            hist_now = float(macd["hist"].iloc[-1])
            hist_prev = float(macd["hist"].iloc[-2]) if len(macd["hist"]) >= 2 else 0

            # MACD 顶背离检测
            macd_divergence = False
            lookback = min(TOP_SIGNAL_LOOKBACK, len(close) - 1)
            recent = close.iloc[-lookback:]
            recent_hist = macd["hist"].iloc[-lookback:]
            price_peak_idx = recent.idxmax()
            price_peak_pos = recent.index.get_loc(price_peak_idx)
            hist_at_peak = recent_hist.iloc[price_peak_pos] if price_peak_pos < len(recent_hist) else 0
            if price_peak_pos > 0 and price_peak_pos < len(recent_hist) - 1:
                # 价格新高但MACD柱没新高
                later_hist_max = recent_hist.iloc[price_peak_pos + 1:].max()
                current_price = float(close.iloc[-1])
                peak_price = float(recent.max())
                if current_price >= peak_price * 0.98 and later_hist_max < hist_at_peak:
                    macd_divergence = True

            # 量价关系
            vol_5 = vol.iloc[-5:].mean()
            vol_20 = vol.iloc[-20:].mean()
            vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1
            price_chg_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) >= 6 else 0
            price_chg_20d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) >= 21 else 0

            # 价涨量缩预警
            vol_shrinking = vol_ratio < 0.8 and price_chg_5d > 0

            # 蜡烛图顶部信号
            candles = detect_top_candle(high, low, close, open_p, len(df) - 1)

            # ── K线统计 ──
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
            above_ma20 = close.iloc[-1] > ma20
            above_ma60 = close.iloc[-1] > ma60

            # ── 分区判定 ──
            has_top_candle = any("黄昏星" in s[1] or "乌云" in s[1] for s in candles)

            # 收集信号
            signals = []
            score = 0  # 综合评分,越高越强

            if rsi_now > 70:
                signals.append(f"RSI={rsi_now:.0f}(超买)")
                score -= 2
            elif rsi_now > 60:
                signals.append(f"RSI={rsi_now:.0f}(偏强)")
                score += 2
            elif rsi_now > 50:
                signals.append(f"RSI={rsi_now:.0f}(中性偏强)")
                score += 1
            elif rsi_now > 40:
                signals.append(f"RSI={rsi_now:.0f}(中性)")
            elif rsi_now > 30:
                signals.append(f"RSI={rsi_now:.0f}(偏弱)")
                score -= 1
            else:
                signals.append(f"RSI={rsi_now:.0f}(超卖)")
                score -= 1

            if dif_now > 0 and hist_now > 0:
                signals.append("MACD多头")
                score += 2
            elif dif_now < 0 and hist_now < 0:
                signals.append("MACD空头")
                score -= 2
            elif dif_now > 0 > hist_now:
                signals.append("MACD柱缩短(可能死叉)")
                score -= 1

            if macd_divergence:
                signals.append("⚠️ MACD顶背离")
                score -= 3

            if has_top_candle:
                signals.append(f"{candles[0][1]}")
                score -= 3

            if vol_shrinking:
                signals.append("量缩价涨(⚠️)")
                score -= 1
            elif vol_ratio > 1.3 and price_chg_5d > 0:
                signals.append("量增价涨(✅)")
                score += 2

            if above_ma20 and above_ma60:
                score += 2

            # 20日动量
            if price_chg_20d > 10:
                signals.append(f"20日+{price_chg_20d:.0f}%(过热)")
                score -= 1
            elif price_chg_20d > 5:
                signals.append(f"20日+{price_chg_20d:.0f}%(强势)")
                score += 3
            elif price_chg_20d > 2:
                signals.append(f"20日+{price_chg_20d:.0f}%")
                score += 1
            elif price_chg_20d < -10:
                signals.append(f"20日{price_chg_20d:.0f}%(超跌)")
                score += 1  # 超跌反弹机会
            elif price_chg_20d < -5:
                signals.append(f"20日{price_chg_20d:.0f}%(弱势)")
                score -= 2

            # ── 最终分区 ──
            if score >= 4 and not has_top_candle and not macd_divergence:
                zone = "chase"
            elif score <= -3 or macd_divergence or has_top_candle:
                zone = "top_warn"
            elif score <= -1 or rsi_now < 40:
                zone = "weak"
            else:
                zone = "watch"

            # 修正: RSI>80即使高评分也归入高位
            if rsi_now > 80:
                zone = "top_warn"

            # ATR 和价格位
            atr_val = float(compute_atr(high, low, close).iloc[-1]) if len(close) >= 15 else float(close.iloc[-1]) * 0.02
            price_levels = compute_price_levels(float(close.iloc[-1]), atr_val, above_ma20, above_ma60)

            entry = {
                "code": code, "name": name,
                "price": round(float(close.iloc[-1]), 2),
                "rsi": round(rsi_now, 1),
                "macd_dif": round(dif_now, 4),
                "macd_hist": round(hist_now, 4),
                "ma20": round(float(ma20), 2),
                "ma60": round(float(ma60), 2),
                "chg_5d": round(price_chg_5d, 2),
                "chg_20d": round(price_chg_20d, 2),
                "vol_ratio": round(vol_ratio, 2),
                "above_ma20": above_ma20,
                "above_ma60": above_ma60,
                "score": score,
                "signals": signals,
                "divergence": macd_divergence,
                "top_candle": has_top_candle,
                "zone": zone,
                **price_levels,
            }
            details[code] = entry
            zones[zone].append(entry)

        except Exception as e:
            logger.warning(f"L2 {code} {name} 计算失败: {e}")
            continue

    conn.close()

    # 各区内按评分排序
    for z in zones:
        zones[z].sort(key=lambda x: -x["score"])

    # 排除金融类板块 (银行/证券/保险/信托等)
    FINANCIAL_KEYWORDS = ['银行', '证券', '保险', '信托', '金融', '信贷']
    chase_filtered = [s for s in zones["chase"] if not any(kw in s['name'] for kw in FINANCIAL_KEYWORDS)]
    logger.info("追涨区排除金融板块后: %d → %d 个", len(zones["chase"]), len(chase_filtered))

    # 精选个股: 从非金融追涨区挑出具体股票
    chase_codes = [s["code"] for s in chase_filtered]
    stock_picks = analyze_sector_stocks(db_path, chase_codes, latest_date, top_n=5)

    logger.info(
        "L2技术指标: 🔥追涨%d ⏳观察%d ⚠️高位%d ❌弱势%d / 共%d | 个股%d只",
        len(zones["chase"]), len(zones["watch"]),
        len(zones["top_warn"]), len(zones["weak"]),
        len(details), len(stock_picks)
    )

    return {
        "date": latest_date,
        "zones": zones,
        "details": details,
        "total_scanned": len(details),
        "stock_picks": stock_picks,
    }
