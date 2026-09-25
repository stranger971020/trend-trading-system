# -*- coding: utf-8 -*-
"""Index Risk Gauge 共享模块: 组件信号 + 组合评分 + 风险分级。

被 `backtest/index_risk_gauge_backtest.py`(回测) 与 `backtest/v6_daily_report.py`(日报)
共用同一套组件定义与阈值, 避免两处漂移。

回测结论 (2015-01 ~ 2026-08, 2826 交易日):
- 事件 = 上证单日≤-2.5% OR 深成/创业板单日≤-4% OR 未来5日累计≤-3%
- 单组件均不过 G1(lift≥2), 最强 F_弱势领跌 lift≈1.92
- **组合 score≥4 → 未来3日崩盘: lift 2.25, 分半 3.00/1.52, G3 97触发, G4 4/5, G5 误报51.5% → G1-G5 全过**

— ── Changelog ──
# 2026-08-19 Claude: 新建 Index Risk Gauge 共享模块 (Part B, 计划 enumerated-imagining-bunny)
#               组件 A-F + 等权组合评分; HIGH_THRESHOLD=4 由回测 G1-G5 标定
# ─────────────
"""
import numpy as np
import pandas as pd

# 组件定义 (顺序即权重; 回测后等权已足够过门槛)
RISK_COMPONENTS = [
    ("A_MA破位",    "上证跌破MA20且MA50"),
    ("B_深度回撤",  "60日高点回撤≥8%"),
    ("C_广度崩塌",  "L2行业下跌家数占比≥55%"),
    ("D_波动率扩张", "5日/20日波动率≥1.3"),
    ("E_突发破位",  "单日跌≥1.5%且波动率扩张"),
    ("F_弱势领跌",  "深成/创业板5日落后上证≥4pp"),
]

# 评分档位 (score = 触发组件数, 0-6)
HIGH_THRESHOLD = 4     # ≥4 高风险 (回测 G1-G5 全过)
EXTREME_THRESHOLD = 5  # ≥5 极高风险

LEVEL_INFO = {
    "low":     ("低", "绿", "正常波动, 维持常规仓位"),
    "medium":  ("中", "黄", "广度/趋势走弱, 适当降低仓位"),
    "high":    ("高", "橙", "多项技术破位, 建议 ≤50% 仓位"),
    "extreme": ("极高", "红", "全面恶化, 建议 ≤30% 仓位"),
}


def compute_risk_score_series(sh, sc, cyb, l2):
    """计算每日风险组件与组合评分 (按上证交易日对齐)。

    Args:
        sh: DataFrame(trade_date, close, vol) 上证, 按 trade_date 升序
        sc: DataFrame(trade_date, close) 深成
        cyb: DataFrame(trade_date, close) 创业板
        l2: DataFrame(trade_date, down_pct) L2 行业广度 (行业下跌家数占比%)

    Returns:
        DataFrame[trade_date, A_MA破位..F_弱势领跌, score] 全历史序列
        数据不足(回看期)或输入为空时返回空 DataFrame。
    """
    if sh is None or sh.empty or "close" not in sh.columns:
        return pd.DataFrame(columns=["trade_date"] + [c[0] for c in RISK_COMPONENTS] + ["score"])
    sh = sh.reset_index(drop=True)
    out = pd.DataFrame({"trade_date": sh["trade_date"]})
    close = pd.to_numeric(sh["close"], errors="coerce")
    vol = pd.to_numeric(sh["vol"], errors="coerce")

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    dd60 = (close / close.rolling(60).max() - 1) * 100
    vol_exp = close.pct_change().rolling(5).std() / close.pct_change().rolling(20).std()
    ret = close.pct_change() * 100

    # L2 广度 (位置对齐, 避免 trade_date 索引与 RangeIndex 错配)
    if l2 is not None and not l2.empty and "down_pct" in l2.columns:
        l2m = l2.set_index("trade_date")
        l2_down = pd.Series(l2m["down_pct"].reindex(out["trade_date"]).ffill().to_numpy(), index=out.index)
    else:
        l2_down = pd.Series(np.nan, index=out.index)

    # 深成/创业板 5日收益 (按 trade_date reindex 对齐 — 创业板 2010 才上市, 行数少于上证)
    def _pct5_reindex(idx_df):
        if idx_df is None or idx_df.empty or "close" not in idx_df.columns:
            return pd.Series(np.nan, index=out.index)
        s = pd.Series(pd.to_numeric(idx_df["close"], errors="coerce").to_numpy(),
                      index=idx_df["trade_date"]).sort_index()
        return pd.Series((s.pct_change(5) * 100).reindex(out["trade_date"]).to_numpy(), index=out.index)

    sc5 = _pct5_reindex(sc)
    cyb5 = _pct5_reindex(cyb)
    sh5 = close.pct_change(5) * 100

    out["A_MA破位"] = ((close < ma20) & (close < ma50)).fillna(False).astype(int)
    out["B_深度回撤"] = (dd60 <= -8).fillna(False).astype(int)
    out["C_广度崩塌"] = (l2_down >= 55).fillna(False).astype(int)
    out["D_波动率扩张"] = (vol_exp >= 1.3).fillna(False).astype(int)
    out["E_突发破位"] = ((ret <= -1.5) & (vol_exp >= 1.2)).fillna(False).astype(int)
    weak_gap = np.minimum(sc5.fillna(99).to_numpy(), cyb5.fillna(99).to_numpy()) - sh5.to_numpy()
    out["F_弱势领跌"] = (weak_gap <= -4).astype(int)

    out["score"] = out[[c[0] for c in RISK_COMPONENTS]].sum(axis=1)
    return out


def compute_l2_down_pct(raw_l2):
    """sw_l2_index_daily 原始行(trade_date, ts_code, close) → 每日 down_pct(行业下跌家数占比 %)。

    与 backtest/index_risk_gauge_backtest.load_l2_breadth 同口径 (回测校验过)。
    """
    if raw_l2 is None or raw_l2.empty:
        return pd.DataFrame(columns=["trade_date", "down_pct"])
    df = raw_l2.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["prev_close"] = df.groupby("ts_code")["close"].shift(1)
    df["ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]
    daily = df.groupby("trade_date").agg(
        down_pct=("ret", lambda x: (x < 0).sum() / len(x) * 100),
    ).reset_index().sort_values("trade_date")
    return daily


def score_to_level(score):
    """组合评分 → (level, label, color, advice)。"""
    if score >= EXTREME_THRESHOLD:
        level = "extreme"
    elif score >= HIGH_THRESHOLD:
        level = "high"
    elif score >= 1:
        level = "medium"
    else:
        level = "low"
    label, color, advice = LEVEL_INFO[level]
    return level, label, color, advice
