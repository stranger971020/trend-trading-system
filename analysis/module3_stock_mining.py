"""
模块3: 个股精选
- 从高持续性（>=7）和中等持续性（>=5）的 SW L1 行业中选股
- 个股得分 = 超额收益 + 5日动量 + MA20偏离
- 每行业输出 TOP_N 精选个股
"""

import logging

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_LOOKBACK,
    MODULE3_TOP_N,
    STOCK_SCORE_WEIGHTS,
    HIGH_PERSISTENCE,
    MEDIUM_PERSISTENCE,
    COL_TRADE_DATE,
    COL_TS_CODE,
    COL_CLOSE,
    COL_PCT_CHG,
    COL_NAME,
)

logger = logging.getLogger(__name__)

# 不选股的行业（保留在板块分析中，但不纳入个股推荐）
EXCLUDED_KEYWORDS = ["银行", "证券", "保险", "城商行", "金融", "信托", "期货"]
TOP_N_INDUSTRIES = 20      # 从持续性排名前 N 的行业中选股
MIN_L3_STOCKS = 8          # L3 行业最少成分股数（过滤微型行业）


# ============================================================
# 个股评分计算
# ============================================================

def compute_excess_return(
    stock_prices: pd.Series,
    industry_prices: pd.Series,
    lookback: int = MOMENTUM_LOOKBACK,
) -> float:
    """计算个股相对行业的超额收益（百分比）。"""
    if len(stock_prices) < lookback + 1 or len(industry_prices) < lookback + 1:
        return 0.0

    stock_ret = (stock_prices.iloc[-1] / stock_prices.iloc[-(lookback + 1)] - 1) * 100
    ind_ret = (industry_prices.iloc[-1] / industry_prices.iloc[-(lookback + 1)] - 1) * 100
    return stock_ret - ind_ret


def compute_momentum_5d(stock_prices: pd.Series) -> float:
    """计算近5日动量（百分比）。"""
    if len(stock_prices) < 6:
        return 0.0
    return (stock_prices.iloc[-1] / stock_prices.iloc[-6] - 1) * 100


def compute_ma20_deviation(stock_prices: pd.Series) -> float:
    """计算 MA20 偏离度（百分比）——正值表示突破均线。"""
    if len(stock_prices) < MOMENTUM_LOOKBACK:
        return 0.0
    ma20 = stock_prices.rolling(window=MOMENTUM_LOOKBACK, min_periods=MOMENTUM_LOOKBACK).mean()
    close = stock_prices.iloc[-1]
    ma20_val = ma20.iloc[-1]
    if ma20_val <= 0:
        return 0.0
    return (close - ma20_val) / ma20_val * 100


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max 归一化到 [0, 10]。"""
    s_min, s_max = series.min(), series.max()
    if s_max - s_min < 1e-10:
        return pd.Series(5.0, index=series.index)
    return (series - s_min) / (s_max - s_min) * 10.0


# ============================================================
# 主分析函数
# ============================================================

def analyze_stocks(
    stock_daily_df: pd.DataFrame | None = None,
    industry_daily_df: pd.DataFrame | None = None,
    persistence_result: dict | None = None,
    stock_mapping: dict[str, dict[str, str]] | None = None,
    industry_mapping: dict[str, str] | None = None,
) -> dict:
    """从高/中等持续性行业中精选个股。

    Args:
        stock_daily_df: 个股日线数据
        industry_daily_df: 行业日线数据
        persistence_result: 模块2的输出
        stock_mapping: 个股→行业映射
        industry_mapping: 行业代码→名称映射

    Returns:
        dict: {
            "status": "success" | "degraded" | "skipped" | "failed",
            "stocks": [{"ts_code": "000001.SZ", "name": "平安银行", ...}, ...],
            "by_industry": {"银行": [...], ...},
        }
    """
    result = {
        "status": "skipped",
        "stocks": [],
        "by_industry": {},
        "error": None,
    }

    # ---- 检查前置条件 ----
    if persistence_result is None or persistence_result.get("status") != "success":
        result["reason"] = "板块持续性分析不可用"
        logger.info("模块3: 跳过（板块持续性分析不可用）")
        return result

    if stock_daily_df is None or stock_daily_df.empty:
        result["reason"] = "个股日线数据为空，请先运行个股数据更新"
        logger.info("模块3: 跳过（个股日线数据为空）")
        return result

    if industry_daily_df is None or industry_daily_df.empty:
        result["reason"] = "行业日线数据为空"
        logger.info("模块3: 跳过（行业日线数据为空）")
        return result

    if stock_mapping is None:
        result["reason"] = "个股行业映射为空"
        logger.info("模块3: 跳过（个股行业映射为空）")
        return result

    try:
        # ---- 确定目标行业 ----
        persistence_df = persistence_result.get("df")
        if persistence_df is None or persistence_df.empty:
            result["reason"] = "持续性评分为空"
            return result

        # 取持续性排名前 TOP_N_INDUSTRIES 的行业
        target = persistence_df.nlargest(TOP_N_INDUSTRIES, "persistence_score")
        if target.empty:
            result["status"] = "degraded"
            result["reason"] = f"持续性评分为空"
            return result

        target_codes = set(target["ts_code"].tolist())
        logger.info(
            "模块3: Top-%d 行业（持续性 %.2f ~ %.2f）",
            len(target_codes),
            target["persistence_score"].max(),
            target["persistence_score"].min(),
        )

        # ---- 构建行业指数 ----
        # 为每个目标行业计算行业指数（所有成分股平均价格）
        industry_index: dict[str, pd.Series] = {}
        for l1_code in target_codes:
            ind_df = industry_daily_df[industry_daily_df["ts_code"] == l1_code].sort_values(COL_TRADE_DATE)
            if not ind_df.empty and len(ind_df) >= MOMENTUM_LOOKBACK + 1:
                industry_index[l1_code] = ind_df[COL_CLOSE]

        # ---- 按行业分组选股 ----
        w = STOCK_SCORE_WEIGHTS
        all_picks = []
        by_industry = {}

        for l1_code in sorted(target_codes):
            # 获取该行业下的个股代码
            industry_stocks = [
                code for code, info in stock_mapping.items()
                if info.get("l1_code") == l1_code
            ]
            if not industry_stocks:
                continue

            # 获取该行业个股数据
            stock_group = stock_daily_df[
                stock_daily_df[COL_TS_CODE].isin(industry_stocks)
            ]
            if stock_group.empty:
                continue

            # 获取行业价格序列
            ind_prices = industry_index.get(l1_code)
            if ind_prices is None:
                continue

            # 行业名称
            ind_name = ""
            if industry_mapping:
                ind_name = industry_mapping.get(l1_code, l1_code)

            # 对每只个股评分
            stock_scores = []
            for ts_code in stock_group[COL_TS_CODE].unique():
                s_df = stock_group[stock_group[COL_TS_CODE] == ts_code].sort_values(COL_TRADE_DATE)
                if len(s_df) < MOMENTUM_LOOKBACK + 1:
                    continue

                prices = s_df[COL_CLOSE]

                excess = compute_excess_return(prices, ind_prices)
                mom5d = compute_momentum_5d(prices)
                ma20dev = compute_ma20_deviation(prices)

                stock_scores.append({
                    "ts_code": ts_code,
                    "excess_return": excess,
                    "momentum_5d": mom5d,
                    "ma20_deviation": ma20dev,
                })

            if not stock_scores:
                continue

            # 归一化各项得分
            scores_df = pd.DataFrame(stock_scores)
            scores_df["excess_score"] = _minmax_normalize(scores_df["excess_return"])
            scores_df["momentum_score"] = _minmax_normalize(scores_df["momentum_5d"])
            scores_df["ma20_score"] = _minmax_normalize(scores_df["ma20_deviation"])

            # 加权总分
            scores_df["total_score"] = (
                scores_df["excess_score"] * w["excess_return"]
                + scores_df["momentum_score"] * w["momentum_5d"]
                + scores_df["ma20_score"] * w["ma20_deviation"]
            )

            # 排序取 TOP_N
            scores_df = scores_df.sort_values("total_score", ascending=False)
            top_n = scores_df.head(MODULE3_TOP_N)

            # 获取股票名称
            industry_picks = []
            for _, row in top_n.iterrows():
                ts_code = row["ts_code"]
                stock_name = ts_code
                if stock_mapping and ts_code in stock_mapping:
                    stock_name = stock_mapping[ts_code].get("stock_name", ts_code)

                pick = {
                    "ts_code": ts_code,
                    "name": stock_name,
                    "score": round(float(row["total_score"]), 2),
                    "excess_return": round(float(row["excess_return"]), 2),
                    "momentum_5d": round(float(row["momentum_5d"]), 2),
                    "ma20_deviation": round(float(row["ma20_deviation"]), 2),
                    "industry": ind_name,
                    "industry_code": l1_code,
                }
                industry_picks.append(pick)
                all_picks.append(pick)

            if industry_picks:
                by_industry[ind_name or l1_code] = industry_picks

        # 全局排序
        # 排除金融类个股（按行业名称关键词过滤）
        def _is_financial(name: str) -> bool:
            return any(kw in name for kw in EXCLUDED_KEYWORDS)
        all_picks = [p for p in all_picks if not _is_financial(p.get("industry", ""))]
        by_industry = {k: v for k, v in by_industry.items() if not _is_financial(k)}
        all_picks.sort(key=lambda x: x["score"], reverse=True)

        result["stocks"] = all_picks
        result["by_industry"] = by_industry
        result["status"] = "success"
        result.pop("reason", None)

        logger.info(
            "模块3: 从 %d 个行业选出 %d 只个股",
            len(by_industry), len(all_picks),
        )

    except Exception as e:
        logger.error("模块3分析失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def analyze_stocks_l3(
    stock_daily_df: pd.DataFrame | None = None,
    l3_daily_df: pd.DataFrame | None = None,
    l3_persistence_result: dict | None = None,
    stock_mapping: dict[str, dict[str, str]] | None = None,
) -> dict:
    """从高持续性三级行业中精选个股（L3 版本）。

    与 analyze_stocks 的区别：
    - 使用 L3 持续性评分而非 L1
    - 个股映射到 l3_code 而非 l1_code
    - 超额收益计算相对三级行业指数

    Args:
        stock_daily_df: 个股日线数据
        l3_daily_df: 三级行业日线数据
        l3_persistence_result: 模块2 L3 分析输出
        stock_mapping: 个股→行业映射（含 l3_code）

    Returns:
        dict: 与 analyze_stocks 相同格式
    """
    result = {
        "status": "skipped",
        "stocks": [],
        "by_industry": {},
        "error": None,
    }

    if l3_persistence_result is None or l3_persistence_result.get("status") != "success":
        result["reason"] = "L3 持续性分析不可用"
        return result

    if stock_daily_df is None or stock_daily_df.empty:
        result["reason"] = "个股日线数据为空"
        return result

    if l3_daily_df is None or l3_daily_df.empty:
        result["reason"] = "L3 行业日线数据为空"
        return result

    if stock_mapping is None:
        result["reason"] = "个股行业映射为空"
        return result

    try:
        l3_df = l3_persistence_result.get("df")
        if l3_df is None or l3_df.empty:
            result["reason"] = "L3 持续性评分为空"
            return result

        # 过滤微型行业（成分股太少）
        l3_stock_counts = {}
        if stock_mapping:
            for info in stock_mapping.values():
                l3c = info.get("l3_code", "")
                l3_stock_counts[l3c] = l3_stock_counts.get(l3c, 0) + 1

        l3_df_filtered = l3_df.copy()
        if l3_stock_counts:
            valid_codes = {c for c, n in l3_stock_counts.items() if n >= MIN_L3_STOCKS}
            before = len(l3_df_filtered)
            l3_df_filtered = l3_df_filtered[l3_df_filtered["ts_code"].isin(valid_codes)]
            logger.info("L3 过滤: %d → %d (>=%d只成分股)", before, len(l3_df_filtered), MIN_L3_STOCKS)

        target = l3_df_filtered.nlargest(TOP_N_INDUSTRIES, "persistence_score")
        if target.empty:
            result["status"] = "degraded"
            result["reason"] = f"无符合条件的 L3 行业（>=%d只成分股）" % MIN_L3_STOCKS
            return result

        target_codes = set(target["ts_code"].tolist())
        logger.info("模块3(L3): Top-%d L3 行业", len(target_codes))

        # 构建 L3 行业价格索引
        l3_prices_map = {}
        for l3_code in target_codes:
            grp = l3_daily_df[l3_daily_df[COL_TS_CODE] == l3_code].sort_values(COL_TRADE_DATE)
            if not grp.empty and len(grp) >= MOMENTUM_LOOKBACK + 1:
                l3_prices_map[l3_code] = grp[COL_CLOSE]

        w = STOCK_SCORE_WEIGHTS
        all_picks = []
        by_industry = {}

        for l3_code in sorted(target_codes):
            # 获取该 L3 行业下的个股
            industry_stocks = [
                code for code, info in stock_mapping.items()
                if info.get("l3_code") == l3_code
            ]
            if not industry_stocks:
                continue

            stock_group = stock_daily_df[stock_daily_df[COL_TS_CODE].isin(industry_stocks)]
            if stock_group.empty:
                continue

            l3_prices = l3_prices_map.get(l3_code)
            if l3_prices is None:
                continue

            # L3 行业名称
            l3_name = ""
            l3_row = l3_daily_df[l3_daily_df[COL_TS_CODE] == l3_code]
            if not l3_row.empty and COL_NAME in l3_row.columns:
                l3_name = str(l3_row[COL_NAME].iloc[0])
            if not l3_name:
                l3_name = l3_code

            # 个股评分
            stock_scores = []
            for ts_code in stock_group[COL_TS_CODE].unique():
                sdf = stock_group[stock_group[COL_TS_CODE] == ts_code].sort_values(COL_TRADE_DATE)
                if len(sdf) < MOMENTUM_LOOKBACK + 1:
                    continue

                prices = sdf[COL_CLOSE]
                excess = compute_excess_return(prices, l3_prices)
                mom5d = compute_momentum_5d(prices)
                ma20dev = compute_ma20_deviation(prices)

                stock_scores.append({
                    "ts_code": ts_code,
                    "excess_return": excess,
                    "momentum_5d": mom5d,
                    "ma20_deviation": ma20dev,
                })

            if not stock_scores:
                continue

            scores_df = pd.DataFrame(stock_scores)
            scores_df["excess_score"] = _minmax_normalize(scores_df["excess_return"])
            scores_df["momentum_score"] = _minmax_normalize(scores_df["momentum_5d"])
            scores_df["ma20_score"] = _minmax_normalize(scores_df["ma20_deviation"])
            scores_df["total_score"] = (
                scores_df["excess_score"] * w["excess_return"]
                + scores_df["momentum_score"] * w["momentum_5d"]
                + scores_df["ma20_score"] * w["ma20_deviation"]
            )

            scores_df = scores_df.sort_values("total_score", ascending=False)
            top_n = scores_df.head(MODULE3_TOP_N)

            industry_picks = []
            for _, row in top_n.iterrows():
                ts_code = row["ts_code"]
                stock_name = ts_code
                if stock_mapping and ts_code in stock_mapping:
                    stock_name = stock_mapping[ts_code].get("stock_name", ts_code)

                pick = {
                    "ts_code": ts_code,
                    "name": stock_name,
                    "score": round(float(row["total_score"]), 2),
                    "excess_return": round(float(row["excess_return"]), 2),
                    "momentum_5d": round(float(row["momentum_5d"]), 2),
                    "ma20_deviation": round(float(row["ma20_deviation"]), 2),
                    "industry": l3_name,
                    "industry_code": l3_code,
                }
                industry_picks.append(pick)
                all_picks.append(pick)

            if industry_picks:
                by_industry[l3_name] = industry_picks

        # 排除金融类个股（按行业名称关键词过滤）
        def _is_financial(name: str) -> bool:
            return any(kw in name for kw in EXCLUDED_KEYWORDS)
        all_picks = [p for p in all_picks if not _is_financial(p.get("industry", ""))]
        by_industry = {k: v for k, v in by_industry.items() if not _is_financial(k)}
        all_picks.sort(key=lambda x: x["score"], reverse=True)
        result["stocks"] = all_picks
        result["by_industry"] = by_industry
        result["status"] = "success"

        logger.info("模块3(L3): 从 %d 个 L3 行业选出 %d 只个股", len(by_industry), len(all_picks))

    except Exception as e:
        logger.error("模块3 L3 分析失败: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def analyze_stocks_l2(stock_daily_df=None, l2_daily_df=None, l2_persistence_result=None, stock_mapping=None):
    """从持续性 Top-7 L2 行业中精选个股（L2 版本）。

    与 analyze_stocks_l3 逻辑相同，但使用 l2_code 而非 l3_code。
    """
    result = {"status": "skipped", "stocks": [], "by_industry": {}, "error": None}
    if l2_persistence_result is None or l2_persistence_result.get("status") != "success":
        result["reason"] = "L2 持续性分析不可用"; return result
    if stock_daily_df is None or stock_daily_df.empty:
        result["reason"] = "个股数据为空"; return result
    if l2_daily_df is None or l2_daily_df.empty:
        result["reason"] = "L2 数据为空"; return result
    if stock_mapping is None:
        result["reason"] = "映射为空"; return result
    try:
        l2df = l2_persistence_result.get("df")
        if l2df is None or l2df.empty: result["reason"] = "L2 持续性为空"; return result
        # 过滤微型行业
        l2_stock_counts = {}
        for info in stock_mapping.values():
            l2c = info.get("l2_code", "")
            l2_stock_counts[l2c] = l2_stock_counts.get(l2c, 0) + 1
        valid_codes = {c for c, n in l2_stock_counts.items() if n >= MIN_L3_STOCKS}
        l2df = l2df[l2df["ts_code"].isin(valid_codes)]
        target = l2df.nlargest(TOP_N_INDUSTRIES, "persistence_score")
        if target.empty: result["status"] = "degraded"; result["reason"] = "无符合条件的L2行业"; return result
        target_codes = set(target["ts_code"].tolist())
        logger.info("模块3(L2): Top-%d L2 行业", len(target_codes))
        l2_prices_map = {}
        for code in target_codes:
            grp = l2_daily_df[l2_daily_df["ts_code"] == code].sort_values("trade_date")
            if not grp.empty and len(grp) >= MOMENTUM_LOOKBACK + 1:
                l2_prices_map[code] = grp["close"]
        w = STOCK_SCORE_WEIGHTS
        all_picks = []; by_industry = {}
        for l2_code in sorted(target_codes):
            industry_stocks = [c for c, info in stock_mapping.items() if info.get("l2_code") == l2_code]
            if not industry_stocks: continue
            stock_group = stock_daily_df[stock_daily_df["ts_code"].isin(industry_stocks)]
            if stock_group.empty: continue
            l2_prices = l2_prices_map.get(l2_code)
            if l2_prices is None: continue
            l2_name = str(l2_daily_df[l2_daily_df["ts_code"] == l2_code]["name"].iloc[0]) if not l2_daily_df[l2_daily_df["ts_code"] == l2_code].empty else l2_code
            stock_scores = []
            for ts_code in stock_group["ts_code"].unique():
                sdf = stock_group[stock_group["ts_code"] == ts_code].sort_values("trade_date")
                if len(sdf) < MOMENTUM_LOOKBACK + 1: continue
                prices = sdf["close"]
                excess = compute_excess_return(prices, l2_prices)
                mom5d = compute_momentum_5d(prices)
                ma20dev = compute_ma20_deviation(prices)
                stock_scores.append({"ts_code": ts_code, "excess_return": excess, "momentum_5d": mom5d, "ma20_deviation": ma20dev})
            if not stock_scores: continue
            sdf2 = pd.DataFrame(stock_scores)
            sdf2["excess_score"] = _minmax_normalize(sdf2["excess_return"])
            sdf2["momentum_score"] = _minmax_normalize(sdf2["momentum_5d"])
            sdf2["ma20_score"] = _minmax_normalize(sdf2["ma20_deviation"])
            sdf2["total_score"] = sdf2["excess_score"]*w["excess_return"] + sdf2["momentum_score"]*w["momentum_5d"] + sdf2["ma20_score"]*w["ma20_deviation"]
            sdf2 = sdf2.sort_values("total_score", ascending=False)
            top_n = sdf2.head(MODULE3_TOP_N)
            industry_picks = []
            for _, row in top_n.iterrows():
                ts_code = row["ts_code"]
                stock_name = stock_mapping[ts_code].get("stock_name", ts_code) if ts_code in stock_mapping else ts_code
                pick = {"ts_code": ts_code, "name": stock_name, "score": round(float(row["total_score"]), 2),
                        "excess_return": round(float(row["excess_return"]), 2),
                        "momentum_5d": round(float(row["momentum_5d"]), 2),
                        "ma20_deviation": round(float(row["ma20_deviation"]), 2),
                        "industry": l2_name, "industry_code": l2_code}
                industry_picks.append(pick); all_picks.append(pick)
            if industry_picks: by_industry[l2_name] = industry_picks
        # 过滤金融
        def _fin(n): return any(kw in n for kw in EXCLUDED_KEYWORDS)
        all_picks = [p for p in all_picks if not _fin(p.get("industry", ""))]
        by_industry = {k: v for k, v in by_industry.items() if not _fin(k)}
        all_picks.sort(key=lambda x: x["score"], reverse=True)
        result["stocks"] = all_picks; result["by_industry"] = by_industry; result["status"] = "success"
        logger.info("模块3(L2): 从 %d 个 L2 行业选出 %d 只个股", len(by_industry), len(all_picks))
    except Exception as e:
        logger.error("模块3 L2 失败: %s", e, exc_info=True)
        result["status"] = "failed"; result["error"] = str(e)
    return result
