"""
基本面多因子模块（增强版）
- PE/PB 分位数（行业内比较）
- ROE 质量因子
- 🆕 CFO/NP 盈利质量比率（经营现金流/净利润）
- 🆕 FCF Yield 自由现金流收益率
- 🆕 DSO 应收账款周转天数（从 ar_turn 推算）
- 🆕 毛利率稳定性
- 数据缓存到 SQLite
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import tushare as ts

from config import (
    TUSHARE_TOKEN,
    DB_PATH,
    BEIJING_TZ_OFFSET,
)

logger = logging.getLogger(__name__)

# ============================================================
# 阈值配置（可导入，便于外部引用）
# ============================================================

# CFO/NP 比率阈值
CFO_NP_HEALTHY = 1.0       # >= 1.0 视为盈利质量健康
CFO_NP_WARNING = 0.8       # < 0.8 视为盈利质量预警
CFO_NP_BONUS = 0.5         # 健康时加分
CFO_NP_PENALTY = 0.3       # 预警时扣分

# FCF Yield 阈值
FCF_YIELD_HIGH = 0.05      # >= 5% 显著低估
FCF_YIELD_FAIR = 0.02      # >= 2% 合理偏低
FCF_YIELD_BONUS_HIGH = 1.0
FCF_YIELD_BONUS_FAIR = 0.5
FCF_YIELD_PENALTY = 0.3    # < 0 扣分

# DSO 阈值（行业相对后使用，绝对阈值作为兜底）
DSO_HEALTHY = 60           # < 60天 健康
DSO_ELEVATED = 90          # 60-90天 正常
DSO_WARNING = 120          # 90-120天 关注
# > 120天 红色预警
DSO_BONUS_HEALTHY = 0.3
DSO_PENALTY_ELEVATED = 0.2
DSO_PENALTY_WARNING = 0.4

# 总分上限
MAX_FUNDAMENTAL_BONUS = 5.0

# ============================================================
# 建表
# ============================================================

CREATE_FUNDA_TABLE = """
CREATE TABLE IF NOT EXISTS fundamental_cache (
    ts_code      TEXT NOT NULL,
    trade_date   TEXT NOT NULL,
    pe_ttm       REAL,
    pb           REAL,
    roe          REAL,
    roe_quarter  TEXT,
    ar_turn      REAL,
    gross_margin REAL,
    PRIMARY KEY (ts_code, trade_date)
);
"""

CREATE_FINQUAL_TABLE = """
CREATE TABLE IF NOT EXISTS financial_quality_cache (
    ts_code      TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    report_type  TEXT,
    n_income     REAL,
    cfo          REAL,
    capex        REAL,
    fcf          REAL,
    revenue      REAL,
    receivables  REAL,
    total_mv     REAL,
    updated_at   TEXT,
    PRIMARY KEY (ts_code, end_date)
);
"""


def _init_tables():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_FUNDA_TABLE)
    conn.execute(CREATE_FINQUAL_TABLE)
    # 🆕 迁移：为新字段添加列（如果旧表缺少）
    _migrate_schema(conn)
    conn.commit()
    conn.close()


def _migrate_schema(conn):
    """为旧版本数据库添加新列。"""
    cols_to_add = {
        "fundamental_cache": [
            ("ar_turn", "REAL"),
            ("gross_margin", "REAL"),
            ("total_mv", "REAL"),
        ],
    }
    for table, columns in cols_to_add.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col_name, col_type in columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                logger.info("迁移: %s 添加列 %s", table, col_name)


# ============================================================
# 数据获取
# ============================================================

def fetch_daily_basic_batch(stock_codes: list[str]) -> pd.DataFrame:
    """批量获取个股每日估值数据并缓存。

    Returns:
        DataFrame with ts_code, trade_date, pe_ttm, pb
    """
    _init_tables()
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)

    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    end_date = beijing_now.strftime("%Y%m%d")
    start_date = (beijing_now - timedelta(days=10)).strftime("%Y%m%d")

    all_data = []

    for idx, code in enumerate(stock_codes):
        # 检查缓存
        cur = conn.execute(
            "SELECT COUNT(*) FROM fundamental_cache WHERE ts_code=? AND trade_date>=?",
            (code, start_date),
        )
        if cur.fetchone()[0] >= 3:
            cur2 = conn.execute(
                "SELECT ts_code, trade_date, pe_ttm, pb, roe, roe_quarter, ar_turn, gross_margin, total_mv "
                "FROM fundamental_cache WHERE ts_code=? AND trade_date>=? ORDER BY trade_date DESC",
                (code, start_date),
            )
            cached_rows = cur2.fetchall()
            # 如果缓存的 total_mv 为 NULL（旧版本缓存），跳过缓存重新拉取
            if cached_rows and cached_rows[0][8] is not None:
                for row in cached_rows:
                    all_data.append({
                        "ts_code": row[0], "trade_date": row[1],
                        "pe_ttm": row[2], "pb": row[3],
                        "roe": row[4], "roe_quarter": row[5],
                        "ar_turn": row[6], "gross_margin": row[7],
                        "total_mv": row[8],
                    })
                continue
            # else: fall through to fresh fetch

        try:
            df = pro.daily_basic(ts_code=code, start_date=start_date, end_date=end_date,
                                 fields="ts_code,trade_date,pe_ttm,pb,total_mv")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR IGNORE INTO fundamental_cache(ts_code,trade_date,pe_ttm,pb,total_mv) VALUES (?,?,?,?,?)",
                        (code, str(row["trade_date"]), row.get("pe_ttm"), row.get("pb"), row.get("total_mv")),
                    )
                conn.commit()
                for _, row in df.iterrows():
                    all_data.append({
                        "ts_code": code, "trade_date": str(row["trade_date"]),
                        "pe_ttm": row.get("pe_ttm"), "pb": row.get("pb"),
                        "roe": None, "roe_quarter": None,
                        "ar_turn": None, "gross_margin": None,
                        "total_mv": row.get("total_mv"),
                    })
        except Exception as e:
            logger.debug("daily_basic %s 失败: %s", code, e)

        if idx < len(stock_codes) - 1:
            time.sleep(0.3)

    conn.close()
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def fetch_roe_batch(stock_codes: list[str]) -> dict[str, dict]:
    """批量获取最新 ROE、ar_turn、毛利率数据。

    Returns:
        {"000001.SZ": {"roe": 12.5, "quarter": "2026Q1", "roe_prev": 11.2,
                        "ar_turn": 3.9, "gross_margin": 49.6}, ...}
    """
    _init_tables()
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)

    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    # 使用远期 end_date 避免丢失已公告但 f_ann_date 靠后的年报
    end_date_fwd = (beijing_now + timedelta(days=365)).strftime("%Y%m%d")

    result = {}
    for idx, code in enumerate(stock_codes):
        try:
            df = pro.fina_indicator(
                ts_code=code,
                start_date="20240101",
                end_date=end_date_fwd,
                fields="ts_code,end_date,roe,roe_dt,ar_turn,grossprofit_margin",
            )
            if df is not None and not df.empty:
                df = df.sort_values("end_date", ascending=False)
                latest = df.iloc[0]
                roe = latest.get("roe")
                roe_dt = latest.get("roe_dt")

                # 🆕 DSO 用最新年报的 ar_turn（季报的 ar_turn 因累计口径偏低，DSO 会虚高）
                df_annual = df[df["end_date"].str.endswith("1231")]
                if not df_annual.empty:
                    annual_row = df_annual.iloc[0]
                    ar_turn = annual_row.get("ar_turn")
                    gross_margin = annual_row.get("grossprofit_margin")
                else:
                    # 无年报数据（新股等），回退到最新周期
                    ar_turn = latest.get("ar_turn")
                    gross_margin = latest.get("grossprofit_margin")

                # 更新缓存
                trade_date = beijing_now.strftime("%Y%m%d")
                conn.execute(
                    "UPDATE fundamental_cache SET roe=?, roe_quarter=?, ar_turn=?, gross_margin=? "
                    "WHERE ts_code=? AND trade_date=?",
                    (roe, str(latest.get("end_date", "")), ar_turn, gross_margin, code, trade_date),
                )
                conn.commit()

                # ROE 趋势：对比上一季度
                roe_prev = float(df.iloc[1].get("roe", 0)) if len(df) > 1 else None
                trend = 0
                if roe is not None and roe_prev is not None and roe_prev != 0:
                    trend = (float(roe) - roe_prev) / abs(roe_prev)

                result[code] = {
                    "roe": round(float(roe), 2) if roe is not None else None,
                    "quarter": str(latest.get("end_date", "")),
                    "roe_trend": round(trend, 2),
                    "ar_turn": round(float(ar_turn), 2) if ar_turn is not None and not (isinstance(ar_turn, float) and np.isnan(ar_turn)) else None,
                    "gross_margin": round(float(gross_margin), 2) if gross_margin is not None and not (isinstance(gross_margin, float) and np.isnan(gross_margin)) else None,
                }
        except Exception as e:
            logger.debug("fina_indicator %s 失败: %s", code, e)

        if idx < len(stock_codes) - 1:
            time.sleep(0.3)

    conn.close()
    return result


def fetch_financial_quality_batch(stock_codes: list[str]) -> dict[str, dict]:
    """批量获取年度财务质量数据（营收、净利润、现金流、CAPEX）。

    从 income 和 cashflow 表拉取最新年报数据。
    缓存到 financial_quality_cache 表，年报数据每年仅需拉取一次。

    Returns:
        {"000001.SZ": {"n_income": 25e9, "cfo": 30e9, "capex": -5e9,
                        "fcf": 25e9, "revenue": 100e9,
                        "end_date": "20251231"}, ...}
    """
    _init_tables()
    conn = sqlite3.connect(DB_PATH)
    pro = ts.pro_api(TUSHARE_TOKEN)

    beijing_now = datetime.now(timezone(timedelta(hours=BEIJING_TZ_OFFSET)))
    end_date_fwd = (beijing_now + timedelta(days=365)).strftime("%Y%m%d")

    result = {}
    for idx, code in enumerate(stock_codes):
        # 检查缓存（年报数据按 end_date 缓存，一年内有效）
        cur = conn.execute(
            "SELECT ts_code, end_date, n_income, cfo, capex, fcf, revenue, receivables "
            "FROM financial_quality_cache WHERE ts_code=? "
            "ORDER BY end_date DESC LIMIT 1",
            (code,),
        )
        row = cur.fetchone()
        if row:
            result[code] = {
                "n_income": row[2],
                "cfo": row[3],
                "capex": row[4],
                "fcf": row[5],
                "revenue": row[6],
                "receivables": row[7],
                "end_date": row[1],
                "cached": True,
            }
            continue

        n_income_val = None
        cfo_val = None
        capex_val = None
        revenue_val = None
        receivables_val = None
        latest_end_date = None

        # ---- 拉取 income（归母净利润 + 营收） ----
        try:
            df_income = pro.income(
                ts_code=code,
                start_date="20240101",
                end_date=end_date_fwd,
                fields="ts_code,end_date,report_type,n_income_attr_p,revenue",
            )
            if df_income is not None and not df_income.empty:
                # 过滤年报：report_type == '1' 且 end_date 以 1231 结尾
                annual = df_income[
                    (df_income["report_type"] == "1") &
                    (df_income["end_date"].str.endswith("1231"))
                ].drop_duplicates("end_date")
                if not annual.empty:
                    latest_inc = annual.sort_values("end_date", ascending=False).iloc[0]
                    n_income_val = latest_inc.get("n_income_attr_p")
                    revenue_val = latest_inc.get("revenue")
                    latest_end_date = str(latest_inc["end_date"])
        except Exception as e:
            logger.debug("income %s 失败: %s", code, e)

        # ---- 拉取 cashflow（经营性现金流 + CAPEX） ----
        try:
            df_cf = pro.cashflow(
                ts_code=code,
                start_date="20240101",
                end_date=end_date_fwd,
                fields="ts_code,end_date,report_type,n_cashflow_act,c_pay_acq_const_fiolta",
            )
            if df_cf is not None and not df_cf.empty:
                annual_cf = df_cf[
                    (df_cf["report_type"] == "1") &
                    (df_cf["end_date"].str.endswith("1231"))
                ].drop_duplicates("end_date")
                if not annual_cf.empty:
                    latest_cf = annual_cf.sort_values("end_date", ascending=False).iloc[0]
                    cfo_val = latest_cf.get("n_cashflow_act")
                    capex_val = latest_cf.get("c_pay_acq_const_fiolta")
                    if latest_end_date is None:
                        latest_end_date = str(latest_cf["end_date"])
        except Exception as e:
            logger.debug("cashflow %s 失败: %s", code, e)

        # ---- 拉取 balancesheet（应收账款） ----
        try:
            df_bs = pro.balancesheet(
                ts_code=code,
                start_date="20240101",
                end_date=end_date_fwd,
                fields="ts_code,end_date,report_type,accounts_receiv",
            )
            if df_bs is not None and not df_bs.empty:
                annual_bs = df_bs[
                    (df_bs["report_type"] == "1") &
                    (df_bs["end_date"].str.endswith("1231"))
                ].drop_duplicates("end_date")
                if not annual_bs.empty:
                    latest_bs = annual_bs.sort_values("end_date", ascending=False).iloc[0]
                    receivables_val = latest_bs.get("accounts_receiv")
        except Exception as e:
            logger.debug("balancesheet %s 失败: %s", code, e)

        # ---- 计算 FCF ----
        fcf_val = None
        if cfo_val is not None:
            fcf_val = float(cfo_val)
            if capex_val is not None:
                fcf_val += float(capex_val)  # CAPEX 在 cashflow 表中为负值

        # ---- 缓存 ----
        if latest_end_date is not None:
            conn.execute(
                "INSERT OR REPLACE INTO financial_quality_cache "
                "(ts_code, end_date, report_type, n_income, cfo, capex, fcf, revenue, receivables, total_mv, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    code,
                    latest_end_date,
                    "1",
                    n_income_val,
                    cfo_val,
                    capex_val,
                    fcf_val,
                    revenue_val,
                    receivables_val,
                    None,  # total_mv 在 compute_fundamental_score 中单独计算
                    beijing_now.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()

        result[code] = {
            "n_income": n_income_val,
            "cfo": cfo_val,
            "capex": capex_val,
            "fcf": fcf_val,
            "revenue": revenue_val,
            "receivables": receivables_val,
            "end_date": latest_end_date,
            "cached": False,
        }

        if idx < len(stock_codes) - 1:
            time.sleep(0.35)

    conn.close()
    return result


# ============================================================
# DSO 计算工具
# ============================================================

def _compute_dso(ar_turn) -> float | None:
    """从应收账款周转率计算 DSO（天数）。

    DSO = 365 / ar_turn
    返回 None 如果 ar_turn 无效或为 0。
    """
    if ar_turn is None:
        return None
    try:
        val = float(ar_turn)
        if np.isnan(val) or val <= 0:
            return None
        # 极端值封顶：茅台等预收款企业 ar_turn 可达数千
        # 此时 DSO 实际趋近于 0，封顶在 5000 避免异常
        if val > 5000:
            return 0.0
        return round(365.0 / val, 1)
    except (ValueError, TypeError):
        return None


# ============================================================
# 基本面评分
# ============================================================

def compute_fundamental_score(
    stock_picks: list[dict],
    stock_mapping: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """对精选个股计算基本面加分。

    策略（增强版）:
    - PE 分位越低 → 越便宜 → 加分 (max +1.0)
    - PB 分位越低 → 越便宜 → 加分 (max +0.5)
    - ROE 越高 → 质量越好 → 加分 (max +1.0)
    - ROE 趋势向上 → +0.3
    - 🆕 CFO/NP >= 1.0 → 盈利质量健康 (+0.5); < 0.8 → 预警 (-0.3)
    - 🆕 FCF Yield >= 5% → 显著低估 (+1.0); >= 2% → 合理偏低 (+0.5); < 0 → 扣分 (-0.3)
    - 🆕 DSO < 60天 → 健康 (+0.3); > 120天 → 红色预警 (-0.4)

    总分上限: MAX_FUNDAMENTAL_BONUS (5.0)
    """
    if not stock_picks:
        return stock_picks

    codes = [p["ts_code"] for p in stock_picks]

    # 获取估值数据
    df_val = fetch_daily_basic_batch(codes)
    roe_data = fetch_roe_batch(codes)

    # 🆕 获取财务质量数据（现金流 + 营收）
    finqual_data = fetch_financial_quality_batch(codes)

    if df_val.empty and not roe_data and not finqual_data:
        return stock_picks

    # ---- PE/PB 分位数（L3 行业内） ----
    pe_pct = {}
    pb_pct = {}
    if not df_val.empty and stock_mapping:
        latest = df_val.sort_values("trade_date").groupby("ts_code").last().reset_index()
        # 获取 total_mv（万元）用于 FCF Yield
        total_mv_map = {}
        for code in latest["ts_code"]:
            row = latest[latest["ts_code"] == code]
            if not row.empty:
                mv = row["total_mv"].values[0] if "total_mv" in row.columns else None
                if mv is not None and not (isinstance(mv, float) and np.isnan(mv)):
                    total_mv_map[code] = float(mv)

        for code in latest["ts_code"]:
            l3 = stock_mapping.get(code, {}).get("l3_code", "")
            if not l3:
                continue
            l3_codes = [c for c, m in stock_mapping.items() if m.get("l3_code") == l3]
            l3_data = latest[latest["ts_code"].isin(l3_codes)]

            pe_vals = l3_data["pe_ttm"].dropna()
            pb_vals = l3_data["pb"].dropna()

            row = latest[latest["ts_code"] == code]
            if not row.empty and len(pe_vals) > 1:
                pe_val = row["pe_ttm"].values[0]
                if not pd.isna(pe_val) and pe_val > 0:
                    pe_pct[code] = (pe_vals < pe_val).sum() / len(pe_vals)

            if not row.empty and len(pb_vals) > 1:
                pb_val = row["pb"].values[0]
                if not pd.isna(pb_val) and pb_val > 0:
                    pb_pct[code] = (pb_vals < pb_val).sum() / len(pb_vals)
    else:
        total_mv_map = {}

    # ---- 应用加分 ----
    for pick in stock_picks:
        code = pick["ts_code"]
        bonus = 0.0
        extra_info = {}

        # ---- PE 分位 ----
        if code in pe_pct:
            pct = pe_pct[code]
            pe_bonus = round((1 - pct) * 1.0, 2)
            bonus += pe_bonus
            extra_info["pe_pct"] = round(pct * 100)

        # ---- PB 分位 ----
        if code in pb_pct:
            pct = pb_pct[code]
            pb_bonus = round((1 - pct) * 0.5, 2)
            bonus += pb_bonus
            extra_info["pb_pct"] = round(pct * 100)

        # ---- ROE ----
        roe_info = roe_data.get(code, {})
        roe = roe_info.get("roe")
        if roe is not None and roe > 0:
            roe_bonus = round(min(1.0, roe / 15 * 1.0), 2)
            bonus += roe_bonus
            extra_info["roe"] = roe

            trend = roe_info.get("roe_trend", 0)
            if trend > 0.05:
                bonus += 0.3
                extra_info["roe_trend"] = "↑"
            elif trend < -0.05:
                extra_info["roe_trend"] = "↓"
            else:
                extra_info["roe_trend"] = "→"

        # ---- 🆕 CFO/NP 盈利质量 ----
        fq = finqual_data.get(code, {})
        n_income = fq.get("n_income")
        cfo = fq.get("cfo")

        if n_income is not None and cfo is not None and n_income != 0:
            cfo_np_ratio = round(float(cfo) / float(n_income), 2)
            extra_info["cfo_np"] = cfo_np_ratio

            if n_income > 0:
                if cfo_np_ratio >= CFO_NP_HEALTHY:
                    bonus += CFO_NP_BONUS
                    extra_info["cfo_np_flag"] = "✅"
                elif cfo_np_ratio < CFO_NP_WARNING:
                    bonus -= CFO_NP_PENALTY
                    extra_info["cfo_np_flag"] = "🚩"
                else:
                    extra_info["cfo_np_flag"] = "⚪"
            else:
                # 净利润为负：如 CFO > 0 则不加不扣，CFO < 0 轻微扣分
                if cfo > 0:
                    extra_info["cfo_np_flag"] = "⚪"
                else:
                    bonus -= 0.2
                    extra_info["cfo_np_flag"] = "🚩"
        elif cfo is not None and n_income is None:
            extra_info["cfo_np"] = None
        elif n_income is not None and n_income == 0:
            extra_info["cfo_np"] = None

        # ---- 🆕 FCF Yield ----
        fcf = fq.get("fcf")
        total_mv_yuan = None
        if code in total_mv_map:
            # total_mv 单位是万元，转为元
            total_mv_yuan = total_mv_map[code] * 10000

        if fcf is not None and total_mv_yuan is not None and total_mv_yuan > 0:
            fcf_yield = round(fcf / total_mv_yuan, 4)
            extra_info["fcf_yield"] = fcf_yield

            if fcf_yield >= FCF_YIELD_HIGH:
                bonus += FCF_YIELD_BONUS_HIGH
                extra_info["fcf_flag"] = "✅"
            elif fcf_yield >= FCF_YIELD_FAIR:
                bonus += FCF_YIELD_BONUS_FAIR
                extra_info["fcf_flag"] = "⚪"
            elif fcf_yield < 0:
                bonus -= FCF_YIELD_PENALTY
                extra_info["fcf_flag"] = "🚩"
            else:
                extra_info["fcf_flag"] = "⚪"
        elif fcf is not None:
            extra_info["fcf"] = round(fcf / 1e8, 2)  # 亿为单位供参考
        elif total_mv_yuan is not None:
            pass  # 无 FCF 数据

        # ---- 🆕 DSO（应收账款周转天数） ----
        ar_turn = roe_info.get("ar_turn")
        dso = _compute_dso(ar_turn)
        if dso is not None:
            extra_info["dso"] = dso

            if dso <= DSO_HEALTHY:
                bonus += DSO_BONUS_HEALTHY
                extra_info["dso_flag"] = "✅"
            elif dso <= DSO_ELEVATED:
                extra_info["dso_flag"] = "⚪"
            elif dso <= DSO_WARNING:
                bonus -= DSO_PENALTY_ELEVATED
                extra_info["dso_flag"] = "⚠️"
            else:
                bonus -= DSO_PENALTY_WARNING
                extra_info["dso_flag"] = "🚩"

        # ---- 🆕 毛利率 ----
        gross_margin = roe_info.get("gross_margin")
        if gross_margin is not None:
            extra_info["gross_margin"] = gross_margin

        # ---- 总分封顶 ----
        bonus = min(MAX_FUNDAMENTAL_BONUS, max(-2.0, bonus))
        extra_info["funda_bonus"] = round(bonus, 2)

        old_score = pick.get("score", 0)
        pick["score"] = round(old_score + bonus, 2)
        pick["fundamental"] = extra_info

    # 统计
    cfo_np_count = sum(1 for p in stock_picks if p.get("fundamental", {}).get("cfo_np") is not None)
    fcf_y_count = sum(1 for p in stock_picks if p.get("fundamental", {}).get("fcf_yield") is not None)
    dso_count = sum(1 for p in stock_picks if p.get("fundamental", {}).get("dso") is not None)
    logger.info(
        "基本面因子: %d 只个股评分已调整 | "
        "CFO/NP覆盖=%d FCF Yield覆盖=%d DSO覆盖=%d",
        len(stock_picks), cfo_np_count, fcf_y_count, dso_count,
    )

    return stock_picks
