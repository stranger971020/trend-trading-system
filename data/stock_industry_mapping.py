"""
个股↔申万一级行业映射 + 全量 A 股股票池

- 从 Tushare index_member_all 获取 SW 成分股（l1/l2/l3 完整层级）
- 从 Tushare stock_basic 获取沪深全量（剔除 ST/北交所/B 股），CSRC 行业经
  industry_to_sw_l1.csv 归并到申万一级，仍无法归并的落「未分类」
- 缓存到 CSV 供离线使用

── Changelog ──
# 2026-08-05 Claude: 股票池从申万成分股(~3000)扩为沪深全量剔除ST(~4999)
#               新增 fetch_full_market_list/build_universe_mapping/load_stock_universe
#               CSV 演进为 14 列(新增 is_sw/source/list_date/market/exchange/csrc_industry)
#               向后兼容旧 8 列格式; 未分类 l1_code="" l1_name="未分类"
#               映射表: data_storage/industry_to_sw_l1.csv (CSRC 110→申万31)
#               告警: 下游 get_stocks_by_industry 不包含未分类股票(无申万 l1_code)
─────────────
"""

import csv
import logging
import os

import tushare as ts

from config import (
    TUSHARE_TOKEN,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

STOCK_MAPPING_CSV = os.path.join(DATA_DIR, "stock_industry_mapping.csv")
SW_L1_MAPPING_CSV = os.path.join(DATA_DIR, "sw_l1_mapping.csv")
INDUSTRY_TO_SW_L1_CSV = os.path.join(DATA_DIR, "industry_to_sw_l1.csv")

UNCLASSIFIED_L1_NAME = "未分类"

_CSV_FIELDNAMES = [
    "ts_code", "stock_name", "l1_code", "l1_name",
    "l2_code", "l2_name", "l3_code", "l3_name",
    "is_sw", "source", "list_date", "market", "exchange", "csrc_industry",
]


def _is_st_name(name: str) -> bool:
    """按股票名判定 ST/*ST（含 SST）。扩池后按当前名剔除。"""
    return "ST" in name.upper()


def _load_sw_l1_name_to_code() -> dict[str, str]:
    """sw_l1_mapping.csv: 申万一级 name → l1_code（31 个行业）。"""
    result: dict[str, str] = {}
    if not os.path.exists(SW_L1_MAPPING_CSV):
        return result
    with open(SW_L1_MAPPING_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row.get("name", "")] = row.get("ts_code", "")
    return result


def _load_csrc_to_sw_l1() -> dict[str, str]:
    """industry_to_sw_l1.csv: CSRC 行业名 → 申万一级名称（110→31 策划映射）。"""
    result: dict[str, str] = {}
    if not os.path.exists(INDUSTRY_TO_SW_L1_CSV):
        return result
    with open(INDUSTRY_TO_SW_L1_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row.get("csrc_industry", "")] = row.get("sw_l1_name", "")
    return result


def _fetch_from_tushare() -> dict[str, dict[str, str]]:
    """从 Tushare 获取当前 SW 成分股映射（Tier1 申万，保留 l1/l2/l3 完整层级）。"""
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.index_member_all(index_code="801010.SI")

    if df is None or df.empty:
        raise RuntimeError("Tushare index_member_all 返回空数据")

    current = df[df["is_new"] == "Y"]

    mapping: dict[str, dict[str, str]] = {}
    for _, row in current.iterrows():
        ts_code = str(row.get("ts_code", ""))
        if not ts_code:
            continue
        mapping[ts_code] = {
            "l1_code": str(row.get("l1_code", "")),
            "l1_name": str(row.get("l1_name", "")),
            "l2_code": str(row.get("l2_code", "")),
            "l2_name": str(row.get("l2_name", "")),
            "l3_code": str(row.get("l3_code", "")),
            "l3_name": str(row.get("l3_name", "")),
            "stock_name": str(row.get("name", "")),
        }

    logger.info("从 Tushare 获取 %d 只申万成分股映射", len(mapping))
    return mapping


def fetch_full_market_list() -> list[dict]:
    """从 Tushare stock_basic 拉沪深全量上市股票，剔除 ST/*ST、北交所(.BJ)、B 股(.B)。

    Returns:
        [{ts_code, stock_name, csrc_industry, market, exchange, list_date}, ...]
    """
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.stock_basic(
        exchange="", list_status="L",
        fields="ts_code,name,industry,market,exchange,list_date",
    )
    if df is None or df.empty:
        raise RuntimeError("Tushare stock_basic 返回空数据")

    out: list[dict] = []
    for _, r in df.iterrows():
        ts_code = str(r.get("ts_code", ""))
        name = str(r.get("name", ""))
        # 剔除北交所 / B 股
        if ts_code.endswith(".BJ") or ts_code.endswith(".B"):
            continue
        # 剔除 ST/*ST
        if _is_st_name(name):
            continue
        out.append({
            "ts_code": ts_code,
            "stock_name": name,
            "csrc_industry": str(r.get("industry") or ""),
            "market": str(r.get("market") or ""),
            "exchange": str(r.get("exchange") or ""),
            "list_date": str(r.get("list_date") or ""),
        })
    logger.info("沪深全量剔除 ST/北交/B 股: %d 只", len(out))
    return out


def build_universe_mapping() -> dict[str, dict[str, str]]:
    """构建全量股票池映射（四档行业补全）。

    Tier1 申万：index_member_all 当前成分，保留 l1/l2/l3 完整层级
    Tier2 CSRC→申万：stock_basic.industry 经 industry_to_sw_l1.csv 归并到申万一级
    Tier3 东财兜底：仅 CSRC 缺失时（实测全池都有），此处留空由调用方补充
    Tier4 未分类：l1_code="" l1_name="未分类" source="unclassified"

    Returns:
        {ts_code: {l1_code, l1_name, ..., is_sw, source, list_date, market, exchange, csrc_industry}}
    """
    sw = _fetch_from_tushare()
    universe = fetch_full_market_list()

    sw_name2code = _load_sw_l1_name_to_code()
    csrc2sw = _load_csrc_to_sw_l1()

    mapping: dict[str, dict[str, str]] = {}
    for u in universe:
        code = u["ts_code"]
        is_sw = code in sw
        sw_info = sw.get(code, {})

        if is_sw:
            l1_code = sw_info.get("l1_code", "")
            l1_name = sw_info.get("l1_name", "")
            source = "tushare_sw"
        else:
            sw_l1_name = csrc2sw.get(u["csrc_industry"], "")
            l1_code = sw_name2code.get(sw_l1_name, "") if sw_l1_name else ""
            l1_name = sw_l1_name if l1_code else UNCLASSIFIED_L1_NAME
            source = "csrc" if l1_code else "unclassified"

        mapping[code] = {
            "l1_code": l1_code,
            "l1_name": l1_name if l1_code else (sw_info.get("l1_name", "") if is_sw else UNCLASSIFIED_L1_NAME),
            "l2_code": sw_info.get("l2_code", ""),
            "l2_name": sw_info.get("l2_name", ""),
            "l3_code": sw_info.get("l3_code", ""),
            "l3_name": sw_info.get("l3_name", ""),
            "stock_name": u["stock_name"],
            "is_sw": "1" if is_sw else "0",
            "source": source,
            "list_date": u["list_date"],
            "market": u["market"],
            "exchange": u["exchange"],
            "csrc_industry": u["csrc_industry"],
        }

    logger.info("全量池映射: %d 只 (申万 %d / CSRC %d / 未分类 %d)",
                len(mapping),
                sum(1 for v in mapping.values() if v["is_sw"] == "1"),
                sum(1 for v in mapping.values() if v["source"] == "csrc"),
                sum(1 for v in mapping.values() if v["source"] == "unclassified"))
    return mapping


def _save_to_csv(mapping: dict[str, dict[str, str]]) -> None:
    """保存映射到 CSV（14 列新格式）。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STOCK_MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for ts_code, info in sorted(mapping.items()):
            row = {"ts_code": ts_code, **info}
            writer.writerow(row)
    logger.info("个股映射已保存至 %s (%d 只)", STOCK_MAPPING_CSV, len(mapping))


def _load_from_csv() -> dict[str, dict[str, str]]:
    """从 CSV 缓存加载映射（兼容旧 8 列格式，缺列自动补默认）。"""
    mapping: dict[str, dict[str, str]] = {}
    with open(STOCK_MAPPING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        for row in reader:
            ts_code = row["ts_code"]
            l1_code = row.get("l1_code", "")
            # 旧格式(8列)缺 is_sw/source 时按 l1_code 推断
            is_sw = row.get("is_sw", "1" if l1_code else "0")
            source = row.get("source", "tushare_sw" if l1_code else "unclassified")
            mapping[ts_code] = {
                "l1_code": l1_code,
                "l1_name": row.get("l1_name", ""),
                "l2_code": row.get("l2_code", ""),
                "l2_name": row.get("l2_name", ""),
                "l3_code": row.get("l3_code", ""),
                "l3_name": row.get("l3_name", ""),
                "stock_name": row.get("stock_name", ""),
                "is_sw": is_sw,
                "source": source,
                "list_date": row.get("list_date", ""),
                "market": row.get("market", ""),
                "exchange": row.get("exchange", ""),
                "csrc_industry": row.get("csrc_industry", ""),
            }
    logger.info("从 CSV 缓存加载 %d 只个股映射", len(mapping))
    return mapping


def load_stock_industry_mapping(force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """加载个股→行业映射（全量池）。

    优先从 CSV 缓存读取，若不存在或 force_refresh=True 则重新构建。

    Returns:
        {ts_code: {l1_code, l1_name, ..., is_sw, source, list_date, ...}}
    """
    if not force_refresh and os.path.exists(STOCK_MAPPING_CSV):
        try:
            loaded = _load_from_csv()
            if loaded:
                return loaded
        except Exception:
            logger.warning("CSV 缓存读取失败，尝试重建")

    mapping = build_universe_mapping()
    try:
        _save_to_csv(mapping)
    except Exception:
        logger.warning("个股映射 CSV 写入失败（不影响继续运行）")
    return mapping


def load_stock_universe(force_refresh: bool = False) -> list[dict]:
    """加载全量股票池（含 list_date/market/exchange/is_st 等元信息），供回填与过滤。

    Returns:
        [{ts_code, stock_name, l1_code, l1_name, is_sw, source, list_date,
          market, exchange, csrc_industry}, ...] 按 ts_code 排序
    """
    mapping = load_stock_industry_mapping(force_refresh=force_refresh)
    return [{"ts_code": code, **info} for code, info in sorted(mapping.items())]


def get_stocks_by_industry(
    stock_mapping: dict[str, dict[str, str]],
    target_l1_codes: set[str],
) -> dict[str, list[str]]:
    """按 SW L1 行业分组个股。

    注意：未分类股票 l1_code 为空，不会出现在任何目标行业分组中。

    Returns:
        {"801780.SI": ["000001.SZ", "002142.SZ", ...], ...}
    """
    result: dict[str, list[str]] = {code: [] for code in target_l1_codes}

    for ts_code, info in stock_mapping.items():
        l1 = info.get("l1_code", "")
        if l1 in target_l1_codes:
            result[l1].append(ts_code)

    for code in list(result.keys()):
        if not result[code]:
            del result[code]

    return result
