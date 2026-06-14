"""
个股↔申万一级行业映射
- 从 Tushare index_member_all 获取所有 SW 成分股关系
- 构建 stock_ts_code → l1_code/l1_name 的快速查找表
- 缓存到 CSV 供离线使用
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


def _fetch_from_tushare() -> dict[str, dict[str, str]]:
    """从 Tushare 获取当前 SW 成分股映射。

    调用一次 index_member_all 即返回所有行业的成分股（约 3000 只）。

    Returns:
        {"000001.SZ": {"l1_code": "801780.SI", "l1_name": "银行", ...}, ...}
    """
    pro = ts.pro_api(TUSHARE_TOKEN)
    # 用一个任意 L1 代码调用，实际返回所有行业
    df = pro.index_member_all(index_code="801010.SI")

    if df is None or df.empty:
        raise RuntimeError("Tushare index_member_all 返回空数据")

    # 只取当前成分股（is_new == 'Y'）
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

    logger.info("从 Tushare 获取 %d 只个股的行业映射", len(mapping))
    return mapping


def _save_to_csv(mapping: dict[str, dict[str, str]]) -> None:
    """保存映射到 CSV。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STOCK_MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["ts_code", "stock_name", "l1_code", "l1_name",
                      "l2_code", "l2_name", "l3_code", "l3_name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts_code, info in sorted(mapping.items()):
            row = {"ts_code": ts_code, **info}
            writer.writerow(row)
    logger.info("个股映射已保存至 %s", STOCK_MAPPING_CSV)


def _load_from_csv() -> dict[str, dict[str, str]]:
    """从 CSV 缓存加载映射。"""
    mapping: dict[str, dict[str, str]] = {}
    with open(STOCK_MAPPING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_code = row["ts_code"]
            mapping[ts_code] = {
                "l1_code": row["l1_code"],
                "l1_name": row["l1_name"],
                "l2_code": row["l2_code"],
                "l2_name": row["l2_name"],
                "l3_code": row["l3_code"],
                "l3_name": row["l3_name"],
                "stock_name": row["stock_name"],
            }
    logger.info("从 CSV 缓存加载 %d 只个股映射", len(mapping))
    return mapping


def load_stock_industry_mapping(force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """加载个股→行业映射。

    优先从 CSV 缓存读取，若不存在或 force_refresh=True 则从 Tushare 获取。

    Returns:
        {"000001.SZ": {"l1_code": "801780.SI", "l1_name": "银行", ...}, ...}
    """
    if not force_refresh and os.path.exists(STOCK_MAPPING_CSV):
        try:
            return _load_from_csv()
        except Exception:
            logger.warning("CSV 缓存读取失败，尝试从 Tushare 获取")

    mapping = _fetch_from_tushare()
    try:
        _save_to_csv(mapping)
    except Exception:
        logger.warning("个股映射 CSV 写入失败（不影响继续运行）")
    return mapping


def get_stocks_by_industry(
    stock_mapping: dict[str, dict[str, str]],
    target_l1_codes: set[str],
) -> dict[str, list[str]]:
    """按 SW L1 行业分组个股。

    Args:
        stock_mapping: 完整的个股→行业映射
        target_l1_codes: 需要筛选的 L1 行业代码集合

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
