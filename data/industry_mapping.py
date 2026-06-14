"""
申万一级行业代码↔名称映射
- 从 Tushare index_classify 获取 31 个 SW L1 行业代码和名称
- 缓存到 CSV，离线时可用
"""

import csv
import logging
import os

import tushare as ts

from config import (
    TUSHARE_TOKEN,
    MAPPING_CSV,
)

logger = logging.getLogger(__name__)

# 硬编码回退映射（31 个 SW L1 行业，基于 SW2021 分类）
_FALLBACK_MAPPING: dict[str, str] = {
    "801010.SI": "农林牧渔",
    "801020.SI": "煤炭",
    "801030.SI": "基础化工",
    "801040.SI": "钢铁",
    "801050.SI": "有色金属",
    "801080.SI": "电子",
    "801110.SI": "家用电器",
    "801120.SI": "食品饮料",
    "801130.SI": "纺织服饰",
    "801140.SI": "轻工制造",
    "801150.SI": "医药生物",
    "801160.SI": "公用事业",
    "801170.SI": "交通运输",
    "801180.SI": "房地产",
    "801200.SI": "商贸零售",
    "801210.SI": "社会服务",
    "801230.SI": "综合",
    "801710.SI": "建筑材料",
    "801720.SI": "建筑装饰",
    "801730.SI": "电力设备",
    "801740.SI": "国防军工",
    "801750.SI": "计算机",
    "801760.SI": "传媒",
    "801770.SI": "通信",
    "801780.SI": "银行",
    "801790.SI": "非银金融",
    "801880.SI": "汽车",
    "801890.SI": "机械设备",
    "801950.SI": "石油石化",
    "801960.SI": "环保",
    "801980.SI": "美容护理",
}


def _fetch_from_tushare() -> dict[str, str]:
    """从 Tushare 获取最新的 SW L1 行业映射。

    Returns:
        dict: {"801010.SI": "农林牧渔", "801020.SI": "采掘", ...}
    """
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.index_classify(level="L1", src="SW2021")

    if df is None or df.empty:
        raise RuntimeError("Tushare index_classify 返回空数据，请检查 token 或网络")

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row.get("index_code", ""))
        name = str(row.get("industry_name", ""))
        if code and name:
            mapping[code] = name

    if not mapping:
        raise RuntimeError("未能从 index_classify 结果中解析出行业代码和名称")

    logger.info("从 Tushare 获取到 %d 个 SW L1 行业", len(mapping))
    return mapping


def _save_to_csv(mapping: dict[str, str]) -> None:
    """将映射保存到 CSV 文件。"""
    os.makedirs(os.path.dirname(MAPPING_CSV), exist_ok=True)
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_code", "name"])
        for code, name in sorted(mapping.items()):
            writer.writerow([code, name])
    logger.info("行业映射已保存至 %s", MAPPING_CSV)


def _load_from_csv() -> dict[str, str]:
    """从 CSV 缓存加载行业映射。"""
    mapping: dict[str, str] = {}
    with open(MAPPING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["ts_code"]] = row["name"]
    logger.info("从 CSV 缓存加载 %d 个行业映射", len(mapping))
    return mapping


def load_industry_mapping(force_refresh: bool = False) -> dict[str, str]:
    """加载 SW L1 行业代码→名称映射。

    优先从 CSV 缓存读取；若缓存不存在或 force_refresh=True，
    则从 Tushare 实时获取并更新缓存。

    Args:
        force_refresh: 是否强制从 Tushare 重新获取

    Returns:
        dict: {"801010.SI": "农林牧渔", ...}
    """
    if not force_refresh and os.path.exists(MAPPING_CSV):
        try:
            return _load_from_csv()
        except Exception:
            logger.warning("CSV 缓存读取失败，尝试从 Tushare 获取")

    try:
        mapping = _fetch_from_tushare()
        try:
            _save_to_csv(mapping)
        except Exception:
            logger.warning("CSV 缓存写入失败（不影响继续运行）")
        return mapping
    except Exception as e:
        logger.warning("Tushare 获取失败: %s，使用硬编码回退映射", e)
        return dict(_FALLBACK_MAPPING)
