"""
eastmoney_industry.py — 东财行业分类兜底（Tier3，防御性）

全量池扩池 (2026-08-05)：行业补全四档中的第三档。
当前 CSRC(stock_basic.industry) 已覆盖全部 110 类（未分类=0），
本模块仅在 CSRC 缺失/无法归并的极少数股票上启用，不接入热路径。

映射: 东财行业名 → 申万一级名称（与 industry_to_sw_l1.csv 同构的 keyword 表）

── Changelog ──
# 2026-08-05 Claude: 新增东财行业兜底模块（全量池扩池 Tier3 防御）
#               当前未接入 build_universe_mapping 热路径（CSRC 全覆盖）
─────────────
"""

import logging

logger = logging.getLogger(__name__)

# 东财行业名 → 申万一级名称（keyword 归并，供 CSRC 缺失时兜底）
# 与 data_storage/industry_to_sw_l1.csv 的 CSRC 映射同构
EM_TO_SW_L1 = {
    "银行": "银行", "保险": "非银金融", "证券": "非银金融",
    "房地产": "房地产", "酿酒行业": "食品饮料", "食品饮料": "食品饮料",
    "电子元件": "电子", "半导体": "电子", "光学光电子": "电子",
    "电力行业": "公用事业", "公用事业": "公用事业", "燃气": "公用事业",
    "通信设备": "通信", "通信服务": "通信", "软件服务": "计算机",
    "互联网服务": "传媒", "游戏": "传媒", "文化传媒": "传媒",
    "汽车整车": "汽车", "汽车零部件": "汽车", "汽车服务": "汽车",
    "医药商业": "医药生物", "化学制药": "医药生物", "生物制品": "医药生物",
    "中药": "医药生物", "医疗器械": "医药生物", "医疗服务": "医药生物",
    "煤炭行业": "煤炭", "石油行业": "石油石化", "贵金属": "有色金属",
    "有色金属": "有色金属", "钢铁行业": "钢铁", "工程建设": "建筑装饰",
    "装修装饰": "建筑装饰", "水泥建材": "建筑材料", "玻璃陶瓷": "建筑材料",
    "家用轻工": "轻工制造", "造纸印刷": "轻工制造", "包装材料": "轻工制造",
    "纺织服装": "纺织服饰", "商业百货": "商贸零售", "贸易行业": "商贸零售",
    "旅游酒店": "社会服务", "教育": "社会服务", "综合行业": "综合",
    "环保行业": "环保", "化肥行业": "基础化工", "化学原料": "基础化工",
    "化学制品": "基础化工", "塑料制品": "基础化工", "橡胶制品": "基础化工",
    "化纤行业": "基础化工", "农药兽药": "基础化工", "工程建设": "建筑装饰",
    "专用设备": "机械设备", "通用设备": "机械设备", "仪器仪表": "机械设备",
    "工程机械": "机械设备", "航天航空": "国防军工", "船舶制造": "国防军工",
    "电网设备": "电力设备", "电机": "电力设备", "电源设备": "电力设备",
    "光伏设备": "电力设备", "风电设备": "电力设备", "电池": "电力设备",
    "家电行业": "家用电器", "白色家电": "家用电器", "小家电": "家用电器",
    "农牧饲渔": "农林牧渔", "种植业": "农林牧渔", "水产养殖": "农林牧渔",
    "食品加工": "食品饮料", "饮料": "食品饮料", "物流行业": "交通运输",
    "港口水运": "交通运输", "民航机场": "交通运输", "高速公路": "交通运输",
    "交运设备": "机械设备", "铁路基建": "建筑装饰", "输配电气": "电力设备",
    "船舶制造": "国防军工", "酿酒行业": "食品饮料",
    "美容护理": "美容护理", "化妆品": "美容护理", "美妆": "美容护理",
}


def em_industry_to_sw_l1(em_industry: str) -> str:
    """东财行业名 → 申万一级名称；无匹配返回空串（由调用方落未分类）。"""
    return EM_TO_SW_L1.get(em_industry, "")


def fetch_eastmoney_industry(ts_codes: list[str]) -> dict[str, str]:
    """批量获取东财行业分类（逐股 a-stock-data eastmoney_stock_info 或快照）。

    Args:
        ts_codes: 需要补全的股票代码列表（通常是 CSRC 无法归并的少数）

    Returns:
        {ts_code: em_industry} 仅返回有结果的
    """
    result: dict[str, str] = {}
    try:
        # 优先尝试 a-stock-data skill 的接口（若已安装）
        from skills_a_stock_data import eastmoney_stock_info  # noqa: F401
        has_skill = True
    except ImportError:
        has_skill = False

    if not has_skill:
        logger.warning("a-stock-data skill 未安装，跳过东财兜底（CSRC 已覆盖时无影响）")
        return result

    for code in ts_codes:
        try:
            info = eastmoney_stock_info(code)
            ind = info.get("industry", "")
            if ind:
                result[code] = ind
        except Exception as e:
            logger.warning("东财行业拉取失败 %s: %s", code, e)
    logger.info("东财行业兜底: %d/%d 只", len(result), len(ts_codes))
    return result


if __name__ == "__main__":
    # 自测: 抽查几个东财行业名映射
    for k, v in list(EM_TO_SW_L1.items())[:10]:
        print(f"  {k} → {v}")
