"""
模块0-L2: 二级行业领先信号
- 计算 L2 相对其所属 L1 的超额动量
- 比 L3 粒度更适合投资方向判断
"""
import logging, numpy as np, pandas as pd
from config import MOMENTUM_LOOKBACK, L3_LEADING_THRESHOLD, L3_STRONG_LEADING

logger = logging.getLogger(__name__)

def analyze_l2_leading(l2_df, l1_df):
    result = {"status":"failed","df":pd.DataFrame(),"leading_count":0,"strong_leading":[],"by_l1":{},"error":None}
    try:
        if l2_df is None or l2_df.empty or l1_df is None or l1_df.empty:
            result["status"]="degraded"; result["error"]="数据为空"; return result
        l1_prices = {}
        for code, grp in l1_df.groupby("ts_code"):
            grp = grp.sort_values("trade_date")
            if len(grp) >= MOMENTUM_LOOKBACK+1: l1_prices[code] = grp["close"]
        records = []
        for code, grp in l2_df.groupby("ts_code"):
            grp = grp.sort_values("trade_date")
            if len(grp) < MOMENTUM_LOOKBACK+1: continue
            parent = grp["parent_l1"].iloc[0] if "parent_l1" in grp.columns else ""
            pname = grp["parent_name"].iloc[0] if "parent_name" in grp.columns else ""
            l2_name = grp["name"].iloc[0] if "name" in grp.columns else code
            if not parent or parent not in l1_prices: continue
            l2_p = grp["close"]; l1_p = l1_prices[parent]
            l2_ret = (l2_p.iloc[-1]/l2_p.iloc[-(MOMENTUM_LOOKBACK+1)]-1)*100
            l1_ret = (l1_p.iloc[-1]/l1_p.iloc[-(MOMENTUM_LOOKBACK+1)]-1)*100
            excess = l2_ret - l1_ret
            label = "🔥强烈领先" if excess>=L3_STRONG_LEADING else ("⚡领先" if excess>=L3_LEADING_THRESHOLD else ("同步" if excess>=0 else "落后"))
            records.append({"l2_code":code,"l2_name":l2_name,"parent_l1":parent,"parent_name":pname,"l2_return_20d":round(l2_ret,2),"excess_momentum":round(excess,2),"label":label})
        if not records: result["status"]="degraded"; result["error"]="无有效数据"; return result
        df = pd.DataFrame(records).sort_values("excess_momentum", ascending=False)
        df["rank"] = range(1, len(df)+1)
        leading = df[df["excess_momentum"]>=L3_LEADING_THRESHOLD]
        strong = df[df["excess_momentum"]>=L3_STRONG_LEADING]
        by_l1 = {}
        for _, r in leading.iterrows():
            pn = r["parent_name"] or r["parent_l1"]
            by_l1.setdefault(pn,[]).append({"l2_name":r["l2_name"],"excess":r["excess_momentum"]})
        result.update({"df":df,"leading_count":len(leading),"strong_leading":[{"name":r["l2_name"],"excess":r["excess_momentum"],"parent":r["parent_name"]} for _,r in strong.iterrows()],"by_l1":by_l1,"status":"success"})
        logger.info("模块0-L2: %d个L2, %d领先(%d强烈)", len(df), len(leading), len(strong))
    except Exception as e:
        logger.error("L2 leading失败: %s", e, exc_info=True)
        result["status"]="failed"; result["error"]=str(e)
    return result
