#!/usr/bin/env python3
"""移除 V4/V5 中残缺的 20260806 切片（20260806 全量数据已就绪后重建）。

背景: 22:30 cron 管线自愈在 data_refresh 个股拉取未完成时跑了 retrain_ml --update，
只写入 2524 只的 20260806。本脚本把残缺版备份到 .bak_partial，
主路径写回"移除 20260806"的版本，使后续 retrain_ml --update 能重建完整 20260806。
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

DATA = "data_storage"
TARGET_DATE = "20260806"

def strip(full_path):
    t0 = time.time()
    bak = full_path + ".bak_partial"
    if not os.path.exists(bak):
        os.replace(full_path, bak)          # 残缺版 → .bak_partial
    df = pd.read_parquet(bak)               # 从备份读
    keep = df[df["trade_date"].astype(str) != TARGET_DATE]
    n_removed = len(df) - len(keep)
    tmp = full_path + ".tmp"
    keep.to_parquet(tmp, index=False)
    os.replace(tmp, full_path)              # 主路径 ← 移除 20260806 的版本
    print(f"✓ {os.path.basename(full_path)}: {len(df):,} → {len(keep):,} 行 "
          f"(移除 {n_removed} 行 {TARGET_DATE}), {time.time()-t0:.0f}s")

for p in ["feature_matrix_v4.parquet", "feature_matrix_v5.parquet"]:
    strip(os.path.join(DATA, p))
