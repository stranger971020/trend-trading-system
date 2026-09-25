# -*- coding: utf-8 -*-
"""
波段引擎 (Swing Engine) — 预测未来 20 日收益分布的个股排序模型

与 V6 winrate_engine 的区别:
  目标: ±2% 中性带 → **对齐用户目标**(20%)
        0: fwd20 ≤ -10%   (亏损区)
        1: -10% ~ +5%     (平庸)
        2: +5% ~ +20%     (小赢)
        3: ≥ +20%         (目标达成) ← 用户真正想抓的
  输出: p_win = P(class3); expect_ret = Σ P_k × repr_k; swing_score = expect_ret 的截面分位

训练: 3 折 walk-forward（时间序列, 无前视）
  折1 训练≤2020 → 测试 2021
  折2 训练≤2022 → 测试 2023-2024
  折3 训练≤2024 → 测试 2025-2026

Changelog
─────────
2026-09-25 Hermes: 新建。回应用户"波段版模型 + 决策单第③层"。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.strategy_feature_masker import ENGINE_FEATURE_MASKS  # noqa: E402

MATRIX = os.path.join(PROJECT_ROOT, "data_storage", "feature_matrix_v5.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data_storage", "swing_models")
REPORT = os.path.join(PROJECT_ROOT, "backtest", "swing_engine_report.json")
FULL_REPORT = os.path.join(PROJECT_ROOT, "backtest", "swing_engine_report.md")

FWD_DAYS = 20
EDGES = [-0.10, 0.05, 0.20]          # 分档边界
REPR = [-0.15, -0.02, 0.12, 0.30]    # 各档代表收益
CLASS_W = {0: 1.5, 1: 1.0, 2: 1.0, 3: 2.0}
N_CLASS = 4

EXTRA_FEATURES = ["composite_regime_factor", "regime_adaptive_weight",
                  "vol_regime_zscore_position", "low_vol_anomaly_score",
                  "momentum_crowding_divergence", "vol_regime_zscore_position"]

FOLDS = [("20201231", "20210101", "20211231"),
         ("20221231", "20230101", "20241231"),
         ("20241231", "20250101", "20261231")]


def feature_columns():
    u = set()
    for f in ENGINE_FEATURE_MASKS.values():
        u |= set(f)
    return sorted(u)


def load_matrix(cols):
    schema = set(pq.ParquetFile(MATRIX).schema_arrow.names)
    keep = [c for c in cols if c in schema]
    missing = [c for c in cols if c not in schema]
    if missing:
        print(f"  跳过矩阵中不存在的特征 {len(missing)}: {missing[:8]}", file=sys.stderr)
    print(f"  读取 {len(keep)+3} 列 ...", file=sys.stderr)
    t0 = time.time()
    tbl = pq.read_table(MATRIX, columns=["ts_code", "trade_date", "close"] + keep)
    df = tbl.to_pandas()
    print(f"  {len(df):,} 行 · {time.time()-t0:.0f}s", file=sys.stderr)
    return df, keep


def build_labels(df):
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort")
    g = df.groupby("ts_code", sort=False)["close"]
    df["fwd20"] = g.transform(lambda s: s.shift(-FWD_DAYS) / s - 1)
    f = df["fwd20"]
    df["y"] = np.where(f <= EDGES[0], 0, np.where(f <= EDGES[1], 1,
                      np.where(f < EDGES[2], 2, 3)))
    df.loc[f.isna(), "y"] = -1
    return df


def train_fold(df, feat_cols, train_end):
    import lightgbm as lgb
    tr = df[(df["trade_date"] <= train_end) & (df["y"] >= 0)]
    X = tr[feat_cols].astype("float32")
    y = tr["y"].astype(int).to_numpy()
    w = np.array([CLASS_W[v] for v in y], dtype="float32")
    print(f"    训练样本 {len(X):,} · 类别分布 {np.bincount(y, minlength=N_CLASS).tolist()}", file=sys.stderr)
    t0 = time.time()
    m = lgb.LGBMClassifier(objective="multiclass", num_class=N_CLASS, n_estimators=200,
                           num_leaves=31, learning_rate=0.05, min_child_samples=50,
                           subsample=0.8, colsample_bytree=0.8, verbosity=-1,
                           force_col_wise=True, n_jobs=-1)
    m.fit(X, y, sample_weight=w)
    print(f"    训练完成 {time.time()-t0:.0f}s", file=sys.stderr)
    return m


def _spearman(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 50:
        return np.nan
    return d["a"].corr(d["b"], method="spearman")


def evaluate(m, df, feat_cols, t0, t1):
    te = df[(df["trade_date"] >= t0) & (df["trade_date"] <= t1) & (df["y"] >= 0)].copy()
    if te.empty:
        return None
    proba = m.predict_proba(te[feat_cols].astype("float32"))
    te["expect_ret"] = proba @ np.array(REPR)
    te["p_win"] = proba[:, 3]
    # 按日 Rank IC
    ics = te.groupby("trade_date").apply(lambda g: _spearman(g["expect_ret"], g["fwd20"]))
    ic = float(ics.mean())
    ic_pos = float((ics > 0).mean())
    # 分位分层
    te["decile"] = te.groupby("trade_date")["expect_ret"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
    lay = te.groupby("decile")["fwd20"].agg(["mean", "median", "size"])
    # p_win 校准
    te["pb"] = pd.cut(te["p_win"], [0, .05, .10, .20, .40, 1.0], labels=["<5%", "5-10%", "10-20%", "20-40%", "≥40%"])
    cal = te.groupby("pb", observed=True).agg(n=("fwd20", "size"), actual_mean=("fwd20", "mean"),
                                              hit20=("fwd20", lambda x: (x >= 0.20).mean()))
    top = te[te["decile"] == 9]
    return {"test": f"{t0}~{t1}", "n": int(len(te)),
            "rank_ic": round(ic, 4), "ic_pos_days": round(ic_pos, 3),
            "top_decile_fwd20_mean": round(float(top["fwd20"].mean()), 4),
            "top_decile_hit20": round(float((top["fwd20"] >= 0.20).mean()), 4),
            "all_hit20": round(float((te["fwd20"] >= 0.20).mean()), 4),
            "layers": {str(int(k)): {"mean": round(float(r["mean"]), 4), "n": int(r["size"])}
                       for k, r in lay.iterrows()},
            "calibration": {str(k): {"n": int(r["n"]), "actual_mean": round(float(r["actual_mean"]), 4),
                                     "hit20": round(float(r["hit20"]), 4)} for k, r in cal.iterrows()}}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    fc = feature_columns()
    print("加载特征矩阵 ...", file=sys.stderr)
    df, avail = load_matrix(fc)
    df = build_labels(df)
    print(f"可用特征 {len(avail)}", file=sys.stderr)
    out = {"folds": [], "fwd_days": FWD_DAYS, "edges": EDGES, "repr": REPR, "n_features": len(avail)}
    for tr_end, te0, te1 in FOLDS:
        print(f"折: 训练≤{tr_end} 测试 {te0}~{te1}", file=sys.stderr)
        m = train_fold(df, avail, tr_end)
        ev = evaluate(m, df, avail, te0, te1)
        if ev:
            out["folds"].append(ev)
            print(f"    RankIC {ev['rank_ic']:+.4f} · Top10% D20 {ev['top_decile_fwd20_mean']*100:+.2f}% "
                  f"· hit20 {ev['top_decile_hit20']*100:.1f}% (全样本 {ev['all_hit20']*100:.1f}%)", file=sys.stderr)
        out["last_model"] = os.path.join(MODEL_DIR, f"swing_{tr_end}.pkl")
        import pickle
        with open(out["last_model"], "wb") as fh:
            pickle.dump({"model": m, "features": avail, "edges": EDGES}, fh)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # markdown
    L = ["# 波段引擎训练报告\n", "> 目标: 未来20日收益 4 档；class3 = ≥+20%\n",
         f"> 特征 {len(avail)} 个 · {len(out['folds'])} 折 walk-forward\n"]
    L.append("\n| 测试期 | 笔数 | Rank IC | IC>0天数占比 | Top10%月收益 | Top10%达标≥20% | 全样本≥20% |")
    L.append("|---|---|---|---|---|---|---|")
    for f_ in out["folds"]:
        L.append(f"| {f_['test']} | {f_['n']:,} | **{f_['rank_ic']:+.4f}** | {f_['ic_pos_days']*100:.0f}% | "
                 f"{f_['top_decile_fwd20_mean']*100:+.2f}% | **{f_['top_decile_hit20']*100:.1f}%** | {f_['all_hit20']*100:.1f}% |")
    L.append("\n## 分层（最后一折）\n")
    if out["folds"]:
        L.append("| 分位 | 平均月收益 |")
        L.append("|---|---|")
        for k, v in out["folds"][-1]["layers"].items():
            L.append(f"| D{int(k)+1} | {v['mean']*100:+.2f}% |")
        L.append("\n## p_win 校准（最后一折）\n")
        L.append("| 预测 p_win | 样本 | 实际月均 | 实际≥20%率 |")
        L.append("|---|---|---|---|")
        for k, v in out["folds"][-1]["calibration"].items():
            L.append(f"| {k} | {v['n']:,} | {v['actual_mean']*100:+.2f}% | {v['hit20']*100:.1f}% |")
    with open(FULL_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\n✅ {REPORT} / {FULL_REPORT}", file=sys.stderr)


if __name__ == "__main__":
    main()
