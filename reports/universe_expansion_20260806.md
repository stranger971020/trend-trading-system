# 全量 A 股股票池扩池总结（2026-08-05 → 08-06）

## 背景与目标

原股票池 = 申万一级行业指数**当前成分股**（Tushare `index_member_all` + `is_new=='Y'`）约 3000 只，仅覆盖全市场 ~55%，导致 V6 日报衰减榜/温度/选股覆盖面不足。

本次扩池到**沪深全量剔除 ST**（~4999 只），贯穿数据层→特征层→模型层→排名/报告层，全量重训模型。

## 已确认决策
1. 股票池：沪深主板+科创+创业板，剔除 ST/*ST、北交所(.BJ)、B股 → **4999 只**
2. 历史回溯：**点时间点宇宙**——新股票按上市日期逐步纳入（Tushare daily 天然只返回上市后数据，无幸存者偏差）
3. 行业补全：申万一级优先 → CSRC→申万（`industry_to_sw_l1.csv` 110→31 策划表）→ 东财兜底 → 未分类桶

## 变更清单

### 数据层
| 文件 | 变更 |
|------|------|
| `data/stock_industry_mapping.py` | 重写：全量清单 `fetch_full_market_list`、四档行业补全、CSV 14 列演进、`load_stock_universe()` |
| `data/stock_daily_updater.py` | `fetch_all_stocks` 加 `list_dates`（点时间点回填） |
| `data_refresh.py` | 股票池全量 + `refresh_market_index`（沪深300 入表，修复 `load_market_index` 回退 bug） |
| `data/eastmoney_industry.py` | 新增：东财行业兜底（防御性，CSRC 全覆盖故未接入热路径） |
| `config.py` | `INCLUDE_DELISTED`、`UNIVERSE_CSV_PATH`、`MARKET_INDEX_CODE` |

### 特征层
| 文件 | 变更 |
|------|------|
| `analysis/ml_model.py` | `has_sw_mapping` 二元特征；未分类股票行业特征用沪深300 市场指数代理 |
| `build_features_v5.py` | 同上（`load_base_features`） |
| `integrate_v5_features.py` | 未分类落 `UNCLASSIFIED` 合成桶；`rank_l1/l2_in_all` 对 UNCLASSIFIED 排除不排（50 中性） |
| `analysis/strategy_feature_masker.py` | `has_sw_mapping` 加入 `MOMENTUM_FEATURES`（行业特征最多的引擎） |

### 模型层
| 文件 | 变更 |
|------|------|
| `analysis/retrain_ml.py` | `load_stock_daily max_stocks=None` 全量 + universe 过滤；`--stocks` 默认 None |
| `backtest/v6_walkforward_benchmark.py` / `v6_three_layer_backtest.py` / `v6_production_backtest.py` | 同上 + 修 `%d` 日志 None 崩溃 |

### 报告/健康检查
| 文件 | 变更 |
|------|------|
| `backtest/v6_daily_report.py` | `FINAL_JSON` 参数化；`POSITION_RULES` 按新审计更新；衰减榜覆盖动态 |
| `backtest/v6_phase1_feasibility.py` / `v6_diversification_dashboard.py` | 硬编码 3000 → 动态 `len(mapping)` |
| `daily_bot_doc_check.py` | 新增 4 项断言：股票池数、行业覆盖、特征矩阵维度、universe 来源 |

## 新旧基准对比（v6_audit_full_20260806.json, 98 folds）

| 指标 | 旧(申万3000) | 新(全量4999) | 变化 |
|------|-------------|-------------|------|
| Rank IC | 0.1111 | **0.2729** | +145% |
| IC 正值折 | 93.9% | 95.9% | +2.0pp |
| T+1 WR | 46.4% | **51.75%** | +5.4pp |
| D+5 WR | 47.5% | **50.6%** | +3.1pp |
| Gate 阻塞率 | 79.5% | 84.3% | — |

**新仓位规则（Regime×温度）**：bear低温满仓(WR 74.1%)、bull低温满仓(61.9%)、bull高温空仓(33.3%) 等。

## 端到端验证
- `stock_daily`: **4999 只 / 10.3M 行**（全量回填 44 分钟）
- `feature_matrix_v5.parquet`: 9,893,465 行，最新日覆盖 4959 只，含 `has_sw_mapping` 列
- 日报生成：**衰减榜覆盖 4951 只**（原 2980），温度/仓位用新审计
- 健康检查 4 项新断言：全绿
- 新模型：`v6_{momentum,reversion,breakout}_20260728.pkl`

## 回滚
旧产物备份于 `backup/20260805/`（parquet .bak、模型、映射 CSV、代码）。`stock_daily` 受 `.stock_daily_protected` 保护（append-only，不破坏）。

## 已知限制
- 退市股未纳入（`INCLUDE_DELISTED=False` 开关预留）
- ST 按当前名剔除（"未来变 ST"股票历史段仍在池，~100 只，远期可用 Tushare `namechange` 严格化）
- 资金流/基本面缓存未扩容（V6 三引擎特征掩码不含这些列，模型影响为零）
- 龙虎榜 `limit_list_d` 接口无 Tushare 权限（既有限制）
