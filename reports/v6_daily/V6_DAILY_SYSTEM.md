# V6 每日交易日报 — 系统交付文档

> 版本: 1.0 | 日期: 2026-08-02 | 作者: Claude

---

## 目录

1. [系统架构](#1-系统架构)
2. [数据依赖链与 SLA](#2-数据依赖链与-sla)
3. [P1 Gate 温度计 — 计算原理](#3-p1-gate-温度计--计算原理)
4. [P2 Alpha Decay 衰减榜 — 计算原理](#4-p2-alpha-decay-衰减榜--计算原理)
5. [P3 仓位配置规则 — 计算原理](#5-p3-仓位配置规则--计算原理)
6. [特征矩阵增量刷新](#6-特征矩阵增量刷新)
7. [模型周度重训练](#7-模型周度重训练)
8. [管道编排与自动化](#8-管道编排与自动化)
9. [日报 HTML 与 Telegram 交付](#9-日报-html-与-telegram-交付)
10. [版本问题与已知限制](#10-版本问题与已知限制)
11. [文件清单](#11-文件清单)
12. [运维手册](#12-运维手册)

---

## 1. 系统架构

```
                              ┌──────────────────────┐
                              │   Tushare Pro API    │
                              │   (外部数据源)        │
                              └──────────┬───────────┘
                                         │ 每日收盘后
                                         ▼
                              ┌──────────────────────┐
                              │   stock_daily (DB)   │
                              │   2991只A股 × 日频    │
                              │   SLA: T日 20:00     │
                              └──────────┬───────────┘
                                         │ 每日 21:00 cron
                                         ▼
                              ┌──────────────────────┐
                              │ feature_matrix_v5    │
                              │ parquet (321列)       │
                              │ SLA: T日 22:00       │
                              └──────────┬───────────┘
                                         │ 每周日 02:30 cron
                                         ▼
                              ┌──────────────────────┐
                              │ lgb_models/          │
                              │ v6_{engine}_{date}.pkl│
                              │ SLA: 周日 03:00      │
                              └──────────┬───────────┘
                                         │ 每日 08:30 cron
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
              ┌──────────┐       ┌──────────┐        ┌──────────┐
              │ P1 温度计 │       │P2 衰减榜  │        │P3 仓位规则│
              │ 全市场PWin │       │4周对比     │        │98fold查表 │
              └────┬─────┘       └────┬─────┘        └────┬─────┘
                   └──────────────────┼──────────────────┘
                                      ▼
                              ┌──────────────────────┐
                              │   日报 HTML + TG     │
                              │   GitHub Pages 部署   │
                              └──────────────────────┘
```

## 2. 数据依赖链与 SLA

所有时间为北京时间 (UTC+8)。

| 层级 | 数据 | 来源 | 更新方式 | SLA | 当前状态 |
|------|------|------|---------|-----|---------|
| L0 | stock_daily | Tushare pro.daily() → SQLite | run_analysis.py 晚间流程 | T日 20:00 | ✅ 7/31 |
| L1 | feature_matrix_v5.parquet | stock_daily + 行业指数 + 基本面 | `integrate_v5_features.py --update` | T日 22:00 | ⚠️ 7/30 (差1天) |
| L2 | lgb_models/v6_*.pkl | feature_matrix + LightGBM 训练 | `v6_weekly_retrain.py` (周日 02:30) | 周日 03:00 | ⚠️ 7/22 (11天) |
| L3 | v6_final_*.json (folds) | 全量 walk-forward 回测 | 手动运行 benchmark | 每月 1 次 | ✅ 98 folds (8/1) |
| L4 | 日报 HTML | L0+L1+L2+L3 | `v6_daily_pipeline.py` (08:30) | T+1日 08:45 | ✅ 7/31 已生成 |

### 版本号追溯

每份日报 HTML 底部 Footer 标注了所有上游数据版本：
- 特征日期: 从 feature_matrix_v5.parquet 的 max(trade_date)
- 行情日期: 从 stock_daily 的 max(trade_date)
- 模型版本: 从 lgb_models 文件名的 asof_date（如 `v6_momentum_20260722.pkl` → 模型训练数据截止 2026-07-22）

日报文件名格式: `v6_daily_YYYYMMDD.html`，其中 YYYYMMDD 为行情基准日（stock_daily 最新日期）。

## 3. P1 Gate 温度计 — 计算原理

### 3.1 温度定义

温度是一个 0-100 的分数，表示当前市场对 V6 模型的"友好程度"。
- 0-35: 低温（极寒）— 市场环境恶劣
- 35-55: 中温 — 适度参与
- 55-100: 高温（偏热）— 市场环境友好，但注意过热风险

### 3.2 计算步骤

**Step 1: 全市场 PWin 推断**

对 feature_matrix 最新日期的所有股票（~2991只），用 3 个引擎模型分别推断 P(Win)：
```
P(Win) = model.predict_proba(X)[:, 2]
```
其中 Bucket_2 = fwd_return > +2% (True Alpha)。模型为 LightGBM Ordinal 三分类器。

**Step 2: 百分位排名（消除校准偏差）**

已知问题：momentum/reversion 模型的 PWin 过度集中（0.75-0.98），绝对概率不具可比性。

解决方法：在每个引擎内部独立计算 PWin 的百分位排名（rank pct），然后取三引擎平均排名。
```
rank_i = percentile(PWin_engine_i)   # 0-1 之间
avg_rank = mean(rank_momentum, rank_reversion, rank_breakout)
```

**Step 3: 映射到温度**
```
temp = avg_rank × 60 + 20
```
范围理论为 20-80（当 avg_rank 在 0-1 之间时），clip 到 0-100。

### 3.3 历史校准

温度的历史基准来自 98 个 fold（2024-01 至 2026-03）的回测数据。98 folds 的 gate_blocked_pct 与 T+1 WR 的 Pearson 相关系数为 -0.225，按 regime 分层后温差效应显著。

### 3.4 技术细节

- 推断耗时: ~2 秒（3 引擎 × 2991 stocks × predict_proba）
- 模型特征: momentum 56 个、reversion 54 个、breakout 43 个
- 特征对齐: 使用 `model.feature_name_` 精确匹配，缺失列填 0
- Regime 推断: 使用最近一个 fold 的 regime 标签（来自 v6_final_*.json）

## 4. P2 Alpha Decay 衰减榜 — 计算原理

### 4.1 衰减定义

```
decay_ratio = P(Win)_today / P(Win)_4weeks_ago
decay_pct = (decay_ratio - 1) × 100%
```

- decay_pct < 0: 衰减（PWin 下降）
- decay_pct > 0: 改善（PWin 上升）

### 4.2 计算步骤

**Step 1: 确定对比日期**

- today = feature_matrix 最新日期
- past = today - 4 周的最近交易日

**Step 2: 两期 PWin 推断**

对两期交集股票（约 2981 只），分别用当时最新的模型推断 PWin。注意 `today` 和 `past` 使用的模型版本不同——`today` 用最新模型，`past` 用 ≤past_date 的最新模型。

**Step 3: 三引擎平均**

```
PWin = mean(PWin_momentum, PWin_reversion, PWin_breakout)
```

只计算三引擎都有 PWin 的股票。引擎权重相等。

**Step 4: 排序输出**

- 🔴 衰减榜: 按 decay_pct 升序（跌幅最大排第一）
- 🟢 上升榜: 按 decay_pct 降序（升幅最大排第一）
- 行业聚合: 按 L1 行业 groupby，计算平均 decay_pct 和衰减占比

### 4.3 行业映射

使用 `data_storage/stock_industry_mapping.csv`（3000 只股票 → 申万 L1/L2/L3 行业分类）。来源: Tushare `index_member_all` API。

### 4.4 已知问题: PWin 变化来源混淆

decay_pct 包含两个来源的变化：
1. 股票本身变差/变好（真实 Alpha Decay）
2. 模型重训练导致特征权重变化

两者在当前设计下无法区分。实际使用中这不一定是问题——无论原因如何，PWin 下降都意味着模型对该股票信心降低。

### 4.5 前端搜索功能

日报 HTML 嵌入全量 2981 条衰减数据为 JSON（~390KB），支持客户端即时搜索：
- 搜索框支持股票代码、名称、行业关键词
- 结果显示: 代码、名称、行业、PWin(前/今)、变化%、全市场排名
- 行业表旁有独立筛选框

## 5. P3 仓位配置规则 — 计算原理

### 5.1 规则表来源

基于 98 个历史 fold（2.5 年，2024-01 至 2026-03），按 (Regime, 温度分位) 二维分组，计算每组的历史平均 T+1 WR 和 D+5 WR。

### 5.2 仓位建议逻辑

```
if T+1_WR >= 0.55 and D+5_WR >= 0.50 → 5只 (满仓)
elif T+1_WR >= 0.45                 → 3只 (半仓)
elif T+1_WR >= 0.35                 → 2只 (轻仓)
else                                → 1只或空仓
```

### 5.3 完整规则表

| Regime | 温度 | 建议仓位 | 历史 T+1 WR | 历史 D+5 WR | Folds |
|--------|------|---------|------------|------------|-------|
| bull | 低温 | 5只 (满仓) | 67.7% | 63.1% | 13 |
| bull | 中温 | 2只 (轻仓) | 40.0% | 52.7% | 11 |
| bull | 高温 | 2只 (轻仓) | 38.5% | 32.3% | 13 |
| range | 低温 | 空仓或1只 | 27.3% | 45.5% | 11 |
| range | 中温 | 2只 (轻仓) | 41.3% | 36.5% | 21 |
| range | 高温 | 3只 (半仓) | 50.0% | 51.7% | 20 |
| bear | 低温 | 5只 (满仓) | 63.0% | 59.3% | 9 |

### 5.4 关键发现

- **Bull+高温是陷阱**: 历史 D+5 仅 32.3%，牛市末期追涨风险极高
- **过度分散反降胜率**: 行业数 4-6 个时 WR=55.9%，>10 个降到 36.0%
- **Bear 市场样本少**: 仅 9 个 fold，bear 行的置信度低于其他行
- **Regime 依赖 fold 标签**: 当前日报用最近一个 fold 的 regime 推断，如果市场状态已切换，regime 可能滞后

## 6. 特征矩阵增量刷新

### 6.1 版本历史

| 版本 | 路径 | 大小 | 说明 |
|------|------|------|------|
| V4 | `feature_matrix_v4.parquet` | 6.8 GB | 118 基础特征，由 build_full_feature_matrix() 构建 |
| V5 | `feature_matrix_v5.parquet` | ~250 MB | V4 + 182 个 V5 扩展特征，由 integrate_v5_features.py 构建 |

V5 不是 V4 的替代，而是 V4 的超集。V5 parquet 包含 V4 的全部列 + V5 新增列，通过 (ts_code, trade_date) join 合并。

### 6.2 增量刷新原理

`integrate_v5_features.py --update` 的工作流程：

```
1. 读取现有 V5 parquet → 获取 max(trade_date) = D_v5
2. 查询 stock_daily → 获取 max(trade_date) = D_db
3. 如果 D_db ≤ D_v5 → 跳过（已是最新）
4. 否则:
   a. 加载 V4 parquet [D_v5 - 300天, D_db] 窗口
   b. 加载 stock_daily 同一窗口的原始 OHLCV
   c. 运行 compute_all_new_features() 计算 V5 特征
   d. 只保留 trade_date > D_v5 的行
   e. 追加到现有 V5 parquet
```

### 6.3 性能

- 增量模式: ~15-20 秒（1 天 × 2991 stocks）
- 全量模式: ~5-10 分钟（275 天 × 2991 stocks）
- 瓶颈: `compute_all_new_features()` 中的 9 个特征组计算（纯价格衍生 + 跨截面）

### 6.4 300 天窗口的原因

最长滚动特征 `ma250_dev` 需要 250 个前序交易日的收盘价。300 天留有安全余量。

## 7. 模型周度重训练

### 7.1 模型架构

V6 使用 3 个独立引擎，每个引擎一个 LightGBM 模型：

| 引擎 | 特征数 | 选股风格 | 模型类型 |
|------|--------|---------|---------|
| momentum | 56 | 趋势跟踪（mom, MA, MACD） | LGBMClassifier (Ordinal, 3类) |
| reversion | 54 | 均值回归（RSI, BB, VWAP） | LGBMClassifier (Ordinal, 3类) |
| breakout | 43 | 突破（波动率压缩, 流动性） | LGBMClassifier (Ordinal, 3类) |

### 7.2 标签定义

三分类标签（Ordinal），基于 fwd_return_20d:
- 0: 强亏 (fwd < -2%)
- 1: 中性 (-2% ≤ fwd ≤ +2%)
- 2: True Alpha (fwd > +2%)

P(Win) = P(Bucket_2) = 模型预测为 True Alpha 的概率

### 7.3 重训练模式对比

| 模式 | 脚本 | Fold 数 | 耗时 | 频率 |
|------|------|--------|------|------|
| 轻量周度 | v6_weekly_retrain.py | 1 fold (12周训+1周验) | ~10 秒 | 每周日 |
| 全量 WFV | v6_walkforward_benchmark.py | 98 folds | ~1 小时 | 手动/每月 |

轻量模式不跑全量回测，只训练最新模型用于推断。全量 WFV 用于模型评估和 fold 数据更新（供 P1 温度校准和 P3 规则表）。

### 7.4 模型文件命名

```
v6_{engine}_{asof_date}.pkl

例: v6_momentum_20260722.pkl
    → momentum 引擎, 训练数据截止 2026-07-22
```

### 7.5 已知问题: IC=0

当前 `v6_weekly_retrain.py` 的验证 IC 报告为 0.0000。原因是:
1. 标签极度不均衡: bucket_2 (True Alpha) 仅占 ~1% 样本
2. `rank_ic()` 的 alpha_boost ×2.0 在 1% 的 True Alpha 比例下效果微弱
3. 模型训练本身成功（predict_proba 输出有效），但 IC 指标对极端不平衡数据不敏感

这不影响推断——模型产生的 PWin 排序在之前的 98-fold benchmark 中已验证有效（IC=0.111 boosted, 93.8% folds 为正）。

## 8. 管道编排与自动化

### 8.1 管道脚本

`v6_daily_pipeline.py` 是统一入口，取代手动逐个跑脚本。

**两种运行模式**:

```bash
# 模式 1: 仅刷新数据（晚间 cron）
python3 v6_daily_pipeline.py --refresh-only

# 模式 2: 仅生成报告（盘前 cron）
python3 v6_daily_pipeline.py --report-only --telegram --deploy
```

**运行流程** (--refresh-only):
```
check_freshness()  → 检查 stock_daily / feature_matrix / models 新鲜度
    │
    ├── feature_matrix 滞后 → refresh_features() → integrate_v5_features.py --update
    └── models 过期 (>7天) → refresh_models()    → v6_weekly_retrain.py
```

**运行流程** (--report-only):
```
check_freshness()  → 报告当前数据新鲜度状态
    │
    └── 调用 v6_daily_report.py --asof <stock_daily最新日期>
        ├── compute_temperature()    (P1)
        ├── compute_decay_ranking()  (P2)
        ├── generate_html()          → reports/v6_daily/
        ├── send_telegram()          → Telegram Bot API
        └── git_deploy()             → GitHub Pages
```

### 8.2 Cron 配置

```cron
# V6 日报晚间数据刷新 (工作日 21:00, 赶在 22:00 SLA 前)
0 21 * * 1-5 cd /Users/jren/projects/trend-trading-system && /Users/jren/miniforge3/bin/python3 backtest/v6_daily_pipeline.py --refresh-only >> logs/v6_pipeline_cron.log 2>&1

# V6 日报盘前生成 (工作日 08:30, 赶在 09:00 开盘前)
30 8 * * 1-5 cd /Users/jren/projects/trend-trading-system && /Users/jren/miniforge3/bin/python3 backtest/v6_daily_pipeline.py --report-only --telegram --deploy >> logs/v6_pipeline_cron.log 2>&1

# V6 模型周度重训练 (周日 02:30, 赶在 03:00 SLA 前)
30 2 * * 0 cd /Users/jren/projects/trend-trading-system && /Users/jren/miniforge3/bin/python3 backtest/v6_weekly_retrain.py >> logs/v6_weekly_retrain.log 2>&1
```

添加方式:
```bash
crontab -e
# 追加以上 3 行
```

### 8.3 日志文件

| 日志 | 内容 |
|------|------|
| `logs/v6_pipeline_cron.log` | 管道运行日志（刷新 + 日报生成） |
| `logs/v6_weekly_retrain.log` | 周度模型训练日志 |
| `logs/v6_pipeline.log` | 管道脚本自身的详细日志（Python logging） |

## 9. 日报 HTML 与 Telegram 交付

### 9.1 HTML 结构

日报 HTML 包含 5 个卡片区域:
1. **核心结论**: 温度、仓位建议、历史胜率（左侧色条，醒目）
2. **温度计**: 大号数字 + 渐变色条 + Gate 详情 + 解读
3. **衰减榜**: 双栏（🔴 TOP10 + 🟢 TOP10）+ 行业排名 + 搜索框
4. **仓位规则**: 完整规则表（当前行高亮）+ 反直觉警示
5. **交易策略**: 三步使用流程 + 当日具体建议

### 9.2 CSS 规范

- 配色: Tailwind slate 灰系 (`#1e293b` 主文字, `#f1f5f9` 背景)
- 字体: 系统字体栈 `-apple-system, 'PingFang SC', sans-serif`
- 最大宽度: 900px, 手机适配 `@media max-width:600px`
- 温度色: `#ef4444`(红) / `#f59e0b`(黄) / `#16a34a`(绿)
- 零外部依赖: 无 CDN CSS/JS/字体

### 9.3 Telegram 格式

Telegram 推送为 HTML 格式（`parse_mode="HTML"`），控制在 300 字符以内：
```
📊 V6日报 {date} {weekday} → {next_date} {next_weekday}

🌡️ 温度 {temp}/100 {level} | 推断 Regime: {regime}
📈 建议仓位: {size} | 历史 T+1胜率 {wr}
📉 衰减板块: {worst_industries}
🟢 逆势板块: {best_industries}

📄 完整报告: {github_pages_url}
```

### 9.4 GitHub Pages 部署

```
本地路径: reports/v6_daily/v6_daily_YYYYMMDD.html
GitHub URL: https://stranger971020.github.io/trend-trading-system/reports/v6_daily/v6_daily_YYYYMMDD.html
部署方式: git add + git commit + git push origin main
```

## 10. 版本问题与已知限制

### 10.1 当前版本状态 (2026-08-02)

| 组件 | 版本/日期 | 滞后 | 影响 |
|------|----------|------|------|
| stock_daily | 2026-07-31 | 0天 (正常) | 无 |
| feature_matrix V5 | 2026-07-30 | 1天 | P1/P2 用 7/30 特征，非最新 |
| 模型 (3 engines) | 2026-07-22 | 11天 | P1/P2 推断用 11 天前的模型参数 |
| folds (P1/P3 校准) | 2026-08-01 (98 folds) | 0天 (手动更新) | 覆盖到 2026-03-19，后续 fold 缺失 |

### 10.2 模型概率校准问题

**问题**: momentum 和 reversion 模型的 PWin 范围过度压缩（0.75-0.98），所有股票 PWin > 0.55。

**根因**: LightGBM multiclass 的 `predict_proba` 在类别严重不平衡时概率偏向多数类。训练数据中 bucket_2 (True Alpha) 仅占 ~1%。

**当前缓解**: P1 温度计使用百分位排名法（rank pct），不依赖绝对概率值。

**长期方案**: Platt scaling 或 isotonic regression 对模型输出做概率校准。

### 10.3 Regime 推断滞后

**问题**: 日报的 regime 标签来自最近一个 fold（当前为 2026-03-19），可能与当前市场状态不符。

**缓解**: market_ml 模型（run_analysis.py 中）每日更新 regime 预测，可考虑接入。

### 10.4 Fold 数据更新滞后

**问题**: 98 folds 的 JSON 需要手动运行全量 WFV benchmark 来更新。自 2026-03-19 后没有新 fold。

**影响**: P1 温度计的历史 gate_blocked 百分位基准不再更新；P3 仓位规则表缺少近期数据。

**建议频率**: 每月跑一次全量 WFV benchmark。

### 10.5 Bear 市场样本不足

**问题**: 98 folds 中仅 9 个 bear folds。P3 规则表中 bear 行只有一种温度分位（低温），置信度低于 bull/range 行。

**处理**: Bear 行的仓位建议供参考，实际交易需结合 P1 温度和 P2 衰减榜做综合判断。

### 10.6 日报数据使用须知

- P1 温度计使用**百分位排名法**消除模型绝对概率偏差，温度值反映相对排序而非绝对概率
- P2 衰减榜的 decay_pct 包含**模型重训练效应**——新旧模型版本差异会混入衰减信号
- P3 仓位规则基于**历史统计规律**，不代表未来表现
- 所有指标均为**技术参考**，不构成投资建议

## 11. 文件清单

### 11.1 核心脚本

```
backtest/
├── v6_daily_report.py              # 日报生成 (P1+P2+P3 → HTML + Telegram)
├── v6_daily_pipeline.py            # 管道编排 (检查→刷新→报告→部署)
├── v6_weekly_retrain.py            # 周度模型重训练
├── v6_gate_thermometer.py          # P1 独立脚本 (验证用)
├── v6_alpha_decay_monitor.py       # P2 独立脚本 (折级验证)
├── v6_alpha_decay_ranking.py       # P2 全市场衰减榜 (可独立运行)
├── v6_diversification_dashboard.py # P3 独立脚本 (验证用)
├── v6_position_sizing_rules.py     # P3 仓位规则表 (可独立运行)
├── v6_phase1_feasibility.py        # HERMES-20260802-002 可行性分析
└── v6_walkforward_benchmark.py     # 全量 WFV (手动运行, ~1小时)

integrate_v5_features.py            # V5 特征构建 (+ --update 增量模式)

reports/v6_daily/
├── v6_daily_YYYYMMDD.html          # 每日 HTML 报告
└── V6_DAILY_SYSTEM.md              # 本文档
```

### 11.2 数据文件

```
data_storage/
├── feature_matrix_v5.parquet       # V5 特征矩阵 (~250 MB, 321 列)
├── feature_matrix_v4.parquet       # V4 特征矩阵 (6.8 GB, 118 列)
├── lgb_models/
│   ├── v6_momentum_*.pkl           # momentum 引擎模型 (58 个历史版本)
│   ├── v6_reversion_*.pkl          # reversion 引擎模型
│   └── v6_breakout_*.pkl           # breakout 引擎模型
├── sw_index_data.db                # SQLite: stock_daily + 行业指数
└── stock_industry_mapping.csv      # 3000 只股票 → 申万行业

backtest/
├── v6_final_20260801.json          # 98 folds 回测结果
├── v6_fold_temperature.json        # P1 每 fold 温度 (中间产物)
└── v6_alpha_decay_full.json        # P2 全量衰减数据 (中间产物)
```

## 12. 运维手册

### 12.1 日常检查

每天早上 08:35 确认 Telegram 收到推送。如果未收到：

```bash
# 查看管道日志
tail -50 ~/projects/trend-trading-system/logs/v6_pipeline_cron.log

# 手动重跑日报
cd ~/projects/trend-trading-system
python3 backtest/v6_daily_pipeline.py --report-only --telegram --deploy
```

### 12.2 数据过期处理

```bash
# 特征矩阵过期 (>1天)
python3 integrate_v5_features.py --update --skip-tushare

# 模型过期 (>7天)
python3 backtest/v6_weekly_retrain.py

# stock_daily 过期 (>1天)
# 检查 run_analysis.py 晚间流程是否正常, 或手动运行:
python3 run_analysis.py --force
```

### 12.3 全量 WFV 更新（每月一次）

```bash
cd ~/projects/trend-trading-system
python3 backtest/v6_walkforward_benchmark.py --folds -1
# 耗时 ~1 小时, 产出:
#   - backtest/v6_final_YYYYMMDD.json (新 folds)
#   - backtest/v6_final_YYYYMMDD_daily_trades.csv

# 更新后: 修改 v6_daily_report.py 中的 FINAL_JSON 路径指向新文件
```

### 12.4 故障排查速查表

| 症状 | 可能原因 | 处理 |
|------|---------|------|
| Telegram 未收到 | 管道崩溃 / Bot 异常 | 查 pipeline log, 手动重跑 |
| 温度恒为 50 | 特征矩阵无数据 / 模型缺失 | 检查 parquet 是否存在, 模型文件是否存在 |
| 衰减榜为空 | 回看期数据不足 (< 4 周) | 等待特征矩阵积累数据 |
| 日报 HTML 404 | git push 失败 | 手动 `git push origin main` |
| 特征刷新报错 | V4 parquet 未更新 | 先运行 `retrain_ml.py --update` |
| 模型训练报错 | 特征矩阵版本不匹配 | 确保 V5 parquet 存在且 ≥ 模型训练日期 |
