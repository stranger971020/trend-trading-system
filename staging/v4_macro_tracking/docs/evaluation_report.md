# 《每日交易参考》v4 — 评估报告与对接方案

> **生成日期:** 2026-07-29  
> **最后修改:** 2026-07-29 (v4 修正)  
> **Staging 路径:** `/Users/jren/projects/trend-trading-system/staging/v4_macro_tracking/`  
> **状态:** ✅ 全部 29/29 测试通过 (含 4 项实时数据测试)

---

## 一、交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| Staging 完整目录结构 | `staging/v4_macro_tracking/` | ✅ |
| 核心模块 | `tools/macro_pullback_tracker.py` | ✅ |
| 外部化 YAML 配置 | `macro_config_default.yaml` | ✅ |
| 集成测试 (模拟 + 实时) | `tests/test_macro_staging.py` | ✅ 29/29 |
| 文档 | `README.md` | ✅ |
| CLI 独立运行 | `python3 tools/macro_pullback_tracker.py` | ✅ |

## 二、模块架构

```
                        ┌──────────────────────────────────┐
                        │       MacroPullbackTracker        │
                        │    (宏观情绪监测与反弹预估)        │
                        └──────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │  ThresholdCalc   │   │ HistPatternMapper│   │ MicroSignalChecker│
   │  ──────────────  │   │  ──────────────  │   │  ──────────────  │
   │  RSI(14)         │   │  10年暴跌事件检测  │   │  ETF溢价/份额   │
   │  ATR 极值突破    │   │  修复天数分布     │   │  长下影线形态    │
   │  均线乖离率      │   │  每日反弹胜率     │   │  融资盘极限      │
   │  成交量偏离度    │   │  Best/Worst Case  │   │                  │
   └──────────────────┘   └──────────────────┘   └──────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                        ┌──────────────────────────────────┐
                        │  IndicatorEngine                 │
                        │  (RSI / ATR / MA / Vol / 形态)    │
                        └──────────────────────────────────┘
                                    │
                        ┌──────────────────────────────────┐
                        │  DataFetcher                     │
                        │  Tushare Pro (主) + akshare (备)  │
                        └──────────────────────────────────┘
```

## 三、测试结果详表

| 测试组 | 用例数 | 通过 | 覆盖内容 |
|--------|--------|------|---------|
| TestConfigLoading | 3 | 3/3 | 默认配置、外部YAML加载、缺失文件回退 |
| TestEnvironmentIsolation | 2 | 2/2 | Staging 环境独立性验证 |
| TestIndicatorEngine | 6 | 6/6 | RSI/ATR/MA乖离/成交量/下影线 |
| TestThresholdCalculator | 4 | 4/4 | RSI超卖/ATR突破/乖离极值/量偏离 |
| TestHistoricalPatternMapper | 4 | 4/4 | 事件检测、修复计算、统计、胜率曲线 |
| TestMicroSignalChecker | 2 | 2/2 | 下影线检测、融资信号 |
| TestEndToEnd | 1 | 1/1 | 模拟数据端到端流程 |
| TestOutputFormat | 3 | 3/3 | Dict结构、JSON序列化、YAML Header |
| *TestLiveDataFetching | 4 | 4/4 | *CSI300行情、全流程、融资数据、ETF |
| **合计** | **29** | **29/29** | |

\* 需 `--live` 参数 + Tushare Token

## 四、对接方案: 无侵入式 Hook

### 方案 A: 子进程调用（零耦合，推荐）

```python
# 在 run_analysis.py 中任意位置，以子进程方式调用
import subprocess, json

def run_macro_tracker() -> dict:
    """调用宏观监测模块，不引入任何 Python import 依赖"""
    staging_root = "/path/to/staging/v4_macro_tracking"
    cmd = ["python3", f"{staging_root}/tools/macro_pullback_tracker.py",
           "--format", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return {}
```

**优点:** 零依赖、不污染命名空间、可独立升级  
**缺点:** 每次调用有 ~2s 子进程开销

### 方案 B: Python Hook 调用（低耦合）

```python
# 在 run_analysis.py 的 generate_report 函数中
# 保持 100% 解耦，作为 Hook 注入

def add_macro_to_report(report_dict: dict) -> dict:
    """Hook: 将宏观监测数据注入报告"""
    import sys
    staging_path = "/path/to/staging/v4_macro_tracking"
    if staging_path not in sys.path:
        sys.path.insert(0, staging_path)
    
    from tools.macro_pullback_tracker import MacroPullbackTracker
    tracker = MacroPullbackTracker(
        config_path=f"{staging_path}/macro_config_default.yaml"
    )
    report = tracker.run()
    report_dict["macro_tracker_v4"] = tracker.to_dict(report)
    return report_dict
```

**优点:** 无子进程开销、可传参  
**缺点:** 需确保 staging 路径在 sys.path 中

### 方案 C: 独立调度（最强隔离）

将 `macro_pullback_tracker.py` 注册为独立 cron job，输出 JSON：

```bash
# crontab 条目
0 20 * * 1-5 cd /path/to/staging/v4_macro_tracking && \
    python3 tools/macro_pullback_tracker.py --format json \
    --output /path/to/reports/macro_report_latest.json
```

### 输出对接格式

```yaml
# 嵌入 daily_report.md YAML Header
macro_tracker_v4:
  generated_at: "2026-07-29 20:00:00"
  index:
    code: "000300.SH"
    name: "沪深300"
    current_price: 4569.52
  confidence_rate: 0.41
  overall_verdict: "谨慎观望"
  recovery_window:
    mean_days: 24
    adjusted_mean_days: 24
    recovery_rate: 89.1
    key_win_rates:
      T+1: 28.5
      T+5: 52.0
      T+10: 62.6
      T+30: 81.3
  micro_signals_detected: 1
```

## 五、生产部署检查清单

当评估通过后，按以下步骤合并：

- [ ] **步骤 1**: 将 `tools/macro_pullback_tracker.py` 复制到主项目对应目录
- [ ] **步骤 2**: 将 `macro_config_default.yaml` 复制到主项目配置目录
- [ ] **步骤 3**: 在 `run_analysis.py` 中选择一种 Hook 方案接入
- [ ] **步骤 4**: 运行 `python3 run_analysis.py --dry-run` 验证不崩溃
- [ ] **步骤 5**: 运行 `python3 run_analysis.py --force` 生成完整报告验证输出
- [ ] **步骤 6**: 确认 `daily_bot_doc_check.py` 中 HEALTH_CHECKS 已覆盖新模块

## 六、约束核查

| 约束要求 | 满足情况 |
|----------|---------|
| 复用 akshare/tushare 数据源 | ✅ 复用 config.py Tushare Token + 备用 akshare |
| 输出兼容 daily_report.md YAML Header | ✅ `to_markdown_yaml_header()` 方法 |
| 不硬编码阈值 | ✅ 全部通过 YAML 外部注入 |
| 不修改原代码库 | ✅ Staging 完全独立 |
| 模块可独立运行 | ✅ CLI + Python API 双入口 |

## 七、v4 修正记录 (2026-07-29)

### 修正背景
基于 CSI300（2016-2026，共9.7年/2447个交易日）实际数据的系统验证，发现：
- RSI < 15、ATR ≥ 2.5、成交量 ≤ 0.4x 三个阈值在9.7年间**触发0次**，导致置信度评分中50%权重（0.25+0.15+0.10）从未被激活
- 修复时间估算仅依赖RSI乘数（0.7/0.85），完全忽略跌幅深度（Pearson r≥0.3的显著正相关）

### P0 修正：阈值配置（3项）
| 维度 | 原值 → 新值 | 文件 | 行 |
|:---|:---:|:---|:---:|
| RSI 极端超卖 | 15 → **18** | `macro_config_default.yaml` | L23 |
| ATR 波动突破 | 2.5 → **2.0** | `macro_config_default.yaml` | L30 |
| 地量萎缩比率 | 0.4 → **0.6** | `macro_config_default.yaml` | L53 |

嵌入式默认配置（`DEFAULT_CONFIG_YAML` in `macro_pullback_tracker.py`）同步更新。

### P1 修正：修复时间双因子调节（核心改进）
**`_estimate_recovery_window()`** 中替换 RSI-only 调整逻辑为：

```
adjusted_mean_days = mean_days × drawdown_factor × rsi_factor
```

| 因子 | 条件 | 系数 | 说明 |
|:---|:---|---:|:---|
| 跌幅因子（主） | 回撤 < 5% | 1.0 | 轻微调整 |
| | 回撤 < 10% | 1.3 | 中等跌幅 |
| | 回撤 < 20% | 1.6 | 深度跌幅 |
| | 回撤 ≥ 20% | 2.0 | 极端跌幅 |
| RSI因子（辅） | RSI ≤ 25 | 0.9 | 极端超卖（略快） |
| | RSI ≤ 35 | 1.0 | 超卖（中性） |
| | RSI > 35 | 1.1 | 非超卖（略慢） |

**修正确认示例**（2026-07-29 沪深300 实时数据）：
```
旧: RSI=42 → adjusted_mean_days = 24.5天（完全忽略 -20.8% 回撤）
新: 回撤-20.8%(系数2.0) × RSI42(系数1.1) → adjusted_mean_days = 53天 ✅
```

### 验证结果
- 单元测试: 25/25 通过（模拟数据）
- 实时数据测试: 29/29 通过（含4项Tushare实时数据）
- CLI 独立运行: JSON/md/text 三种格式均正常输出
- 置信度评分: 现行场景 RSI=42(正常) → RSI置信度0.5分(正常区间)；若触发新阈值 RSI<18 → 置信度将升至1.0分
