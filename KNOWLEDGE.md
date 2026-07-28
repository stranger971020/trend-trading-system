# A股趋势交易系统 — 项目知识库

> 持久性知识，记录关键设计决策和系统演进历史。

---

## 架构决策

### D1. 为什么用 GitHub Pages 而非独立服务器？

**决策：** 报告 HTML 部署到 GitHub Pages，从 `main` 分支根目录 serving。

**理由：**
- 零运维成本，无需 VPS
- 自动化：`git push` 后 GitHub Pages 自动构建部署（1-2min 延迟）
- GitHub Pages URL `https://stranger971020.github.io/trend-trading-system/` 可直接在 Telegram 中点击

**代价：** push 后 1-2 分钟 URL 返回 404，需等待构建完成。

### D2. 为什么用 Telegram 而非微信作为推送通道？

**决策：** 主推送为 Telegram Bot API，微信（Server 酱）为辅。

**理由：**
- Telegram API 无消息长度硬限制（微信限制严格）
- 支持 HTML 富文本格式
- 不需要用户互动确认（微信需要关注公众号）
- 双通道互为备份

### D3. 为什么 ML 模型按 micro/mid/mega 三分组而非统一模型？

**决策：** 按流通市值三分组（micro < 30亿, mid 30-200亿, mega > 200亿），每组训练独立 LightGBM 模型。

**理由：**
- 不同市值区间的特征表现差异大（大票看基本面，小票看资金流）
- 三分组在回测中验证优于统一模型
- 分组后特征工程可差异优化

### D4. 为什么 `rerank_with_ml()` 内部加载模型而非外部传入？

**决策：** `rerank_with_ml()` 内部通过 `load_model(group)` 加载，调用方传 `models=None` 即可。

**理由：**
- 简化调用接口：调用方无需管理模型生命周期
- 避开跨模块依赖：`load_model()` 的参数 `group` 需在训练时确定
- 特征从内部缓存（parquet）加载，不走 `build_feature_matrix()` 全量计算（30min 瓶颈）

**历史教训：** 之前调用 `load_model(stock_result, stock_daily_df)` 缺少 `group` 参数导致崩溃 → try-except 吞异常 → 数周未被发现。

### D5. 为什么保持 A股日报 HTML 生成接口但置空？

**决策：** `html_path = None` 保留，不删除生成函数引用，且 git push 已从 `if html_path:` 解耦。

**理由：**
- 避免删除引用导致 import 链断裂（`generate_html_report` 仍被 import）
- git push 不应依赖日报是否存在（两个不同的关注维度）
- 如需恢复日报功能，只需重新填充生成逻辑，不改调用方

---

## 系统演进

### v2.0 (2026-06-14) — 五阶段全系统完成

初始版本，包含 16 分析模块、Telegram 推送、GitHub Pages 展示。`_git_push_reports()` 在此版本中被意外放入 `if html_path:` 死代码块。

### v4 ML (2026-07 中旬) — 模型推理集成

- 118 特征全量 Walk-Forward 训练
- 三分组 LightGBM 模型
- 日频推理 → top5 注入报告
- `rerank_with_ml()` API 确立

### 2026-07-28 — 系统性排障

- 修复 8 个断裂 Bug（参见 JOURNAL.md 2026-07-28）
- 新增三层 watchdog 机制
- 确立变更留痕规范
- 确立报告管道接口契约

---

## 运维要点

### 服务清单

| 服务 | 守护方式 | 故障表现 |
|---|---|---|
| `telegram-ai-bridge` | launchd + crontab watchdog 每30min | Bot 不回复 |
| `run_analysis.py` | crontab 工作日 09:00 / 21:30 | 未收到推送 |
| 健康检查 | launchd + crontab backup 每4h | EX_CONFIG 静默停机 |
| GitHub Pages | push 后自动构建 | 1-2min 内 404 为正常 |

### 故障诊断步骤

1. 查 `logs/cron.log` — 分析报告运行日志
2. 查 `launchctl list` — 检查 bot 进程
3. 查 `crontab -l` — 检查定时任务是否丢失
4. 查 GitHub Pages API `status` — 是否 building
