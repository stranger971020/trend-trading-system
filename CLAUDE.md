# A股趋势交易系统 — 项目管理规则

> 项目级规则，补充 `~/.claude/CLAUDE.md` 中的全局规则。

---

## 架构决策记录

### 报告管道接口契约

修改 `run_analysis.py` / `trading_plan_report.py` 等核心文件时，须区分**稳定接口**和**可变实现**：

**稳定接口（不可随意修改，改则断下游）：**
- `report_text` — Telegram 推送拼接链接 + 发送消息体
- `html_path` — 控制台打印（已从 git push 解耦）
- `trading_html_path` — Obsidian Vault 同步
- `stock_result["stocks"]` — 线性评分 + ML 重排共用
- Telegram `parse_mode="HTML"` — 所有消息体中的 `<` `>` 须转义为 `&lt;` `&gt;`
- 报告文件名 `trading_YYYYMMDD_slot.html` — GitHub Pages URL 依赖此格式

**可变实现（安全修改区）：**
- ML 评分逻辑、选股参数、报告文案、HTML 样式、健康检查逻辑

> 拿不准就当接口处理——grep 全文件确认下游后再动手。

### 常用命令

```bash
# 正常运行（晚间）
python3 run_analysis.py

# 早间运行
python3 run_analysis.py --morning

# 验证不崩溃
python3 run_analysis.py --dry-run

# 强制运行（跳过交易日检查）
python3 run_analysis.py --force
```

### 提交规范

遵循项目中已有的 commit message 风格：
```
📊 日报更新 YYYYMMDD        # 报告推送
🐛 修复...                    # Bug 修复
🎯 新增...                    # 新功能
```

---

## 文件级规范

### run_analysis.py / trading_plan_report.py

结构性改动须在文件头部写入 Changelog：

```python
# ── Changelog ──
# 2026-07-28 Claude: 废弃 A股日报，report_text=None
#               告警: 下游所有 report_text 引用需加 None guard
# ─────────────
```

改动后必须 `--dry-run` 验证不崩溃，再 `--force` 完整运行。

### 健康检查同步

任何涉及配置、路径、依赖的修复完成后，须检查 `daily_bot_doc_check.py` 的 `HEALTH_CHECKS` 和 `FILE_GROUPS` 是否有对应监控项，无则新增。

---

## 与本项目无关

本项目不涉及 Hermes 任务处理、Obsidian 记忆双写、复用优先搜索等全局规则——这些已在 `~/.claude/CLAUDE.md` 中定义，不重复写入本项目规则。
