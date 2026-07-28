# A股趋势交易系统 — 会话日志

## 2026-07-28 系统性排障与修复

### 做了什么？

1. **Claude Bridge 断联修复**
   - launchd 在 macOS 睡眠后丢失 bootstrap 状态，`KeepAlive=true` 不工作
   - 执行 `launchctl bootstrap gui/$(id -u) ...` 恢复，增加 30min 看门狗（`service-resurrection.sh`）

2. **GitHub Pages 404 修复**
   - `_git_push_reports()` 自 2026-06-15 创建起就一直放在 `if html_path:` 死代码块内
   - 早期 A股日报尚能生成，git push 实际在执行；A股日报废弃后 `html_path` 永久 `None`，git push 静默停摆数日
   - 将 git push 移出死代码块，验证 GitHub Pages 返回 200

3. **ML 模型从未运行修复**
   - `load_model()` 缺少 `group` 参数，调用即崩溃，全程回退到线性评分
   - 改为直接调用 `rerank_with_ml(stocks, daily_df, models=None)`，内部从缓存加载模型
   - ML Top5 注入 5 只 ✅

4. **Telegram 格式修复**
   - `report_text` 为 None 后 line 838（拼接 GitHub 链接）crash → 加 None guard
   - `<` `>` 在 `parse_mode="HTML"` 下未转义 → 转义为 `&lt;` `&gt;`
   - 恢复格式缺失的分行符和视觉分界

5. **launchd EX_CONFIG 根因定位**
   - `StandardOutPath` 包含空格 → launchd 在运行脚本前即报 EX_CONFIG(78) → 永久禁用定时任务
   - 修复：移除 plist 的 stdout/stderr，由 wrapper 脚本 `>> "$LOG" 2>&1` 自管理日志

6. **健康检查系统重建**
   - `bot-health-check.sh` + `service-resurrection.sh` + crontab 三层 watchdog
   - 移除 wrapper 脚本的 `set -e` 防止 EX_CONFIG 传播

7. **修复必同步检查机制**
   - 新增规则：任何修复须检查 `daily_bot_doc_check.py` 的 `HEALTH_CHECKS` / `FILE_GROUPS` 是否有对应监控

8. **变更留痕 + 溯源举证规范**
   - 新增 CLAUDE.md 规则：改核心文件须 Changelog + grep 全文件 + dry-run 验证
   - 新增规则：归因他人前查 git blame / 文件修改时间

### 踩了哪些坑？

1. **归因错误（3次）**
   - 错误推定 Hermes 修改了 `report_text=None` → 实际是自己上一会话改的
   - 错误推定 Hermes 修改了 `.bak` 文件 → 无证据，.bak 是干净的 pre-Hermes 备份
   - 教训：无 git blame 数据不得归因；有疑问直接提问而非推定

2. **PEP 709 异常变量作用域**
   - Python 3.12+ 在 `except ... as e:` 块退出后删除 `e`，mypy 静态检查据此报错
   - 以为是 mypy 索引问题，实际是 PEP 709 行为变化

3. **死代码块隐形吞噬功能**
   - 多会话协作下，A 会话废弃变量设为 None，B 会话新功能放在 `if var:` 内 →
   - git push、Obsidian 同步等功能静默消失数日无人察觉
   - 教训：变量状态变更必须 grep 全文件 + Changelog 标注影响范围

4. **ML 模型加载流程未完整验证**
   - `rerank_with_ml()` 签名与调用不一致持续数周未被发现
   - 因为 `try-except` 吞掉了异常，仅日志有一条 WARNING
   - 教训：try-except 要明确日志级别 ERROR 而非 WARNING

### 下一步做什么？

1. **监控明早 09:00 晨间分析首跑**
   - 这是首个带全部修复的运行：ML 模型 + GitHub 推送 + 完整格式
   - 确认 cron.log 中无异常

2. **考虑增加 ML 特征缓存新鲜度监控**
   - 当前 `rerank_with_ml()` 从缓存加载特征；缓存若过旧（>24h）应告警

3. **GitHub token 过期预警**
   - token 明文在 remote URL 中，过期后 push 会失败
   - 可考虑切 SSH key 或在 token 中设置更长过期时间
