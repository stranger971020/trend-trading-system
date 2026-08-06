#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# rebuild_full_universe_resume.sh — 全量池重建「断点续跑」
#
# V4 重建/V4 walk-forward/V5 重建 已完成（见 rebuild_full_universe.log），
# 本脚本从第 4 步（V6 walk-forward 审计）继续。
#
# 为什么从 4 续: step1-3 产物已存在(feature_matrix_v4/v5.parquet),
#   重跑浪费 ~40 分钟; step4 曾因 --stocks None 的 %d 日志格式崩溃已修复。
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python3"
LOG="logs/rebuild_full_universe.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "── 4. V6 walk-forward 审计（续跑）──"
$PY backtest/v6_walkforward_benchmark.py --n-folds 98 --since 20240101 \
    --out backtest/v6_audit_full_$(date +%Y%m%d) 2>&1 | tee -a "$LOG"

log "── 5. V6 周度重训 ──"
$PY backtest/v6_weekly_retrain.py 2>&1 | tee -a "$LOG"

# 5.5 promote 新审计 JSON → 审计统计脚本/v6_daily_report 读取的路径
#     (这些脚本硬编码 v6_final_20260801.json; 旧文件已备份到 backup/20260805/)
latest_json=$(ls -t backtest/v6_audit_full_*.json 2>/dev/null | head -1)
if [ -n "$latest_json" ]; then
  cp "$latest_json" backtest/v6_final_20260801.json
  cp "$latest_json" backtest/v6_final_$(date +%Y%m%d).json
  log "── 5.5 promote 审计 JSON: $latest_json → v6_final_20260801.json / v6_final_$(date +%Y%m%d).json"
else
  log "⚠️ 未找到 v6_audit_full_*.json，跳过 promote"
fi

log "── 6. 审计统计重建（读新 JSON）──"
$PY backtest/v6_gate_thermometer.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_position_sizing_rules.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_alpha_decay_monitor.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_diversification_dashboard.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_phase1_feasibility.py 2>&1 | tee -a "$LOG"

log "✅ 全量池重建（续跑）完成。"
log "  后续: 按新审计结果更新 v6_daily_report.py 的 POSITION_RULES 表"
log "        端到端验证: python3 backtest/v6_daily_report.py"
