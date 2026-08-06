#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# rebuild_full_universe.sh — 全量池扩池后的一次性重建脚本（周末跑，约 4-5h）
#
# 依赖顺序（严格串行）:
#   1. stock_daily 全量回填（阶段2 已完成/确认）
#   2. V4 全量重建（retrain_ml --rebuild）
#   3. V4 walk-forward（--walk-forward）
#   4. V5 全量重建（integrate_v5_features 全量模式，非 --update）
#   5. V6 walk-forward 审计（v6_walkforward_benchmark）
#   6. V6 周度重训（v6_weekly_retrain）
#   7. 审计统计重建（温度/仓位/衰减/多样化仪表盘）
#
# 前置: 阶段0 备份已完成（lgb_models/parquet/映射CSV 已备份到 backup/20260805/）
# 用法: bash scripts/rebuild_full_universe.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python3"
LOG="logs/rebuild_full_universe.log"
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 0) 前置检查: 股票池已扩
log "── 0. 前置检查 ──"
n_pool=$($PY -c "
import sys; sys.path.insert(0,'.')
from data.stock_industry_mapping import load_stock_universe
print(len(load_stock_universe()))")
log "股票池: $n_pool 只"
if [ "$n_pool" -lt 4500 ]; then log "❌ 股票池未扩到全量，中止"; exit 1; fi

# 1) V4 全量重建（写入 feature_matrix_v4.parquet）
log "── 1. V4 全量重建 ──"
$PY analysis/retrain_ml.py --rebuild 2>&1 | tee -a "$LOG"

# 2) V4 walk-forward（全量池）
log "── 2. V4 walk-forward ──"
$PY analysis/retrain_ml.py --walk-forward 2>&1 | tee -a "$LOG"

# 3) V5 全量重建（非 --update，全量模式）
log "── 3. V5 全量重建 ──"
$PY integrate_v5_features.py 2>&1 | tee -a "$LOG"

# 4) V6 walk-forward 审计（新池新模型）
log "── 4. V6 walk-forward 审计 ──"
$PY backtest/v6_walkforward_benchmark.py --n-folds 98 --since 20240101 \
    --out backtest/v6_audit_full_$(date +%Y%m%d) 2>&1 | tee -a "$LOG"

# 5) V6 周度重训（生成 lgb_models/v6_* 最新模型）
log "── 5. V6 周度重训 ──"
$PY backtest/v6_weekly_retrain.py 2>&1 | tee -a "$LOG"

# 6) 审计统计重建（温度/仓位/衰减/多样化仪表盘）
log "── 6. 审计统计重建 ──"
$PY backtest/v6_gate_thermometer.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_position_sizing_rules.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_alpha_decay_monitor.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_diversification_dashboard.py 2>&1 | tee -a "$LOG"
$PY backtest/v6_phase1_feasibility.py 2>&1 | tee -a "$LOG"

# 7) promote 新审计 JSON 到 v6_daily_report 读取路径
log "── 7. promote 审计 JSON ──"
latest_json=$(ls -t backtest/v6_audit_full_*.json 2>/dev/null | head -1)
if [ -n "$latest_json" ]; then
  cp "$latest_json" backtest/v6_final_$(date +%Y%m%d).json
  log "新审计 JSON: $latest_json → backtest/v6_final_$(date +%Y%m%d).json"
  log "  注意: v6_daily_report.py 用 --final-json 指定，或更新 FINAL_JSON 常量"
else
  log "⚠️ 未找到 v6_audit_full_*.json，跳过 promote"
fi

log "✅ 全量池重建完成。请核对:"
log "  1. IC/WR 对比旧基准 (IC=0.1111, WR=46.4%)"
log "  2. 健康检查: python3 ~/projects/obsidian-hermes/scripts/daily_bot_doc_check.py"
log "  3. 端到端日报: python3 backtest/v6_daily_report.py"
