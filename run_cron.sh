#!/bin/bash
# A股趋势交易系统 - Cron 启动脚本
# 用法: 将此脚本加入 crontab
# 0 20 * * 1-5 /Users/jren/projects/trend-trading-system/run_cron.sh

export PATH="/Users/jren/miniforge3/bin:$PATH"
export HOME="/Users/jren"

echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始执行..."
echo "========================================"

python3 run_analysis.py >> logs/cron.log 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成 (exit=$EXIT_CODE)"
echo ""
