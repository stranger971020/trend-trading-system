"""
A股趋势交易系统 - 全局配置（示例）
复制为 config.py 并填入你自己的 API 凭证
"""

import os

# ============================================================
# API 凭证（必填）
# ============================================================
TUSHARE_TOKEN = "你的_Tushare_Token"
TELEGRAM_BOT_TOKEN = "你的_Telegram_Bot_Token"
TELEGRAM_CHAT_ID = "你的_Telegram_Chat_ID"

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_storage")
DB_PATH = os.path.join(DATA_DIR, "sw_index_data.db")
MAPPING_CSV = os.path.join(DATA_DIR, "sw_l1_mapping.csv")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============================================================
# 数据获取参数
# ============================================================
INITIAL_FETCH_DAYS = 120
API_RATE_LIMIT = 0.35
AK_API_RATE_LIMIT = 0.30
API_RETRY_COUNT = 2
API_RETRY_DELAY = 5.0

# ... (其余参数与 config.py 完全一致，详见仓库中的完整文件)
