"""
Telegram 消息推送
- 使用 urllib 直接调用 Telegram Bot API
- 自动分段超长消息
"""

import json
import logging
import time
import urllib.parse
import urllib.request

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_SEND_DELAY,
)

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


def send_single_message(text: str, parse_mode: str = "HTML") -> bool:
    """发送单条 Telegram 消息。

    Args:
        text: 消息文本
        parse_mode: "HTML" | "Markdown" | "" （空字符串表示纯文本）

    Returns:
        True 如果发送成功
    """
    # 如果 parse_mode 为空，不传该参数
    if parse_mode:
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
        }
    else:
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
        }

    try:
        encoded_data = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(TELEGRAM_API_URL, data=encoded_data)
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            if result.get("ok"):
                return True
            else:
                logger.warning("Telegram API 返回错误: %s", result.get("description"))
                return False
    except urllib.error.URLError as e:
        logger.error("Telegram 网络请求失败: %s", e)
        return False
    except Exception as e:
        logger.error("Telegram 发送异常: %s", e)
        return False


def _split_by_paragraphs(text: str, max_len: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """按段落边界智能分段。

    策略：
    1. 先按双换行（段落）分割
    2. 累加段落，超过 max_len 时发送当前批次
    3. 如果单个段落本身就超长，按字符硬截断
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果单段落超长，硬截断
        if len(para) > max_len:
            # 先保存当前 chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 硬截断长段落
            for i in range(0, len(para), max_len):
                chunks.append(para[i:i + max_len].strip())
            continue

        # 检查加入当前段落后是否超长
        test_chunk = current_chunk + "\n\n" + para if current_chunk else para
        if len(test_chunk) > max_len:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = test_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def send_report(report_text: str) -> dict:
    """发送报告到 Telegram，自动分段。

    Args:
        report_text: 完整的报告文本

    Returns:
        dict: {"total": N, "sent": M, "failed": F}
    """
    chunks = _split_by_paragraphs(report_text)
    total = len(chunks)
    sent = 0
    failed = 0

    logger.info("Telegram 推送: 共 %d 段消息", total)

    for i, chunk in enumerate(chunks, 1):
        # 添加分段标记
        if total > 1:
            if i == 1:
                label = f"{chunk}\n\n<i>(1/{total} — 续)</i>"
            elif i == total:
                label = f"<i>({i}/{total} — 完)</i>\n\n{chunk}"
            else:
                label = f"{chunk}\n\n<i>({i}/{total} — 续)</i>"
        else:
            label = chunk

        success = send_single_message(label)
        if success:
            sent += 1
            logger.info("  [%d/%d] ✓ 发送成功 (%d 字符)", i, total, len(chunk))
        else:
            failed += 1
            logger.warning("  [%d/%d] ✗ 发送失败", i, total)

            # 尝试用纯文本模式重试一次
            if failed > 0:
                logger.info("  尝试纯文本模式重试...")
                time.sleep(1)
                success2 = send_single_message(chunk, parse_mode="")
                if success2:
                    sent += 1
                    failed -= 1

        # 段间延迟
        if i < total:
            time.sleep(TELEGRAM_SEND_DELAY)

    return {"total": total, "sent": sent, "failed": failed}
