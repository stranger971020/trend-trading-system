"""
微信消息推送（Server酱）
- 通过 Server酱 (sct.ftqq.com) 推送到个人微信
- 需先在 sct.ftqq.com 注册获取 SendKey
"""
import json, logging, urllib.request, urllib.parse

from config import WECHAT_SENDKEY

logger = logging.getLogger(__name__)
WECHAT_API_URL = f"https://sct.ftqq.com/{WECHAT_SENDKEY}.send"


def send_wechat(title: str, content: str = "") -> bool:
    """发送微信推送消息"""
    if not WECHAT_SENDKEY or WECHAT_SENDKEY == "your_sendkey_here":
        logger.debug("微信 SendKey 未配置，跳过推送")
        return False

    data = urllib.parse.urlencode({"title": title, "desp": content}).encode("utf-8")
    try:
        req = urllib.request.Request(WECHAT_API_URL, data=data)
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") == 0:
                logger.info("✅ 微信推送成功")
                return True
            else:
                logger.warning("微信推送失败: %s", result.get("message", "未知错误"))
                return False
    except Exception as e:
        logger.warning("微信推送异常: %s", e)
        return False
