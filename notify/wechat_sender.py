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
    """发送微信推送消息（Server酱 GET 方式）"""
    if not WECHAT_SENDKEY or WECHAT_SENDKEY == "your_sendkey_here":
        logger.debug("微信 SendKey 未配置，跳过推送")
        return False

    params = urllib.parse.urlencode({"title": title, "desp": content})
    url = f"{WECHAT_API_URL}?{params}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            import json as _json
            try:
                result = _json.loads(body)
                if result.get("code") == 0:
                    logger.info("✅ 微信推送成功")
                    return True
                else:
                    logger.warning("微信推送失败: %s", result.get("message", body[:100]))
                    return False
            except (_json.JSONDecodeError, TypeError):
                # 非 JSON 响应（可能HTML）
                if "成功" in body or "推送" in body:
                    logger.info("✅ 微信推送成功")
                    return True
                logger.warning("微信推送返回异常: %s", body[:100])
                return False
    except Exception as e:
        logger.warning("微信推送异常: %s", e)
        return False
