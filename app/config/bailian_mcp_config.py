# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _normalize_mcp_base_url(url: str | None) -> str | None:
    if not url:
        return url

    normalized = url.strip()
    if normalized.endswith("/sse"):
        normalized = normalized[:-4] + "/mcp"

    # DashScope official path uses `WebSearch`; normalize old lowercase config.
    normalized = normalized.replace("/mcps/webSearch/", "/mcps/WebSearch/")
    return normalized


# 定义mcp的服务配置
@dataclass
class McpConfig:
    mcp_base_url: str
    api_key : str

mcp_config = McpConfig(
    mcp_base_url=_normalize_mcp_base_url(os.getenv("MCP_DASHSCOPE_BASE_URL")),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)


# ──────────────────────────────────────────────────────────
# 📖 阅读导航
# 上一篇: app/config/mineru_config.py
# 下一篇: app/core/logger.py
# ──────────────────────────────────────────────────────────
