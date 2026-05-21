# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()

# 定义minerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_token : str
    bypass_proxy: bool

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_token=os.getenv("MINERU_API_TOKEN"),
    bypass_proxy=os.getenv("MINERU_BYPASS_PROXY", "true").strip().lower() in {"1", "true", "yes", "on"},
)

MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_token
MINERU_BYPASS_PROXY = mineru_config.bypass_proxy


# ──────────────────────────────────────────────────────────
# 📖 阅读导航
# 上一篇: app/config/minio_config.py
# 下一篇: app/config/bailian_mcp_config.py
# ──────────────────────────────────────────────────────────
