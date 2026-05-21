"""
文件导入流程图端到端测试

本测试脚本验证 LangGraph 文件导入流程图（kb_import_app）的完整执行。
使用 "万用表RS-12的使用.pdf" 作为测试文件，逐步执行导入图的所有节点，
并输出最终状态和图结构。

运行方式：
    python test/04-test_graph_flow.py
"""

import json

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

logger.info("===== 开始测试 =====")

initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
final_state = None

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in kb_import_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"节点: {key}")
        final_state = value

# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("图结构:")
# uv add grandalf
kb_import_app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")