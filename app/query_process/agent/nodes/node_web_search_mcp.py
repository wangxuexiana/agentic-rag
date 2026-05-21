"""
联网搜索节点 (Web Search MCP Node)

本模块实现 Agentic RAG 查询流程中的「联网搜索」环节。
通过阿里云百炼 MCP（Model Context Protocol）服务调用 WebSearch 工具，
获取与用户问题相关的互联网信息，补充本地知识库覆盖不到的时效性内容。

核心流程：
1. 检查 Router 是否启用 web_search 工具，未启用则跳过
2. 从 state 中获取改写后的查询词
3. 通过 MCP SSE 客户端异步调用百炼 WebSearch 工具
4. 解析返回的 JSON 结果，提取 title/url/snippet
5. 返回结构化搜索结果供 Rerank 节点使用

同步/异步桥接：
- LangGraph 节点是同步函数，但 MCP 调用是异步的
- _run_async_from_sync() 负责在同步上下文中安全执行协程
- 兼容 FastAPI 事件循环场景（避免 asyncio.run() 嵌套报错）

当 Router 未启用 web_search 工具时，本节点自动跳过，返回空结果。
"""

import sys
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agents.mcp import MCPServerStreamableHttp

from app.utils.task_utils import add_done_task, add_running_task
from app.config.bailian_mcp_config import mcp_config

from app.core.logger import logger


# ==================== Java 开发者阅读提示 ====================
# 这个节点负责联网搜索。
# 它不会直接改写本地知识库，而是把外部网页结果作为“补充证据”带回查询链路。
#
# 这个文件比较特别的一点是：
# - LangGraph 节点本身是同步函数
# - 但 MCP/WebSearch 调用通常是异步的
# 所以这里会有“同步代码桥接异步调用”的处理。
# ===========================================================


def node_web_search_mcp(state):
    """
    LangGraph同步节点函数：处理MCP搜索逻辑，作为整个搜索流程的入口。

    该节点会调用 mcp_call 异步函数获取搜索结果，并将其解析为结构化数据存储到 state 中。

    :param state: LangGraph的全局状态对象，包含 session_id, rewritten_query 等信息
    :return: 字典，包含结构化的搜索结果 web_search_docs，供后续节点使用
    """
    logger.info("---node_web_search_mcp 开始处理---")

    # 节点职责：
    # 1. 只在 run_web_search=True 时执行
    # 2. 用 rewritten_query 调外部 MCP WebSearch
    # 3. 把返回结果规范化为 title/url/snippet 结构
    # 4. 作为本地 KB 之外的补充证据写入 web_search_docs
    # 1. 标记任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    if False:
        logger.info("WEB_SEARCH_ENABLED=false，外部网页搜索服务已关闭。")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"web_search_docs": []}

    # 如果 Router 没有启用 web_search 工具，这个节点直接跳过
    if not state.get("run_web_search", False):
        logger.info("Router 未启用 web_search，当前节点跳过执行。")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"web_search_docs": []}

    # 2. 获取查询词
    query = state.get("rewritten_query", "")
    if not query:
        # 尝试回退到原始查询
        query = state.get("original_query", "")

    docs = []

    # 3. 执行搜索
    if query:
        try:
            # 同步-异步桥接：通过asyncio.run()执行异步的mcp_call函数
            logger.info(f"启动异步 MCP 调用，Query: {query}")

            # ======================================================================
            # MCP 返回结果格式解析说明
            # ----------------------------------------------------------------------
            # result 是一个 CallToolResult 对象 (定义在 agents.mcp.types 中)
            # result.content 是一个 TextContent 对象的列表，通常只有一项
            # result.content[0].text 是一个 JSON 字符串，包含实际的搜索结果
            #
            # 示例数据结构：
            # result.content[0].text = """
            # {
            #   "pages": [
            #     {
            #       "title": "HAK 180 烫金机使用手册",
            #       "url": "http://example.com/manual",
            #       "snippet": "在出厂默认状态下，若想设置局部转印..."
            #     },
            #     ...
            #   ]
            # }
            # """
            # ======================================================================
            # 不能直接 asyncio.run()。
            # 非流式 /query 会在 FastAPI 事件循环线程里同步调用 query_app.invoke()，
            # 这里如果再嵌套 asyncio.run()，运行时就会报错。
            result = _run_async_from_sync(mcp_call(query))

            # 4. 解析结果
            if result and not result.isError and result.content:
                # 解析MCP原始结果：提取文本内容并转为JSON对象
                # result.content 通常是一个列表，第一项包含文本结果
                raw_text = result.content[0].text
                try:
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []

                    logger.info(f"MCP 返回原始页面数量: {len(pages)}")

                    # 遍历结果，统一封装为结构化格式
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()

                        # 过滤无核心摘要的结果
                        if not snippet:
                            continue

                        docs.append({"title": title, "url": url, "snippet": snippet})

                except json.JSONDecodeError:
                    logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")
            else:
                if result and result.isError:
                    logger.error(f"MCP 返回错误: {result}")
                else:
                    logger.warning("MCP 返回结果为空或无效")

            logger.info(f"结构化搜索结果数量: {len(docs)}")

        except Exception as e:
            logger.error(f"MCP 搜索节点执行异常: {e}", exc_info=True)
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")

    # 5. 标记任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---node_web_search_mcp 处理结束---")

    # 若有有效搜索结果，返回结果供后续节点使用；无则返回空字典
    if docs:
        return {"web_search_docs": docs}
    return {}





def _run_async_from_sync(coro):
    """
    在同步函数里安全执行协程。

    这个节点属于同步 LangGraph 节点，但底层 MCP 调用是 async 的。
    如果当前线程已经有正在运行的事件循环，再直接 asyncio.run(coro)
    会抛出:
        RuntimeError: asyncio.run() cannot be called from a running event loop

    因此这里分两种情况：
    1. 当前线程没有运行中的事件循环：直接 asyncio.run(coro)；
    2. 当前线程已经有事件循环：把协程丢到独立线程里运行，让那个线程
       自己创建新的事件循环，再把结果同步取回。

    这样可以保证：
    - 同步节点依然保持“阻塞拿结果”的调用方式；
    - 不会和 FastAPI 当前线程里的事件循环冲突；
    - 不需要把整条查询图一次性重构成 async 版本。
    """
    # 执行步骤：
    # 1. 检查当前线程是否已有事件循环
    # 2. 没有则直接 asyncio.run()
    # 3. 若已有则丢到子线程中运行协程
    # 4. 解决同步 LangGraph 节点调用异步 MCP 的桥接问题
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

async def mcp_call(query):
    """
    异步调用百炼MCP搜索服务的核心函数。
    
    该函数负责初始化MCP客户端，建立SSE连接，调用远程工具，并返回原始结果。
    
    :param query: 搜索查询词（通常是经过改写后的精准Query）
    :return: MCP返回的原始结果对象 (包含 content, isError 等字段)
    """
    
    # ==================================================================================
    # 初始化百炼MCP SSE客户端
    # ----------------------------------------------------------------------------------
    # MCPServerSse 是一个基于 SSE (Server-Sent Events) 协议的 MCP 客户端实现。
    # 它的作用是连接到阿里云百炼提供的 MCP 服务端点，从而让我们可以像调用本地函数一样调用远程工具。
    #
    # 参数解释：
    # name: 客户端名称，用于日志标识，方便调试。
    # params: 连接配置字典
    #   - url: MCP 服务的 SSE 接口地址 (例如: .../mcps/WebSearch/sse)
    #   - headers: HTTP 请求头，必须包含 Authorization 字段传入 API Key 进行鉴权。
    #   - timeout: 连接建立和整体请求的超时时间。
    #   - sse_read_timeout: 读取 SSE 事件流的超时时间，防止流中断导致挂起。
    # ==================================================================================
    # 执行步骤：
    # 1. 创建 MCP 客户端
    # 2. 建立与百炼 WebSearch 的连接
    # 3. 调用 bailian_web_search 工具
    # 4. 返回原始结果对象，供上层节点解析
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 300,
            "sse_read_timeout": 300
        }
    )

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_config.mcp_base_url}")
        # 建立与MCP服务的SSE连接（异步方法，需await）
        await search_mcp.connect()
        
        logger.info(f"[MCP] 连接成功，正在调用工具 'bailian_web_search' 查询: {query}")
        # 调用百炼MCP的搜索工具（核心步骤）
        # tool_name: "bailian_web_search" 是百炼官方定义的工具名称
        # arguments: 工具所需的参数，这里需要 "query" (查询词) 和 "count" (返回数量)
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search", 
            arguments={"query": query, "count": 5}
        )
        logger.info("[MCP] 工具调用完成，已获取返回结果")
        return result
        
    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)
        return None
        
    finally:
        # 无论调用成功/失败，最终都关闭MCP连接（释放资源，异步方法）
        await search_mcp.cleanup()





if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "="*50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("="*50)
    
    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("="*50)
        
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/agent/nodes/node_rrf.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
