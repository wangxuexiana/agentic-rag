"""
Dynamic re-retrieval node.
"""

import json
import sys
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.query_process.agent.tool_registry import get_retry_upgrade_tools
from app.utils.debug_trace_utils import append_trace_event
from app.utils.task_utils import add_done_task, add_running_task


# ==================== Java 开发者阅读提示 ====================
# 这个节点负责“第二轮补检索前的准备”。
# 前面 Reflection 已经判断出证据不足，这里要做两件事：
# 1. 生成下一轮更聚焦的 followup_query
# 2. 升级 selected_tools，决定下一轮是否补 Web / KG 等工具
#
# 然后它会把 state 重置到“准备再次检索”的状态，再交回主图继续流转。
# ===========================================================


def _build_followup_query_rule(state: QueryGraphState) -> str:
    # 规则兜底步骤：
    # 1. 读取上一轮 query
    # 2. 读取 Reflection 输出的 missing_facts
    # 3. 尽量把“还缺什么”显式拼回 followup_query
    base_query = state.get("rewritten_query") or state.get("original_query") or ""
    missing_facts: List[str] = state.get("missing_facts") or []
    item_names: List[str] = state.get("item_names") or []

    prefix = ""
    if item_names:
        prefix = "、".join(str(x).strip() for x in item_names if str(x).strip())

    if not missing_facts:
        return f"{prefix} {base_query}".strip()

    facts_text = "；".join(str(x).strip() for x in missing_facts if str(x).strip())
    if not facts_text:
        return f"{prefix} {base_query}".strip()

    query = f"{base_query}。补充检索重点：{facts_text}"
    if prefix and prefix not in query:
        query = f"{prefix} {query}"
    return query.strip()


def _rewrite_followup_query_with_llm(state: QueryGraphState) -> tuple[str, str]:
    # 执行步骤：
    # 1. 把上一轮 query、反思原因、缺失事实、商品名交给 LLM
    # 2. 让 LLM 生成更聚焦的 followup_query
    # 3. 同时返回 retry_intent，描述这一轮补检索重点
    base_query = state.get("rewritten_query") or state.get("original_query") or ""
    reflection_reason = state.get("reflection_reason", "")
    missing_facts = state.get("missing_facts") or []
    item_names = state.get("item_names") or []
    retrieval_round = int(state.get("retrieval_round", 1))

    prompt = f"""
你是 Agentic RAG 的 retry query rewrite 节点。
你的任务是基于上一轮检索后“仍然缺失的信息”，重写出一条更适合下一轮检索的 followup query。
请严格输出 JSON：
{{
  "followup_query": "新的检索问题",
  "retry_intent": "一句话说明这一轮补检索重点"
}}

要求：
1. followup_query 必须围绕原问题，不要发散
2. followup_query 要突出 missing_facts
3. 不要原样重复上一轮 query
4. 如果已经确认了具体产品/型号，followup_query 里必须保留该名称，不要泛化成通用词
4. 只输出 JSON

当前轮次：{retrieval_round}

原问题：
{state.get("original_query", "")}

上一轮 query：{base_query}

反思原因：
{reflection_reason}

缺失事实：{json.dumps(missing_facts, ensure_ascii=False)}
已确认产品：{json.dumps(item_names, ensure_ascii=False)}
""".strip()

    llm = get_llm_client(json_mode=True)
    messages = [
        SystemMessage(content="你是一个严谨的 retry query rewrite 节点，只输出 JSON。"),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw_text = (response.content or "").strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text.replace("```json", "", 1).strip()
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

    result = json.loads(raw_text)
    followup_query = (result.get("followup_query") or "").strip()
    retry_intent = (result.get("retry_intent") or "").strip()

    if not followup_query:
        raise ValueError("followup_query is empty")

    return followup_query, retry_intent


def node_dynamic_reretrieval(state: QueryGraphState) -> QueryGraphState:
    """
    动态补检索主节点。

    可以把它理解成：
    - 第一轮没答稳
    - 先总结“还缺什么”
    - 再带着更聚焦的问题，重新发起下一轮检索
    """
    # 节点职责：
    # 1. 根据 Reflection 结论准备第二轮检索
    # 2. 重写 followup_query
    # 3. 升级工具集
    # 4. 清空上一轮检索和评分结果，回到 tool_router 重新开始
    logger.info("--- node_dynamic_reretrieval 开始执行 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    current_round = int(state.get("retrieval_round", 1))
    next_round = current_round + 1

    retry_intent = ""
    try:
        followup_query, retry_intent = _rewrite_followup_query_with_llm(state)
        logger.info(f"LLM followup_query rewrite 成功: {followup_query}")
    except Exception as exc:
        logger.error(f"LLM followup_query rewrite 失败，回退规则重写: {exc}", exc_info=True)
        followup_query = _build_followup_query_rule(state)
        retry_intent = "fallback_rule_rewrite"

    selected_tools = get_retry_upgrade_tools(
        current_tools=state.get("selected_tools") or [],
        missing_facts=state.get("missing_facts") or [],
    )

    state["retrieval_round"] = next_round
    state["followup_query"] = followup_query
    state["rewritten_query"] = followup_query
    state["selected_tools"] = selected_tools
    state["retry_intent"] = retry_intent

    state["embedding_chunks"] = []
    state["hyde_embedding_chunks"] = []
    state["kg_chunks"] = []
    state["web_search_docs"] = []
    state["rrf_chunks"] = []
    state["reranked_docs"] = []
    state["evidence_status"] = "unknown"
    state["reflection_reason"] = ""
    state["citations"] = []
    state["final_confidence"] = 0.0
    state["support_score"] = 0.0
    state["coverage_score"] = 0.0
    state["consistency_score"] = 0.0

    append_trace_event(
        session_id=state["session_id"],
        node="node_dynamic_reretrieval",
        retrieval_round=state["retrieval_round"],
        payload={
            "followup_query": state["followup_query"],
            "selected_tools": state["selected_tools"],
            "retry_intent": state.get("retry_intent", ""),
            "missing_facts": state.get("missing_facts", []),
        },
    )

    logger.info(
        f"Dynamic Re-retrieval 准备完成: "
        f"retrieval_round={state['retrieval_round']}, "
        f"selected_tools={state['selected_tools']}, "
        f"retry_intent={state.get('retry_intent')}, "
        f"followup_query={state['followup_query']}"
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
