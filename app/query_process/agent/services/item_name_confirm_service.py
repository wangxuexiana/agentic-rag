"""
实体名确认服务层。

职责边界：
1. 负责实体名的抽取、标准名检索、候选对齐、确认策略判断。
2. 不负责 LangGraph 节点路由，只提供可复用的业务方法。
3. 不负责节点级别的历史读写编排，节点层只是在需要时调用这里的方法。

阅读顺序建议：
1. 先看 `extract_query_info()`，理解实体抽取和查询改写。
2. 再看 `vectorize_and_query_item_names()`，理解标准实体候选如何召回。
3. 再看 `align_item_names()`，理解候选如何分成“确认”和“待澄清”。
4. 最后看 `apply_confirmation_result()`，理解策略结果如何写回 state。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.clients.mongo_history_utils import update_message_item_names
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.llm_utils import get_llm_client


# 一些明显无效的实体名结果，抽取或对齐阶段需要过滤掉。
INVALID_ITEM_NAME_VALUES = {
    "",
    "空字符串",
    "未知商品",
    "未知产品",
    "未知设备",
    "null",
    "none",
    "unknown",
    "n/a",
}


def fallback_extract_item_names(query: str) -> List[str]:
    """
    规则兜底：从原始 query 中直接抽取可能的实体名。

    执行步骤：
    1. 如果 query 为空，直接返回空列表。
    2. 按多组规则匹配常见的“型号 / 中文名 + 型号”模式。
    3. 对匹配结果去空格、去重、保序。
    4. 最多返回前 3 个候选，避免兜底结果过多污染后续检索。
    """
    if not query:
        return []

    patterns = [
        r"[A-Za-z]{2,}\s?-?\d{2,}",
        r"[A-Za-z]+\d{2,}",
        r"[\u4e00-\u9fff]{2,}\s?[A-Za-z]\d{0,4}[A-Za-z]{0,4}",
        r"[\u4e00-\u9fff]{2,}[A-Za-z]",
    ]

    results: List[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, query)
        for item in matches:
            normalized = re.sub(r"\s+", " ", item).strip()
            if normalized and normalized not in results:
                results.append(normalized)
    return results[:3]


def normalize_item_name(text: str) -> str:
    """
    规范化实体名，供比较和去重使用。

    执行步骤：
    1. 去首尾空格。
    2. 去掉中间空格、横杠、下划线的差异。
    3. 统一转成小写，减少大小写影响。
    """
    text = (text or "").strip()
    if not text:
        return ""
    return re.sub(r"[\s\-_]+", "", text).lower()


def is_valid_item_name(text: str) -> bool:
    """
    判断一个实体名是否可用于后续检索与确认。

    执行步骤：
    1. 过滤空值和明显无效值。
    2. 过滤过短、过长的异常结果。
    3. 至少要求包含中文、字母或数字中的一种有效字符。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.lower() in INVALID_ITEM_NAME_VALUES or stripped in INVALID_ITEM_NAME_VALUES:
        return False
    if len(stripped) < 2 or len(stripped) > 80:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", stripped):
        return False
    return True


def is_candidate_aligned(extracted_name: str, candidate_name: str, score: float) -> bool:
    """
    判断向量检索得到的候选标准名，是否和抽取名对得上。

    执行步骤：
    1. 先做规范化比较，完全一致直接通过。
    2. 再看包含关系，处理“中文名 + 型号”“型号简写”等情况。
    3. 再提取字母数字 token，做型号级别的对齐。
    4. 最后做字符集合重叠度判断。
    5. 如果前面都不够强，则用高分阈值做兜底放行。
    """
    extracted_norm = normalize_item_name(extracted_name)
    candidate_norm = normalize_item_name(candidate_name)

    if not candidate_norm:
        return False
    if not extracted_norm:
        return score >= 0.92
    if extracted_norm == candidate_norm:
        return True
    if extracted_norm in candidate_norm or candidate_norm in extracted_norm:
        return True

    extracted_tokens = [token for token in re.findall(r"[a-z0-9]+", extracted_norm) if len(token) >= 4]
    candidate_model_text = "".join(re.findall(r"[a-z0-9]+", candidate_norm))
    if extracted_tokens:
        if not candidate_model_text:
            return False
        if any(token in candidate_model_text for token in extracted_tokens):
            return True
        if score < 0.9:
            return False

    extracted_chars = set(extracted_norm)
    candidate_chars = set(candidate_norm)
    if not extracted_chars or not candidate_chars:
        return False

    overlap_ratio = len(extracted_chars & candidate_chars) / min(len(extracted_chars), len(candidate_chars))
    if overlap_ratio >= 0.6:
        return True

    return score >= 0.92


def dedupe_item_names(item_names: List[str]) -> List[str]:
    """
    对实体名做去重并保留原顺序。

    执行步骤：
    1. 逐个实体名做规范化。
    2. 已见过的规范化结果跳过。
    3. 未见过的保留原始文本形式写入结果。
    """
    results: List[str] = []
    seen = set()
    for item_name in item_names:
        normalized = normalize_item_name(item_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(item_name)
    return results


def extract_query_info(query: str, history: List[Dict[str, Any]], logger) -> Dict[str, Any]:
    """
    第 1 段：从用户问题里抽取实体名，并同时生成重写后的查询。

    执行步骤：
    1. 把最近历史消息拼成 `history_text`，作为抽取上下文。
    2. 加载 `rewritten_query_and_itemnames` 提示词模板。
    3. 调用支持 JSON 输出的 LLM，要求返回：
       - item_names
       - rewritten_query
    4. 解析 JSON 并补默认值。
    5. 如果 LLM 没抽到实体名，则走 `fallback_extract_item_names()` 规则兜底。
    6. 如果 rewritten_query 为空，则回退成原 query。

    返回格式：
    - {
        "item_names": [...],
        "rewritten_query": "...",
      }
    """
    client = get_llm_client(json_mode=True)
    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"

    try:
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
    except Exception as exc:
        logger.error(f"Step 3: 加载提示词失败: {exc}")
        return {"item_names": [], "rewritten_query": query}

    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt),
    ]

    try:
        response = client.invoke(messages)
        content = response.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        result.setdefault("item_names", [])
        result.setdefault("rewritten_query", query)

        if not result["item_names"]:
            fallback_item_names = fallback_extract_item_names(query)
            if fallback_item_names:
                logger.info(f"Step 3: LLM 未提取到实体名，启用规则兜底: {fallback_item_names}")
                result["item_names"] = fallback_item_names

        rewritten_query = (result.get("rewritten_query") or "").strip()
        if not rewritten_query:
            result["rewritten_query"] = query
        return result
    except Exception as exc:
        logger.error(f"Step 3: 实体抽取或解析失败: {exc}")
        return {"item_names": [], "rewritten_query": query}


def vectorize_and_query_item_names(item_names: List[str], logger) -> List[Dict[str, Any]]:
    """
    第 2 段：把抽取到的实体名映射到标准实体候选。

    执行步骤：
    1. 连接 Milvus，并读取 `ITEM_NAME_COLLECTION`。
    2. 对每个 item_name 生成 dense / sparse embedding。
    3. 构造混合检索请求。
    4. 在实体名集合里召回 topK 候选标准名。
    5. 对每个候选保留：
       - item_name
       - score
    6. 返回“抽取名 -> 候选列表”的结构。

    返回格式：
    - [
        {
          "extracted_name": "...",
          "matches": [
            {"item_name": "...", "score": 0.91},
            ...
          ]
        }
      ]
    """
    results: List[Dict[str, Any]] = []

    client = get_milvus_client()
    if not client:
        logger.error("Step 4: 无法连接到 Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: 未找到 ITEM_NAME_COLLECTION")
        return results

    try:
        embeddings = generate_embeddings(item_names)
        for i, name in enumerate(item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5,
                )
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"],
                )

                matches: List[Dict[str, Any]] = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        item_name = entity.get("item_name")
                        score = hit.get("distance")
                        if is_valid_item_name(item_name):
                            matches.append({"item_name": item_name, "score": score})

                results.append({"extracted_name": name, "matches": matches})
            except Exception as exc:
                logger.error(f"Step 4: 处理实体 '{name}' 时出错: {exc}")
                results.append({"extracted_name": name, "matches": []})
    except Exception as exc:
        logger.error(f"Step 4: 向量化或检索失败: {exc}")

    return results


def align_item_names(query_results: List[Dict[str, Any]], logger) -> Dict[str, List[str]]:
    """
    第 3 段：把候选标准名整理成“可直接确认”和“需要澄清”两类。

    执行步骤：
    1. 遍历每个 extracted_name 的候选列表，并按分数降序。
    2. 用 `is_candidate_aligned()` 过滤明显不对齐的候选。
    3. 把候选按高分组和中分组分层：
       - 高分：score > 0.85
       - 中分：score >= 0.6
    4. 规则判断：
       - 单个高分：直接确认
       - 多个高分：优先精确同名，否则取最高分
       - 单个中分：直接确认
       - 多个中分：进入 options，等待用户澄清
    5. 最后对 confirmed 和 options 去重保序。

    返回格式：
    - {
        "confirmed_item_names": [...],
        "options": [...],
      }
    """
    confirmed_item_names: List[str] = []
    options: List[str] = []

    for res in query_results:
        extracted_name = (res.get("extracted_name") or "").strip()
        matches = res.get("matches", []) or []
        if not matches:
            continue

        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        aligned_matches = [
            m
            for m in matches
            if is_candidate_aligned(extracted_name, m.get("item_name", ""), float(m.get("score", 0) or 0))
        ]
        high = [m for m in aligned_matches if m.get("score", 0) > 0.85]
        mid = [m for m in aligned_matches if m.get("score", 0) >= 0.6]

        if len(high) == 1:
            confirmed_item_names.append(high[0].get("item_name"))
            continue

        if len(high) > 1:
            picked = None
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        break
            if not picked:
                picked = high[0]
            confirmed_item_names.append(picked.get("item_name"))
            continue

        if len(mid) == 1:
            confirmed_item_names.append(mid[0].get("item_name"))
            continue

        if len(mid) > 1:
            current_options = [m.get("item_name") for m in mid[:5] if is_valid_item_name(m.get("item_name"))]
            options.extend(current_options)

    result = {
        "confirmed_item_names": dedupe_item_names(confirmed_item_names),
        "options": dedupe_item_names(options),
    }
    logger.info(f"Step 5: 实体对齐结果: {result}")
    return result


def apply_confirmation_result(
    state: Dict[str, Any],
    align_result: Dict[str, Any],
    history: List[Dict[str, Any]],
    rewritten_query: str,
    raw_item_names: List[str],
    logger,
) -> Dict[str, Any]:
    """
    第 4 段：根据对齐结果，决定当前查询应该如何继续。

    执行步骤：
    1. 先读取 planner 的判断：
       - task_type
       - need_clarify
       - clarification_question
    2. 如果 planner 已要求澄清，则优先输出澄清问句。
    3. 如果有 confirmed_item_names：
       - 回写 state["item_names"]
       - 更新历史消息里的 item_names
       - 比较类问题保留必要的原始对象
    4. 如果只有 options：
       - 组装澄清问题写入 state["answer"]
       - 清空 state["item_names"]
    5. 如果 confirmed 和 options 都没有：
       - 允许继续后续检索
       - 如果 raw_item_names 可用，则作为暂定检索对象保留
       - 同时清掉旧 answer，避免图提前短路

    这是服务层里最接近“策略决策”的函数。
    它不做节点路由，只负责把确认策略写回共享 state。
    """
    align_result = align_result or {}
    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    task_type = state.get("task_type", "full_agentic")
    need_clarify = bool(state.get("need_clarify", False))
    clarification_question = state.get("clarification_question", "")

    if need_clarify or task_type == "clarification":
        state["answer"] = clarification_question or "请补充更具体的产品型号、场景或问题细节。"
        state["item_names"] = []
        return state

    if confirmed:
        ids_to_update: List[str] = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))
        if ids_to_update:
            update_message_item_names(ids_to_update, confirmed)

        final_item_names = list(confirmed)
        if state.get("intent_type") == "comparison":
            tentative_item_names = dedupe_item_names(
                [name for name in (raw_item_names or []) if is_valid_item_name(name)]
            )
            if tentative_item_names:
                final_item_names = dedupe_item_names(final_item_names + tentative_item_names)

        state["item_names"] = final_item_names
        state["rewritten_query"] = rewritten_query
        state.pop("answer", None)
        return state

    if options:
        options_str = "、".join(options[:3])
        state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
        state["item_names"] = []
        return state

    tentative_item_names = dedupe_item_names([name for name in (raw_item_names or []) if is_valid_item_name(name)])
    state["item_names"] = tentative_item_names if tentative_item_names else []
    state["rewritten_query"] = rewritten_query
    state.pop("answer", None)
    return state
