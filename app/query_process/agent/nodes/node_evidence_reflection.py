"""
证据反思节点 (Evidence Reflection Node)

本模块实现 Agentic RAG 查询流程中的「证据反思」环节。
在检索结果经过 RRF 融合 + Rerank 重排序之后，本节点负责：

1. 判断当前证据是否足够回答用户问题（sufficient / insufficient / conflicting）
2. 识别证据之间是否存在矛盾
3. 梳理还缺失哪些关键事实（missing_facts）
4. 从支持度（support）、覆盖度（coverage）、一致性（consistency）三个维度给出量化评分

反思结果会写入 state，供下游两个分支使用：
- 证据充分 → 进入 node_answer_output 直接生成答案
- 证据不足 → 进入 node_dynamic_reretrieval 触发补检索

当 LLM 调用失败或 JSON 解析异常时，自动降级到 _fallback_reflection 规则兜底。
"""

import json
import sys
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.debug_trace_utils import append_trace_event
from app.utils.task_utils import add_done_task, add_running_task



def _normalize_doc_dicts(docs, field_name: str) -> List[Dict]:
    """
    对文档列表做一次防御性清洗，确保后续逻辑读到的每一项都是 dict。

    Java 类比：
    - 相当于在真正处理前，先校验 List<?> 里的元素是不是都符合预期类型
    - 如果某一项类型不对，就记录日志并丢掉，避免后面继续访问时报错
    """
    normalized: List[Dict] = []
    for idx, doc in enumerate(docs or []):
        if isinstance(doc, dict):
            normalized.append(doc)
            continue

        preview = str(doc)
        if len(preview) > 160:
            preview = preview[:160] + "..."
        logger.warning(
            f"{field_name}[{idx}] type mismatch: expected dict, got {type(doc).__name__}, value={preview}"
        )
    return normalized


def _build_docs_text(reranked_docs: List[Dict], max_docs: int = 5, max_chars: int = 4000) -> str:
    """
    将重排序后的文档列表格式化为纯文本，供 Reflection 提示词引用。

    :param reranked_docs: Rerank 输出的文档列表，每项含 text/source/score/title/url/chunk_id
    :param max_docs: 最多引用的文档数量（默认 5 条）
    :param max_chars: 总字符上限（默认 4000），超出则截断
    :return: 格式化后的证据文本字符串；无文档时返回 "无证据"
    """
    # 执行步骤：
    # 1. 只取前几条 reranked_docs
    # 2. 每条文档都带上 source/score/title/chunk_id/url 等元信息
    # 3. 拼成给 Reflection LLM 阅读的证据文本
    reranked_docs = _normalize_doc_dicts(reranked_docs, "reranked_docs")
    if not reranked_docs:
        return "无证据"

    blocks = []
    used = 0

    for i, doc in enumerate(reranked_docs[:max_docs], start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        source = doc.get("source", "")
        score = doc.get("score")
        title = doc.get("title", "")
        url = doc.get("url", "")
        chunk_id = doc.get("chunk_id")

        block = [
            f"[证据{i}]",
            f"source={source}" if source else "",
            f"score={score:.4f}" if isinstance(score, (int, float)) else "",
            f"title={title}" if title else "",
            f"chunk_id={chunk_id}" if chunk_id else "",
            f"url={url}" if url else "",
            text[:800],
        ]

        block_text = "\n".join([x for x in block if x]).strip()
        if used + len(block_text) > max_chars:
            break

        blocks.append(block_text)
        used += len(block_text) + 2

    return "\n\n".join(blocks) if blocks else "无证据"


def _normalize_match_text(text: str) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _count_aligned_local_docs(state: QueryGraphState) -> int:
    item_names = state.get("item_names") or []
    if not item_names:
        return 0

    normalized_items = [_normalize_match_text(item) for item in item_names if _normalize_match_text(item)]
    if not normalized_items:
        return 0

    aligned = 0
    for doc in _normalize_doc_dicts(state.get("reranked_docs") or [], "state.reranked_docs"):
        if (doc.get("source") or "").lower() != "local":
            continue
        haystack = _normalize_match_text(f"{doc.get('title', '')} {doc.get('text', '')}")
        if not haystack:
            continue
        if any(item in haystack or haystack in item for item in normalized_items):
            aligned += 1
    return aligned


def _should_relax_insufficient(state: QueryGraphState, result: Dict) -> bool:
    if result.get("evidence_status") not in {"insufficient", "unknown"}:
        return False
    if state.get("need_clarify"):
        return False
    if state.get("task_type") not in {"kb_only", "kb_with_web"}:
        return False
    if state.get("intent_type") not in {"parameter_query", "operation_guide"}:
        return False
    if _count_aligned_local_docs(state) < 1:
        return False

    reranked_docs = _normalize_doc_dicts(state.get("reranked_docs") or [], "state.reranked_docs")
    if len(reranked_docs) >= 2:
        return True
    if int(state.get("retrieval_round", 1) or 1) >= 2:
        return True
    return False


def _relax_reflection_result(state: QueryGraphState, result: Dict) -> Dict:
    if not _should_relax_insufficient(state, result):
        return result

    relaxed = dict(result)
    relaxed["evidence_status"] = "sufficient"
    relaxed["reflection_reason"] = (
        (result.get("reflection_reason") or "").strip()
        + " 规则兜底：已命中同型号本地证据，且当前问题属于本地参数/操作问答，按可回答处理。"
    ).strip()
    relaxed["final_confidence"] = max(float(result.get("final_confidence") or 0.0), 0.58)
    relaxed["support_score"] = max(float(result.get("support_score") or 0.0), 0.62)
    relaxed["coverage_score"] = max(float(result.get("coverage_score") or 0.0), 0.55)
    relaxed["consistency_score"] = max(float(result.get("consistency_score") or 0.0), 0.7)
    if not isinstance(relaxed.get("missing_facts"), list):
        relaxed["missing_facts"] = []
    return relaxed


def _fallback_reflection(state: QueryGraphState) -> dict:
    """
    当 LLM 反思调用失败时的规则兜底策略。

    根据重排序文档数量给出保守判断：
    - >=2 条：sufficient，置信度 0.6
    - ==1 条：insufficient，置信度 0.35
    - ==0 条：insufficient，置信度 0.1

    :param state: 当前会话状态，用于读取 reranked_docs 数量
    :return: 包含 evidence_status/missing_facts/citations/各维度分数的字典
    """
    # 兜底策略：
    # 1. 没法调用 LLM 时，不让整条链路中断
    # 2. 按 reranked_docs 数量给一个保守判断
    # 3. 文档多则倾向 sufficient，文档极少则倾向 insufficient
    reranked_docs = _normalize_doc_dicts(state.get("reranked_docs") or [], "state.reranked_docs")

    if len(reranked_docs) >= 2:
        return {
            "evidence_status": "sufficient",
            "reflection_reason": "兜底策略：已有多条重排证据，暂时认为足够回答。",
            "missing_facts": [],
            "citations": [],
            "final_confidence": 0.6,
            "support_score": 0.7,
            "coverage_score": 0.6,
            "consistency_score": 0.8,
        }

    if len(reranked_docs) == 1:
        return {
            "evidence_status": "insufficient",
            "reflection_reason": "兜底策略：只有一条证据，可能不足以稳定回答。",
            "missing_facts": ["需要更多相关证据交叉验证"],
            "citations": [],
            "final_confidence": 0.35,
            "support_score": 0.4,
            "coverage_score": 0.25,
            "consistency_score": 0.7,
        }

    return {
        "evidence_status": "insufficient",
        "reflection_reason": "兜底策略：没有检索到有效证据。",
        "missing_facts": ["缺少任何可回答当前问题的证据"],
        "citations": [],
        "final_confidence": 0.1,
        "support_score": 0.0,
        "coverage_score": 0.0,
        "consistency_score": 0.0,
    }


def _parse_reflection_output(raw_text: str, state: QueryGraphState) -> dict:
    """
    解析 Reflection LLM 输出的 JSON，并做健壮性校验。

    处理内容包括：
    - 兼容 ```json ... ``` 代码块包裹
    - JSON 解析失败时降级到 _fallback_reflection
    - 补齐缺省字段（evidence_status / missing_facts / citations / 各维度分数）
    - 校验 evidence_status 取值合法性
    - 将分数字段强制转为 float

    :param raw_text: LLM 原始输出文本
    :param state: 当前会话状态（用于 fallback 时的兜底判断）
    :return: 标准化的反思结果字典
    """
    # 解析步骤：
    # 1. 清理 ```json 包裹
    # 2. 解析 JSON
    # 3. 补默认字段并校验类型
    # 4. 统一 evidence_status 和各项分数字段
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        result = json.loads(text)
    except Exception as exc:
        logger.error(f"Reflection JSON 解析失败: {exc}")
        return _fallback_reflection(state)

    result.setdefault("evidence_status", "unknown")
    result.setdefault("reflection_reason", "")
    result.setdefault("missing_facts", [])
    result.setdefault("citations", [])
    result.setdefault("final_confidence", 0.0)
    result.setdefault("support_score", 0.0)
    result.setdefault("coverage_score", 0.0)
    result.setdefault("consistency_score", 0.0)

    if result["evidence_status"] not in {"unknown", "sufficient", "insufficient", "conflicting"}:
        result["evidence_status"] = "unknown"

    if not isinstance(result["missing_facts"], list):
        result["missing_facts"] = []

    if not isinstance(result["citations"], list):
        result["citations"] = []
    else:
        # 校验每个元素必须是 dict，LLM 可能返回 ["字符串", ...] 等非字典元素
        normalized = []
        for idx, c in enumerate(result["citations"]):
            if isinstance(c, dict):
                normalized.append(c)
            else:
                logger.warning(
                    f"citations[{idx}] type mismatch: expected dict, got {type(c).__name__}, value={str(c)[:160]}"
                )
        result["citations"] = normalized

    for key in ["final_confidence", "support_score", "coverage_score", "consistency_score"]:
        try:
            result[key] = float(result[key])
        except Exception:
            result[key] = 0.0

    return result


def node_evidence_reflection(state: QueryGraphState) -> QueryGraphState:
    # 这一步可以理解成“证据审稿”：
    # 前面节点负责把候选材料找回来，
    # 当前节点负责判断这些材料是否足够支撑最终答案。
    """
    证据反思节点：判断当前检索到的证据是否足够回答用户问题。

    执行流程：
    1. 从 state 中提取重排序后的文档（reranked_docs）
    2. 将文档格式化为纯文本
    3. 构造反思提示词，调用 LLM 输出 JSON 格式的判断结果
    4. 解析 LLM 输出，写入 state 的各反思字段
    5. 如果 LLM 调用失败，降级到规则兜底策略

    写入 state 的字段：
    - evidence_status: sufficient / insufficient / conflicting / unknown
    - reflection_reason: 判断原因
    - missing_facts: 缺失的事实列表
    - citations: 引用列表
    - final_confidence / support_score / coverage_score / consistency_score

    :param state: 查询图共享状态
    :return: 更新后的状态
    """
    logger.info("--- node_evidence_reflection 开始执行 ---")
    # 节点职责：
    # 1. 不直接回答问题，而是审查当前证据够不够回答
    # 2. 输出 evidence_status / missing_facts / confidence 等治理字段
    # 3. 决定主图下一步是直接回答，还是再补检索一轮
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    question = state.get("rewritten_query") or state.get("original_query") or ""
    reranked_docs = state.get("reranked_docs") or []
    docs_text = _build_docs_text(reranked_docs)

    reflection_prompt = f"""
你是 Agentic RAG 系统中的 Evidence Reflection 节点。

你的任务不是直接回答用户问题，而是判断：
1. 当前证据是否足够回答问题
2. 证据之间是否冲突
3. 还缺少哪些关键事实
4. 从支持度、覆盖度、一致性三个维度给出评分

请严格输出 JSON：
{{
  "evidence_status": "sufficient | insufficient | conflicting | unknown",
  "reflection_reason": "一句话说明判断原因",
  "missing_facts": ["缺失信息1", "缺失信息2"],
  "citations": [],
  "final_confidence": 0.0,
  "support_score": 0.0,
  "coverage_score": 0.0,
  "consistency_score": 0.0
}}

规则：
1. support_score 表示证据是否直接支持答案
2. coverage_score 表示证据是否覆盖了问题关键点
3. consistency_score 表示证据是否相互一致
4. 取值范围建议都为 0 到 1
5. 只输出 JSON

用户问题：
{question}

当前证据：
{docs_text}
""".strip()

    try:
        llm = get_llm_client(json_mode=True)
        messages = [
            SystemMessage(content="你是一个严谨的 Evidence Reflection 节点，只输出 JSON。"),
            HumanMessage(content=reflection_prompt),
        ]

        logger.info("Reflection 正在调用 LLM 判断证据充分性...")
        response = llm.invoke(messages)
        raw_text = response.content

        logger.debug(f"Reflection 原始输出: {raw_text}")
        result = _parse_reflection_output(raw_text, state)

    except Exception as exc:
        logger.error(f"Reflection 调用失败，进入 fallback: {exc}", exc_info=True)
        result = _fallback_reflection(state)

    result = _relax_reflection_result(state, result)

    state["evidence_status"] = result.get("evidence_status", "unknown")
    state["reflection_reason"] = result.get("reflection_reason", "")
    state["missing_facts"] = result.get("missing_facts", [])
    state["citations"] = result.get("citations", [])
    state["final_confidence"] = result.get("final_confidence", 0.0)
    state["support_score"] = result.get("support_score", 0.0)
    state["coverage_score"] = result.get("coverage_score", 0.0)
    state["consistency_score"] = result.get("consistency_score", 0.0)

    append_trace_event(
        session_id=state["session_id"],
        node="node_evidence_reflection",
        retrieval_round=int(state.get("retrieval_round", 1)),
        payload={
            "evidence_status": state["evidence_status"],
            "reflection_reason": state["reflection_reason"],
            "missing_facts": state["missing_facts"],
            "final_confidence": state["final_confidence"],
            "support_score": state["support_score"],
            "coverage_score": state["coverage_score"],
            "consistency_score": state["consistency_score"],
        },
    )

    logger.info(
        f"Reflection 执行完成: "
        f"evidence_status={state['evidence_status']}, "
        f"final_confidence={state['final_confidence']}, "
        f"support_score={state['support_score']}, "
        f"coverage_score={state['coverage_score']}, "
        f"consistency_score={state['consistency_score']}, "
        f"missing_facts={state['missing_facts']}"
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/agent/nodes/node_dynamic_retrieval.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
