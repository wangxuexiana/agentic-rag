"""
答案输出服务层。

职责边界：
1. 承载答案生成前后的业务逻辑，包括拒答判断、prompt 构造、答案后处理。
2. 不负责 LangGraph 节点路由，节点层只调用这里的方法。
3. 不负责最终历史落库编排，节点层负责收口和持久化。

阅读顺序建议：
1. 先看 `check_existing_answer()` 和 `build_insufficient_evidence_answer()`。
2. 再看 `construct_answer_prompt()` 和 `generate_response()`。
3. 最后看 `append_answer_meta()`、`extract_images_from_docs()`、`normalize_answer_images()`。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.core.load_prompt import load_prompt
from app.lm.llm_utils import get_llm_client
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import set_task_result


# 图片块标记：答案最终会把图片 URL 统一放在这个标记下面。
IMAGE_BLOCK_MARKER = "【图片】"

# 控制 prompt 中上下文拼接的最大字符数，避免参考内容无限膨胀。
MAX_CONTEXT_CHARS = 12000


def normalize_doc_dicts(docs, field_name: str, logger):
    """
    统一清洗文档列表，确保后续逻辑面对的是 dict 列表。

    执行步骤：
    1. 遍历 docs。
    2. 如果元素是 dict，保留。
    3. 如果元素不是 dict，记录 warning 并丢弃。
    4. 返回清洗后的标准列表。
    """
    normalized = []
    for idx, doc in enumerate(docs or []):
        if isinstance(doc, dict):
            normalized.append(doc)
            continue
        preview = str(doc)
        if len(preview) > 160:
            preview = preview[:160] + "..."
        logger.warning(f"{field_name}[{idx}] type mismatch: expected dict, got {type(doc).__name__}, value={preview}")
    return normalized


def format_evidence_status_label(evidence_status: str) -> str:
    """
    把内部 evidence_status 状态码转成中文展示文本。

    主要是给最终答案里的“参考说明”部分使用。
    """
    mapping = {
        "sufficient": "证据较充分",
        "insufficient": "证据不足",
        "conflicting": "证据存在冲突",
        "unknown": "证据状态未知",
    }
    return mapping.get(evidence_status or "unknown", "证据状态未知")


def collect_display_citations(state, logger, max_items: int = 3) -> list:
    """
    组装给最终答案展示的简化引用列表。

    执行步骤：
    1. 优先读取 reflection 阶段已经产出的 `citations`。
    2. 如果 citations 为空，则回退到 `reranked_docs` 的前几条文档。
    3. 统一输出 title / source / chunk_id / url 四类字段。
    """
    citations = state.get("citations") or []
    if citations:
        safe = [c for c in citations if isinstance(c, dict)]
        return safe[:max_items]

    reranked_docs = normalize_doc_dicts(state.get("reranked_docs") or [], "state.reranked_docs", logger)
    out = []
    for doc in reranked_docs[:max_items]:
        out.append(
            {
                "title": doc.get("title") or "",
                "source": doc.get("source") or "",
                "chunk_id": doc.get("chunk_id") or "",
                "url": doc.get("url") or "",
            }
        )
    return out


def normalize_match_text(text: str) -> str:
    """
    把文本规范化成便于匹配的形式。

    执行步骤：
    1. 转小写。
    2. 只保留中英文和数字。
    3. 用于实体名和文档正文之间的弱匹配。
    """
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def count_aligned_local_docs(state, logger) -> int:
    """
    统计当前重排结果里，有多少条本地文档和实体名对齐。

    执行步骤：
    1. 读取 `item_names`，并做规范化。
    2. 遍历 `reranked_docs`，只统计 source=local 的文档。
    3. 用标题 + 正文构造 haystack，再和实体名做包含匹配。
    4. 返回对齐的本地文档数量。
    """
    item_names = state.get("item_names") or []
    if not item_names:
        return 0

    normalized_items = [normalize_match_text(item) for item in item_names if normalize_match_text(item)]
    if not normalized_items:
        return 0

    aligned = 0
    for doc in normalize_doc_dicts(state.get("reranked_docs") or [], "state.reranked_docs", logger):
        if (doc.get("source") or "").lower() != "local":
            continue
        haystack = normalize_match_text(f"{doc.get('title', '')} {doc.get('text', '')}")
        if not haystack:
            continue
        if any(item in haystack or haystack in item for item in normalized_items):
            aligned += 1
    return aligned


def should_force_abstain(state, logger) -> bool:
    """
    判断当前是否应该强制拒答。

    执行步骤：
    1. 先看 evidence_status 是否属于 `insufficient/conflicting`。
    2. 再看 final_confidence 是否过低。
    3. 对本地知识库里的参数/操作类问题做适度放宽：
       - 如果已有至少 1 条对齐的本地文档
       - 且当前并不需要澄清
       则允许继续回答。
    """
    evidence_status = state.get("evidence_status", "unknown")
    final_confidence = float(state.get("final_confidence") or 0.0)
    if evidence_status not in {"insufficient", "conflicting"}:
        return False
    if final_confidence >= 0.35:
        return False
    if state.get("task_type") in {"kb_only", "kb_with_web"} and state.get("intent_type") in {"parameter_query", "operation_guide"}:
        if not state.get("need_clarify") and count_aligned_local_docs(state, logger) >= 1:
            return False
    return True


def build_insufficient_evidence_answer(state, logger) -> str:
    """
    当证据不足且命中拒答策略时，构造保守答案。

    执行步骤：
    1. 判断 evidence_status 是否属于不足/冲突。
    2. 判断是否真的需要强制拒答。
    3. 读取 item_names / reflection_reason / missing_facts。
    4. 组装一段保守回答，明确说明：
       - 当前不能确认
       - 原因是什么
       - 缺什么信息
       - 下一步建议补什么
    """
    evidence_status = state.get("evidence_status", "unknown")
    if evidence_status not in {"insufficient", "conflicting"}:
        return ""
    if not should_force_abstain(state, logger):
        return ""

    item_names = state.get("item_names") or []
    subject = "、".join(str(x).strip() for x in item_names if str(x).strip()) or "当前问题"
    reflection_reason = (state.get("reflection_reason") or "").strip()
    missing_facts = [str(x).strip() for x in (state.get("missing_facts") or []) if str(x).strip()]

    lines = [f"根据当前检索到的证据，我暂时不能确认“{subject}”的结论。"]
    if reflection_reason:
        lines.append(f"原因：{reflection_reason}")
    if missing_facts:
        lines.append("仍缺少的信息：")
        for idx, fact in enumerate(missing_facts[:3], start=1):
            lines.append(f"{idx}. {fact}")
    lines.append("建议补充更准确的产品型号、官方规格页或说明书原文后再确认。")
    return "\n".join(lines).strip()


def check_existing_answer(state, logger) -> bool:
    """
    检查上游节点是否已经把答案写进了 state。

    执行步骤：
    1. 读取 `state["answer"]`。
    2. 如果没有答案，返回 False，交给后续正常生成。
    3. 如果已有答案：
       - 流式模式：直接推 SSE DELTA
       - 非流式模式：直接写 task result
    4. 返回 True，表示当前可直接复用现有答案。
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream")
    if not answer:
        return False
    if is_stream:
        logger.info("Step 1: 上游已提供 answer，直接复用并推流")
        push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    else:
        set_task_result(state["session_id"], "answer", answer)
    return True


def construct_answer_prompt(state, logger) -> str:
    """
    构造最终答案生成 prompt。

    执行步骤：
    1. 读取问题、历史、实体名、重排后的文档。
    2. 优先用 rewritten_query 作为问题，没有则退回 original_query。
    3. 把 reranked_docs 组装成“元信息头 + 正文”的上下文块。
    4. 控制参考文档总长度不超过 `MAX_CONTEXT_CHARS`。
    5. 再按剩余预算拼历史消息。
    6. 格式化 item_names。
    7. 调用 `load_prompt("answer_out", ...)` 生成最终 prompt。
    """
    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "")
    question = rewritten_query if rewritten_query else original_query
    history = state.get("history", [])
    item_names = state.get("item_names", [])
    reranked_docs = state.get("reranked_docs") or []

    docs: List[str] = []
    used = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        source = doc.get("source") or ""
        chunk_id = doc.get("chunk_id")
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        score = doc.get("score")

        meta_parts = [f"[{i}]"]
        if source:
            meta_parts.append(f"[{source}]")
        if chunk_id:
            meta_parts.append(f"[chunk_id={chunk_id}]")
        if url:
            meta_parts.append(f"[url={url}]")
        if score is not None:
            meta_parts.append(f"[score={float(score):.4f}]")
        if title:
            meta_parts.append(f"[title={title}]")

        block = " ".join(meta_parts) + "\n" + text
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        docs.append(block)
        used += len(block) + 2
    context_str = "\n\n".join(docs) if docs else "无参考内容"

    history_str = ""
    rebuilt_used = len(context_str) + 2 if context_str else 0
    for msg in history:
        role = msg.get("role")
        text = msg.get("text")
        if role == "user" and text:
            fragment = f"用户: {text}\n"
        elif role == "assistant" and text:
            fragment = f"助手: {text}\n"
        else:
            fragment = ""
        if not fragment:
            continue
        if rebuilt_used + len(fragment) + 2 > MAX_CONTEXT_CHARS:
            break
        history_str += fragment
        rebuilt_used += len(fragment) + 2
    if not history_str:
        history_str = "无历史对话"

    item_names_str = ", ".join(item_names) if item_names else "无指定商品"
    prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )
    logger.info(f"组装后的提示词为：{prompt}")
    return prompt


def generate_response(state, prompt: str, logger) -> Dict[str, Any]:
    """
    调用 LLM 生成答案，兼容流式和非流式两种模式。

    执行步骤：
    1. 获取统一的 LLM 客户端。
    2. 读取 `session_id / is_stream`。
    3. 如果是流式：
       - 循环读取 chunk
       - 把增量内容推给前端
       - 同时累积成最终 answer
    4. 如果是非流式：
       - 直接 invoke(prompt)
       - 把完整答案写入 state 和 task result
    5. 若生成失败，写入兜底错误答案。
    """
    llm = get_llm_client()
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")

    if is_stream:
        final_text = ""
        try:
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    final_text += delta
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
        except Exception as exc:
            logger.error(f"流式生成出错: {exc}", exc_info=True)
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(exc)})
        state["answer"] = final_text
        return state

    try:
        response = llm.invoke(prompt)
        content = response.content
        state["answer"] = content
        set_task_result(session_id, "answer", content)
    except Exception as exc:
        logger.error(f"生成回答出错: {exc}", exc_info=True)
        state["answer"] = "抱歉，生成回答时出现错误。"
    return state


def append_answer_meta(state, logger) -> str:
    """
    在最终答案末尾追加“参考说明”部分。

    执行步骤：
    1. 先检查 answer 是否为空。
    2. 读取 evidence_status、confidence、citations 等信息。
    3. 如果没有任何证据，也没有证据状态，就不追加。
    4. 如果答案里已经有“参考说明”，避免重复追加。
    5. 组装：
       - 证据判断
       - 置信度 / 支持度 / 覆盖度 / 一致性
       - 判断依据
       - 参考来源
    6. 返回拼接后的新答案文本。
    """
    answer = (state.get("answer") or "").strip()
    if not answer:
        return answer

    reranked_docs = normalize_doc_dicts(state.get("reranked_docs") or [], "state.reranked_docs", logger)
    evidence_status = state.get("evidence_status", "unknown")
    reflection_reason = (state.get("reflection_reason") or "").strip()
    final_confidence = state.get("final_confidence", 0.0)
    support_score = state.get("support_score", 0.0)
    coverage_score = state.get("coverage_score", 0.0)
    consistency_score = state.get("consistency_score", 0.0)

    if not reranked_docs and evidence_status == "unknown":
        return answer
    if "参考说明：" in answer:
        return answer

    citations = collect_display_citations(state, logger)
    status_label = format_evidence_status_label(evidence_status)
    confidence_percent = max(0, min(100, round(float(final_confidence) * 100)))

    lines = [
        "",
        "参考说明：",
        f"证据判断：{status_label}",
        f"置信度：{confidence_percent}%",
        f"支持度：{max(0, min(100, round(float(support_score) * 100)))}%",
        f"覆盖度：{max(0, min(100, round(float(coverage_score) * 100)))}%",
        f"一致性：{max(0, min(100, round(float(consistency_score) * 100)))}%",
    ]
    if reflection_reason:
        lines.append(f"判断依据：{reflection_reason}")
    if citations:
        lines.append("参考来源：")
        for idx, item in enumerate(citations, start=1):
            title = item.get("title") or "未命名来源"
            source = item.get("source") or ""
            chunk_id = item.get("chunk_id") or ""
            url = item.get("url") or ""
            piece = f"{idx}. {title}"
            if source:
                piece += f" [{source}]"
            if chunk_id:
                piece += f" (chunk_id={chunk_id})"
            if url:
                piece += f" {url}"
            lines.append(piece)
    return answer + "\n" + "\n".join(lines)


def extract_images_from_docs(docs, logger):
    """
    从文档列表里提取图片 URL。

    执行步骤：
    1. 先把 docs 规范化成 dict 列表。
    2. 优先检查文档自己的 `url` 字段，看是否就是图片链接。
    3. 再从文档正文里匹配 Markdown 图片语法：
       `![alt](url)`
    4. 对所有结果去重并保序。
    """
    images = []
    seen = set()
    docs = normalize_doc_dicts(docs, "image_docs", logger)
    if not docs:
        return []

    md_img_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    for doc in docs:
        url = (doc.get("url") or "").strip()
        if url and url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")) and url not in seen:
            seen.add(url)
            images.append(url)

        text = (doc.get("text") or "").strip()
        if text:
            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)
                    images.append(img_url)
    return images


def normalize_answer_images(answer: str, image_urls) -> str:
    """
    统一整理答案中的图片块。

    执行步骤：
    1. 先清理答案里已有的旧图片块，避免重复拼接。
    2. 如果没有图片 URL，直接返回清理后的文本。
    3. 如果有图片 URL，则在答案末尾追加：
       - `【图片】`
       - 每行一个 `<url>`
    """
    answer = (answer or "").strip()
    image_urls = image_urls or []
    image_block_pattern = re.compile(r"\n*【图片】\s*\n(?:<.*?>\s*\n?)*", re.MULTILINE)

    if not image_urls:
        return image_block_pattern.sub("", answer).strip()

    cleaned_answer = image_block_pattern.sub("", answer).strip()
    image_block = IMAGE_BLOCK_MARKER + "\n" + "\n".join(f"<{url}>" for url in image_urls)
    return f"{cleaned_answer}\n\n{image_block}".strip()
