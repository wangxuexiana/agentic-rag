"""
文档切分节点。

这一版的目标不是单纯把文档切得更碎，而是让 chunk 更稳定：
1. 先按 Markdown 标题建立结构路径，减少跨章节漂移。
2. 再按段落、列表、表格、代码块做语义分块。
3. 只有在同一章节内、且两个块都很短时才做保守合并。

这样做的直接收益是：
- 文档局部修改时，更容易只影响局部 chunk。
- `section_path` 更稳定，`chunk_key` 的命中率更高。
- 后续召回和增量入库都更可控。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task


DEFAULT_MAX_CONTENT_LENGTH = 2000
# 只有两个短块都小于该阈值时才考虑合并，避免小改动牵连整章重排。
MIN_MERGE_CONTENT_LENGTH = 220
# 合并后的内容也不能太大，否则宁可保留两个块。
MAX_MERGED_CONTENT_LENGTH = 700


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分主节点。

    执行顺序：
    1. 读取并标准化 Markdown。
    2. 按标题构建章节树路径。
    3. 对每个章节按语义块切分，再对超长块继续拆分。
    4. 对极短块做保守合并。
    5. 补齐增量同步依赖的元数据。
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点：{node_name}")
    add_running_task(state["task_id"], node_name)

    try:
        content, file_title, max_len = step_1_get_inputs(state)
        if content is None:
            logger.info(f">>> 节点结束：{node_name}，无有效 Markdown 内容")
            return state

        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)
        sections = step_3_handle_no_title(content, sections, title_count, file_title)
        chunks = step_4_refine_chunks(sections, max_len, file_title)
        enrich_chunk_metadata(
            chunks,
            state.get("doc_id", ""),
            state.get("doc_version", ""),
        )

        state["chunks"] = chunks
        step_5_print_stats(lines_count, chunks)
        step_6_backup(state, chunks)
        logger.info(f">>> 节点完成：{node_name}，共生成 {len(chunks)} 个 chunk")
    except Exception as exc:
        logger.error(f">>> 节点失败：{node_name}，错误：{exc}", exc_info=True)

    return state


def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """
    读取输入并统一换行。
    """
    content = state.get("md_content")
    if not content:
        logger.warning("状态中没有可切分的 Markdown 内容，跳过文档切分")
        return None, None, None

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "Unknown File")
    max_len = int(state.get("max_content_length") or DEFAULT_MAX_CONTENT_LENGTH)
    logger.info(f"步骤1：输入加载完成，file_title={file_title}，max_len={max_len}")
    return content, file_title, max_len


def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    按 Markdown 标题切出基础章节，并记录完整标题路径。

    每个 section 会带上：
    - `title_level`：当前标题层级
    - `title_path`：从一级标题到当前标题的完整路径
    - `body`：当前标题下的正文，不重复包含标题行
    """
    title_pattern = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
    lines = content.split("\n")
    sections: List[Dict[str, Any]] = []
    title_stack: List[str] = []
    current_section: Dict[str, Any] | None = None
    in_code_block = False
    title_count = 0

    def flush_current_section() -> None:
        if current_section is None:
            return
        body = "\n".join(current_section.pop("body_lines", [])).strip("\n")
        current_section["body"] = body
        current_section["content"] = build_section_content(current_section["title"], body)
        sections.append(current_section.copy())

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            if current_section is not None:
                current_section["body_lines"].append(line)
            continue

        match = None if in_code_block else title_pattern.match(line)
        if match:
            flush_current_section()
            level = len(match.group(1))
            title_text = match.group(2).strip()

            if len(title_stack) >= level:
                title_stack[:] = title_stack[: level - 1]
            title_stack.append(title_text)

            current_section = {
                "title": title_text,
                "title_level": level,
                "title_path": title_stack.copy(),
                "body_lines": [],
                "file_title": file_title,
            }
            title_count += 1
            continue

        if current_section is None:
            current_section = {
                "title": file_title,
                "title_level": 0,
                "title_path": [file_title],
                "body_lines": [],
                "file_title": file_title,
            }
        current_section["body_lines"].append(line)

    flush_current_section()
    logger.info(f"步骤2：标题切分完成，识别到 {title_count} 个标题，总行数 {len(lines)}")
    return sections, title_count, len(lines)


def step_3_handle_no_title(
    content: str,
    sections: List[Dict[str, Any]],
    title_count: int,
    file_title: str,
) -> List[Dict[str, Any]]:
    """
    无标题文档兜底。
    """
    if title_count > 0:
        return sections

    logger.warning(f"步骤3：文档未识别到 Markdown 标题，按单章节处理：{file_title}")
    return [
        {
            "title": file_title,
            "title_level": 0,
            "title_path": [file_title],
            "body": content.strip("\n"),
            "content": build_section_content(file_title, content.strip("\n")),
            "file_title": file_title,
        }
    ]


def step_4_refine_chunks(
    sections: List[Dict[str, Any]],
    max_len: int,
    file_title: str,
) -> List[Dict[str, Any]]:
    """
    将基础章节细化为最终 chunk。
    """
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：max_len={max_len} 非法，直接返回原章节")
        return sections

    refined_chunks: List[Dict[str, Any]] = []
    for section in sections:
        refined_chunks.extend(split_section_to_chunks(section, max_len, file_title))
    logger.info(f"步骤4-1：语义切分完成，得到 {len(refined_chunks)} 个初始 chunk")

    merged_chunks = merge_short_chunks(refined_chunks, max_len)
    assign_part_numbers(merged_chunks)
    logger.info(f"步骤4-2：保守合并完成，最终保留 {len(merged_chunks)} 个 chunk")
    return merged_chunks


def split_section_to_chunks(
    section: Dict[str, Any],
    max_len: int,
    file_title: str,
) -> List[Dict[str, Any]]:
    """
    将单个 section 切成多个稳定 chunk。

    规则：
    - 优先按段落、列表、表格、代码块切。
    - 在同一 section 内顺序累积，直到接近长度阈值。
    - 单个语义块过长时，再使用递归切分器继续拆。
    """
    title = (section.get("title") or file_title or "Untitled").strip()
    title_path = section.get("title_path") or [title]
    body = (section.get("body") or "").strip("\n")
    blocks = semantic_blocks_from_body(body)

    if not blocks:
        return [
            create_chunk_from_body(
                title=title,
                title_path=title_path,
                body="",
                file_title=file_title,
                title_level=int(section.get("title_level", 0) or 0),
            )
        ]

    chunks: List[Dict[str, Any]] = []
    current_blocks: List[str] = []

    for block in blocks:
        if current_blocks:
            candidate_body = "\n\n".join(current_blocks + [block]).strip()
            candidate_content = build_section_content(title, candidate_body)
            if len(candidate_content) <= max_len:
                current_blocks.append(block)
                continue

        if current_blocks:
            chunks.append(
                create_chunk_from_body(
                    title=title,
                    title_path=title_path,
                    body="\n\n".join(current_blocks).strip(),
                    file_title=file_title,
                    title_level=int(section.get("title_level", 0) or 0),
                )
            )
            current_blocks = []

        single_content = build_section_content(title, block)
        if len(single_content) <= max_len:
            current_blocks = [block]
            continue

        chunks.extend(
            split_oversize_block(
                block=block,
                title=title,
                title_path=title_path,
                file_title=file_title,
                title_level=int(section.get("title_level", 0) or 0),
                max_len=max_len,
            )
        )

    if current_blocks:
        chunks.append(
            create_chunk_from_body(
                title=title,
                title_path=title_path,
                body="\n\n".join(current_blocks).strip(),
                file_title=file_title,
                title_level=int(section.get("title_level", 0) or 0),
            )
        )

    return chunks


def semantic_blocks_from_body(body: str) -> List[str]:
    """
    把 section 正文按语义边界切成 block。

    这里不直接按固定字符数切，是为了让修改尽量局部化。
    常见的语义边界包括：
    - 空行分段
    - 连续列表项
    - 连续表格行
    - 整个代码块
    """
    if not body:
        return []

    lines = body.split("\n")
    blocks: List[str] = []
    current_lines: List[str] = []
    current_kind: str | None = None
    in_code_block = False
    code_fence = ""

    def flush() -> None:
        nonlocal current_lines, current_kind
        text = "\n".join(current_lines).strip("\n")
        if text:
            blocks.append(text)
        current_lines = []
        current_kind = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            if not in_code_block:
                flush()
                in_code_block = True
                code_fence = fence
                current_kind = "code"
                current_lines = [line]
            else:
                current_lines.append(line)
                if fence == code_fence:
                    flush()
                    in_code_block = False
                    code_fence = ""
            continue

        if in_code_block:
            current_lines.append(line)
            continue

        if not stripped:
            flush()
            continue

        line_kind = classify_line_kind(stripped)
        if current_kind is None:
            current_kind = line_kind
            current_lines = [line]
            continue

        if line_kind == current_kind and line_kind in {"list", "table"}:
            current_lines.append(line)
            continue

        if line_kind == "paragraph" and current_kind == "paragraph":
            current_lines.append(line)
            continue

        flush()
        current_kind = line_kind
        current_lines = [line]

    flush()
    return blocks


def classify_line_kind(stripped_line: str) -> str:
    """
    识别当前行属于哪一类语义块。
    """
    if re.match(r"^([-*+]|\d+\.)\s+", stripped_line):
        return "list"
    if stripped_line.startswith("|") and stripped_line.endswith("|"):
        return "table"
    return "paragraph"


def split_oversize_block(
    block: str,
    title: str,
    title_path: List[str],
    file_title: str,
    title_level: int,
    max_len: int,
) -> List[Dict[str, Any]]:
    """
    单个 block 仍然过长时，再做二次切分。

    这里才使用字符切分器，尽量把“硬切”限制在最小范围内。
    """
    prefix = f"{title}\n\n" if title else ""
    available_len = max_len - len(prefix)
    if available_len <= 0:
        return [
            create_chunk_from_body(
                title=title,
                title_path=title_path,
                body=block,
                file_title=file_title,
                title_level=title_level,
            )
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    chunks: List[Dict[str, Any]] = []
    for text in splitter.split_text(block):
        normalized = text.strip()
        if not normalized:
            continue
        chunks.append(
            create_chunk_from_body(
                title=title,
                title_path=title_path,
                body=normalized,
                file_title=file_title,
                title_level=title_level,
            )
        )
    return chunks


def merge_short_chunks(chunks: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    保守合并短块。

    只在以下条件都满足时合并：
    - 两个 chunk 相邻
    - 属于同一个 `section_path`
    - 两个块本身都很短
    - 合并后仍不大
    """
    if not chunks:
        return []

    merged: List[Dict[str, Any]] = []
    current = chunks[0].copy()

    for next_chunk in chunks[1:]:
        if should_merge_chunks(current, next_chunk, max_len):
            merged_body = join_chunk_bodies(current.get("body", ""), next_chunk.get("body", ""))
            current["body"] = merged_body
            current["content"] = build_section_content(current.get("title", ""), merged_body)
            continue

        merged.append(current)
        current = next_chunk.copy()

    merged.append(current)
    return merged


def should_merge_chunks(current: Dict[str, Any], next_chunk: Dict[str, Any], max_len: int) -> bool:
    """
    判断两个相邻 chunk 是否应该合并。
    """
    if current.get("section_path") != next_chunk.get("section_path"):
        return False

    current_body = current.get("body", "") or ""
    next_body = next_chunk.get("body", "") or ""
    if not current_body or not next_body:
        return False

    if len(current_body) > MIN_MERGE_CONTENT_LENGTH or len(next_body) > MIN_MERGE_CONTENT_LENGTH:
        return False

    merged_body = join_chunk_bodies(current_body, next_body)
    merged_content = build_section_content(current.get("title", ""), merged_body)
    if len(merged_body) > MAX_MERGED_CONTENT_LENGTH or len(merged_content) > max_len:
        return False

    return True


def assign_part_numbers(chunks: List[Dict[str, Any]]) -> None:
    """
    为最终 chunk 重新编号。

    part 需要在最终稳定结构上编号，不能沿用合并前的临时序号。
    """
    path_counters: Dict[str, int] = {}
    for chunk in chunks:
        section_path = chunk.get("section_path", "") or "unknown"
        path_counters[section_path] = path_counters.get(section_path, 0) + 1
        chunk["part"] = path_counters[section_path]
        if not chunk.get("parent_title"):
            chunk["parent_title"] = chunk.get("title") or ""


def build_section_content(title: str, body: str) -> str:
    """
    统一 chunk 内容格式，避免不同步骤拼接方式不一致。
    """
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def create_chunk_from_body(
    title: str,
    title_path: List[str],
    body: str,
    file_title: str,
    title_level: int,
) -> Dict[str, Any]:
    """
    根据标题路径和正文内容生成 chunk。
    """
    normalized_title_path = [part.strip() for part in title_path if part and part.strip()]
    if not normalized_title_path:
        normalized_title_path = [title or file_title or "Untitled"]

    section_path = " / ".join(normalized_title_path)
    parent_title = normalized_title_path[-2] if len(normalized_title_path) > 1 else normalized_title_path[-1]
    content = build_section_content(title, body)
    return {
        "title": title,
        "title_level": title_level,
        "title_path": normalized_title_path,
        "section_path": section_path,
        "parent_title": parent_title,
        "body": body,
        "content": content,
        "file_title": file_title,
        "part": 0,
    }


def join_chunk_bodies(left_body: str, right_body: str) -> str:
    """
    合并正文时统一用空行分隔，尽量保留原始阅读结构。
    """
    left = (left_body or "").strip()
    right = (right_body or "").strip()
    if left and right:
        return f"{left}\n\n{right}"
    return left or right


def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
    输出切分统计，方便观察 chunk 稳定性。
    """
    logger.info("-" * 40 + " 文档切分统计 " + "-" * 40)
    logger.info(f"Markdown 总行数：{lines_count}")
    logger.info(f"最终 chunk 数量：{len(sections)}")
    if sections:
        logger.info(f"首个 chunk 标题：{sections[0].get('title', '')}")
        logger.info(f"首个 chunk 路径：{sections[0].get('section_path', '')}")
    logger.info("-" * 100)


def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """
    把切分结果落到本地 `chunks.json`，便于人工排查。
    """
    local_dir = state.get("local_dir")
    if not local_dir:
        logger.warning("步骤6：未配置 local_dir，跳过 chunk 备份")
        return

    try:
        os.makedirs(local_dir, exist_ok=True)
        backup_path = os.path.join(local_dir, "chunks.json")
        with open(backup_path, "w", encoding="utf-8") as file:
            json.dump(sections, file, ensure_ascii=False, indent=2)
        logger.info(f"步骤6：chunk 备份完成，路径：{backup_path}")
    except Exception as exc:
        logger.error(f"步骤6：chunk 备份失败，错误：{exc}", exc_info=False)


def enrich_chunk_metadata(sections: List[Dict[str, Any]], doc_id: str, doc_version: str) -> None:
    """
    补齐增量同步依赖的稳定元数据。

    - `section_path`：章节路径，尽量反映结构位置。
    - `chunk_hash`：正文内容指纹，用来判断是否改动。
    - `chunk_key`：chunk 身份指纹，用来判断是不是同一个结构块。
    """
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue

        title = (section.get("title") or "").strip()
        title_path = section.get("title_path") or [title or f"chunk-{index}"]
        normalized_path = [part.strip() for part in title_path if part and part.strip()]
        section_path = section.get("section_path") or " / ".join(normalized_path) or f"chunk-{index}"
        body = section.get("body", "")
        normalized_content = normalize_chunk_content(body or section.get("content", ""))

        section["doc_id"] = doc_id
        section["doc_version"] = doc_version
        section["title_path"] = normalized_path
        section["section_path"] = section_path
        section["chunk_hash"] = hashlib.sha1(normalized_content.encode("utf-8")).hexdigest()
        section["chunk_key"] = hashlib.sha1(
            f"{doc_id}|{section_path}|{int(section.get('part', 0) or 0)}".encode("utf-8")
        ).hexdigest()


def normalize_chunk_content(content: str) -> str:
    """
    计算 hash 前先折叠空白，避免只因空格或换行不同就误判为内容变化。
    """
    normalized = (content or "").strip()
    # 统一图片 Markdown 表达。
    # 真实文档更新时，常见情况是：
    # - 本地图片路径变成 MinIO/HTTP 地址
    # - 图片补充了更详细的 alt 文本
    # 这类变化通常不代表正文知识真的改了，如果直接参与 hash，会导致大量无意义 updated。
    normalized = re.sub(
        r"!\[[^\]]*\]\(([^)]+)\)",
        lambda match: f"[IMAGE:{normalize_image_reference(match.group(1))}]",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized)


def normalize_image_reference(reference: str) -> str:
    """
    把图片引用归一成稳定标识。

    目标是让下面这些写法在 hash 上被视为同一张图：
    - images/abc123.jpg
    - http://localhost:9000/.../abc123.jpg
    """
    ref = (reference or "").strip()
    if not ref:
        return ""

    # 优先取文件名。对于当前项目的图片链路，同一张图本质上由文件名唯一标识。
    filename = os.path.basename(ref)
    if filename:
        return filename.lower()
    return ref.lower()
