"""
Planner 服务层。

职责边界：
1. 承载 planner 的规则判断、工具标准化、fallback、JSON 解析纠偏。
2. 不负责 LangGraph 节点编排，节点层只调用这里的方法并写回 state。
3. 把“规划规则”和“流程控制”拆开，降低 node_planner.py 的耦合度。

阅读顺序建议：
1. 先看 `build_planner_prompt()`，理解 Planner 让 LLM 输出什么。
2. 再看 `parse_planner_output()`，理解模型输出如何被解析和纠偏。
3. 最后看几组 `_looks_like_*` 规则函数，理解具体纠偏依据。
"""

from __future__ import annotations

import json
import re


def build_history_text(history: list) -> str:
    """
    把历史对话列表转成一段纯文本，供 Planner prompt 使用。

    执行步骤：
    1. 遍历历史消息。
    2. 提取每条消息的 `role` 和 `text`。
    3. 组装成 `role: text` 的单行文本。
    4. 最后用换行拼接成完整上下文。
    """
    if not history:
        return ""

    lines = []
    for msg in history:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_planner_prompt(history_text: str, rewritten_query: str) -> str:
    """
    构造 Planner prompt。

    执行步骤：
    1. 告诉模型它的职责不是回答问题，而是规划后续检索。
    2. 明确 JSON 输出字段。
    3. 给出检索规则和澄清规则。
    4. 注入历史对话与当前问题。
    """
    return f"""
你是一个 Agentic RAG 系统的 Planner。

你的任务不是回答问题，而是先规划“接下来该怎么查”。

请根据用户问题和历史对话，输出 JSON，字段必须包括：

{{
  "intent_type": "qa | comparison | operation_guide | troubleshooting | parameter_query | unknown",
  "task_type": "kb_only | kb_with_web | kg_only | kb_with_kg | full_agentic | clarification",
  "selected_tools": ["embedding", "hyde", "kg", "web_search"],
  "need_clarify": true,
  "success_criteria": "一句话说明这次检索成功的标准",
  "notes": "补充说明为什么这么规划"
}}

规则：
1. 如果问题明显是设备说明书、操作步骤、参数查询，优先本地知识库。
2. 如果问题可能依赖官网、驱动、下载、版本、时效信息，可以加入 web_search。
3. 如果问题依赖实体关系或结构化关系，可加入 kg。
4. 如果问题模糊到无法继续检索，need_clarify 设为 true，task_type 设为 clarification。
5. 只能输出 JSON，不要输出额外解释。

历史对话：
{history_text if history_text else "无"}

当前问题：
{rewritten_query}
""".strip()


def _looks_like_product_model(query: str) -> bool:
    if not query:
        return False

    patterns = [
        r"[A-Za-z]{2,}\s?-?\d{2,}",
        r"[A-Za-z]+\d{2,}",
        r"\d{2,}[A-Za-z]+",
    ]
    return any(re.search(pattern, query) for pattern in patterns)


def _looks_like_actionable_question(query: str) -> bool:
    if not query:
        return False

    keywords = [
        "怎么", "如何", "步骤", "操作", "设置", "参数", "温度",
        "是什么", "说明书", "故障", "报错", "驱动", "下载", "价格",
        "使用", "安装", "连接", "配置",
    ]
    return any(word in query for word in keywords)


def _looks_like_general_qa_query(query: str) -> bool:
    if not query:
        return False

    keywords = [
        "支持", "官网", "资料", "是什么", "什么类型", "属于什么", "做什么",
        "介绍", "用途", "功能", "蓝牙", "wifi", "wi-fi", "联网",
    ]
    lowered = query.lower()
    return any(word in query for word in keywords) or any(word in lowered for word in ["wifi", "wi-fi"])


def _has_specific_subject_signal(query: str) -> bool:
    if not query:
        return False

    patterns = [
        r"测试设备[A-Za-z0-9]+",
        r"设备[A-Za-z0-9]{2,}",
        r"[A-Za-z]{2,}\s?-?\d{1,}[A-Za-z0-9-]*",
        r"[\u4e00-\u9fffA-Za-z]{1,20}\s?[A-Za-z]?\d{2,}[A-Za-z0-9-]*",
    ]

    if _looks_like_product_model(query):
        return True
    return any(re.search(pattern, query) for pattern in patterns)


def _looks_like_generic_ambiguous_query(query: str) -> bool:
    if not query:
        return False

    generic_subjects = [
        "打印机", "平板", "手机", "路由器", "显示器", "设备", "仪器",
        "电脑", "笔记本", "耳机", "相机", "服务器", "交换机",
    ]
    pronouns = ["这个", "那个", "它", "这款", "那款", "该设备", "该产品"]

    has_generic_subject = any(word in query for word in generic_subjects)
    has_pronoun = any(word in query for word in pronouns)
    actionable = _looks_like_actionable_question(query)
    has_specific_subject = _has_specific_subject_signal(query)
    return (has_pronoun or has_generic_subject) and actionable and not has_specific_subject


def _looks_like_model_only_query(query: str) -> bool:
    if not query:
        return False

    stripped = query.strip()
    if len(stripped) > 24 or " " in stripped:
        return False
    if not _has_specific_subject_signal(stripped):
        return False
    return not _looks_like_actionable_question(stripped)


def _looks_like_comparison_query(query: str) -> bool:
    if not query:
        return False

    keywords = ["比较", "对比", "区别", "差别", "相比", "哪个", "谁更", "更大", "更高", "更重", "更快"]
    return any(word in query for word in keywords)


def _comparison_has_explicit_targets(query: str) -> bool:
    if not query:
        return False

    separators = ["和", "与", "跟", "及", "、", "vs", "VS"]
    if not any(sep in query for sep in separators):
        return False

    for sep in separators:
        if sep not in query:
            continue
        parts = [p.strip(" ，。！？（）()") for p in query.split(sep)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            left_ok = _has_specific_subject_signal(parts[0]) or len(parts[0]) >= 3
            right_ok = _has_specific_subject_signal(parts[1]) or len(parts[1]) >= 3
            if left_ok and right_ok:
                return True
    return False


def build_fallback_plan(query: str) -> dict:
    """
    当 Planner LLM 失败时，返回一份保守但可执行的计划。

    执行步骤：
    1. 默认优先本地知识库。
    2. 默认保留 embedding + hyde。
    3. 只有问题明显依赖官网/下载/版本/价格/时效信息时，再补 web_search。
    """
    lower_query = query.lower()
    need_web_keywords = [
        "官网", "最新", "现在", "价格", "驱动", "下载", "版本", "联网", "网页",
        "官方", "今日", "最近",
    ]
    need_web = any(word in query for word in need_web_keywords)

    selected_tools = ["embedding", "hyde"]
    if need_web:
        selected_tools.append("web_search")

    return {
        "intent_type": "unknown",
        "task_type": "kb_with_web" if need_web else "kb_only",
        "selected_tools": selected_tools,
        "need_clarify": False,
        "success_criteria": "找到足够回答当前问题的证据，并尽量优先使用本地知识库。",
        "notes": f"fallback plan for query: {lower_query[:100]}",
    }


def default_tools_for_task_type(task_type: str) -> list:
    if task_type == "clarification":
        return []
    if task_type == "kb_with_web":
        return ["embedding", "hyde", "web_search"]
    if task_type in {"kb_only", "full_agentic"}:
        return ["embedding", "hyde"]
    if task_type == "kb_with_kg":
        return ["embedding", "hyde", "kg"]
    if task_type == "kg_only":
        return ["kg"]
    return ["embedding", "hyde"]


def enrich_selected_tools(selected_tools: list, task_type: str) -> list:
    normalized = list(selected_tools)

    if task_type in {"kb_only", "kb_with_web", "kb_with_kg", "full_agentic"}:
        for tool in ["embedding", "hyde"]:
            if tool not in normalized:
                normalized.append(tool)

    if task_type == "kb_with_web" and "web_search" not in normalized:
        normalized.append("web_search")

    return normalized


def normalize_selected_tools(selected_tools, task_type: str) -> list:
    """
    清洗 Planner 产出的 selected_tools。

    执行步骤：
    1. 过滤非法工具名。
    2. 对合法工具去重。
    3. 如果为空且 task_type=clarification，则允许空。
    4. 否则按 task_type 给默认工具，并做补齐。
    """
    allowed_tools = {"embedding", "hyde", "kg", "web_search"}

    if not isinstance(selected_tools, list):
        selected_tools = []

    normalized = []
    for tool in selected_tools:
        if tool in allowed_tools and tool not in normalized:
            normalized.append(tool)

    if normalized:
        return enrich_selected_tools(normalized, task_type)

    if task_type == "clarification":
        return []

    return default_tools_for_task_type(task_type)


def build_clarification_question(intent_type: str) -> str:
    """根据意图类型生成默认澄清问句。"""
    if intent_type == "comparison":
        return "请明确要比较的两个产品或型号，我再继续帮你对比。"
    return "请补充更具体的产品型号、场景或问题细节。"


def parse_planner_output(raw_text: str, query: str, logger) -> dict:
    """
    解析并纠偏 Planner 的 JSON 输出。

    执行步骤：
    1. 去掉 ```json 包裹。
    2. 解析成 dict，失败则 fallback。
    3. 补默认字段并清洗 selected_tools。
    4. 用规则函数对 LLM 结果做纠偏。
    5. 返回最终可执行的规划结果。
    """
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        result = json.loads(text)
    except Exception as exc:
        logger.error(f"Planner JSON 解析失败: {exc}")
        return build_fallback_plan(query)

    result.setdefault("intent_type", "unknown")
    result.setdefault("task_type", "full_agentic")
    result.setdefault("selected_tools", [])
    result.setdefault("need_clarify", False)
    result.setdefault("success_criteria", "找到足够回答问题的证据。")
    result.setdefault("notes", "")
    result["selected_tools"] = normalize_selected_tools(
        result.get("selected_tools"),
        result.get("task_type", "full_agentic"),
    )

    if result.get("task_type") == "clarification" and _looks_like_product_model(query):
        logger.info("Planner 结果规则纠偏：检测到具体型号，改为 kb_only")
        result["task_type"] = "kb_only"
        result["need_clarify"] = False
        result["selected_tools"] = default_tools_for_task_type("kb_only")
        result["success_criteria"] = "优先从本地知识库中找到能回答该设备问题的证据。"
        result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=model_query"

    if _looks_like_general_qa_query(query):
        if result.get("intent_type") in {"parameter_query", "unknown"}:
            logger.info("Planner 结果规则纠偏：检测到能力/介绍/官网类问法，intent 改为 qa")
            result["intent_type"] = "qa"
            result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=qa_intent"

        if _has_specific_subject_signal(query) and result.get("task_type") in {"kb_with_web", "full_agentic"}:
            logger.info("Planner 结果规则纠偏：具体对象 QA 问题优先改为 kb_only")
            result["task_type"] = "kb_only"
            result["need_clarify"] = False
            result["selected_tools"] = default_tools_for_task_type("kb_only")
            result["success_criteria"] = "优先在本地知识库中确认该对象的能力、属性或介绍信息，证据不足时再进入二轮补检索。"
            result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=qa_kb_first"

    if _looks_like_model_only_query(query):
        logger.info("Planner 结果规则纠偏：检测到只有型号但缺少明确诉求，改为 clarification")
        result["task_type"] = "clarification"
        result["need_clarify"] = True
        result["selected_tools"] = []
        result["success_criteria"] = "先确认用户想了解该型号的哪方面信息，再继续检索。"
        result["clarification_question"] = "请说明你想了解这个型号的哪方面信息，例如参数、价格、安装步骤或故障处理。"
        result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=model_only_clarify"

    elif _looks_like_generic_ambiguous_query(query):
        logger.info("Planner 结果规则纠偏：检测到泛问题且对象不明确，改为 clarification")
        result["task_type"] = "clarification"
        result["need_clarify"] = True
        result["selected_tools"] = []
        result["success_criteria"] = "先确认具体产品型号或对象范围，再继续检索。"
        result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=generic_query_clarify"

    if _looks_like_comparison_query(query) and not _comparison_has_explicit_targets(query):
        logger.info("Planner 结果规则纠偏：比较题对象不明确，改为 clarification")
        result["intent_type"] = "comparison"
        result["task_type"] = "clarification"
        result["need_clarify"] = True
        result["selected_tools"] = []
        result["success_criteria"] = "先确认比较的两个对象，再继续检索。"
        result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=comparison_targets_missing"

    elif _looks_like_comparison_query(query) and _comparison_has_explicit_targets(query):
        if result.get("task_type") != "kb_only":
            logger.info("Planner 结果规则纠偏：比较题对象明确，优先改为 kb_only")
            result["intent_type"] = "comparison"
            result["task_type"] = "kb_only"
            result["need_clarify"] = False
            result["selected_tools"] = default_tools_for_task_type("kb_only")
            result["success_criteria"] = "优先从本地知识库中召回两个对象的关键参数或事实用于比较。"
            result["notes"] = f"{result.get('notes', '')} | corrected_by_rule=comparison_kb_first"

    return result
