"""
查询流程状态定义模块（QueryGraphState）

功能概述：
    定义查询流程（query_process）的完整状态结构，贯穿"规划→检索→融合→反思→生成"全链路。
    所有节点共享同一状态对象，通过字典式读写实现节点间数据传递。

状态数据流向：
    用户输入 → Planner规划 → 商品名确认 → 多路检索 → RRF融合 → Rerank重排
    → 证据反思 → (动态补检索) → 答案生成 → 历史记录写入

核心类型定义：
    - IntentType: 用户意图类型（问答/对比/操作指导/故障排查/参数查询/未知）
    - TaskType: 任务路由类型（仅知识库/知识库+网络/仅图谱/知识库+图谱/全量/澄清）
    - EvidenceStatus: 证据充分性状态（未评估/充分/不足/冲突）
    - ToolName: 可调用的检索工具名称（embedding/hyde/kg/web_search）
    - RetrievalPlan: 检索计划结构（由Planner节点生成，指导后续检索策略）
    - QueryGraphState: 查询流程全量状态（TypedDict，所有字段可选，支持渐进填充）

阅读提示：
    阅读各个节点时，可以把状态字段按四层来理解：
    1. 输入层：用户问题、session、history
    2. 规划层：intent、task_type、selected_tools
    3. 检索层：各路召回结果、融合结果、重排结果
    4. 治理与输出层：evidence_status、missing_facts、citations、answer
"""
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


# ==================== Java 开发者阅读提示 ====================
# 1. TypedDict
#    可以把它理解成“对 dict 做了字段约束的接口定义”。
#    如果你熟悉 Java，可以近似类比成：
#    - Map<String, Object> 的动态读写方式
#    - 再额外加上一层“这个 Map 里通常有哪些 key”的静态说明
#
# 2. total=False
#    表示下面定义的所有字段都不是强制必填。
#    这和 Java 里一个 DTO 所有字段都允许为 null 的感觉接近，
#    只是 Python 这里仍然是用 dict 存，而不是强类型 POJO。
#
# 3. Literal["a", "b", "c"]
#    可以近似理解成“字符串枚举”，只是 Python 没有真的声明 enum 类，
#    而是通过类型提示约束这个字段只应该出现这些固定字符串。
#
# 4. state.get("xxx")
#    这是 Python dict 的内置方法：
#    - dict.get(key)              -> 取值；如果 key 不存在，返回 None
#    - dict.get(key, default)     -> 取值；如果 key 不存在，返回 default
#    你可以把它类比成 Java 里：
#    - map.get("xxx")
#    - map.getOrDefault("xxx", defaultValue)
#
# 5. 为什么查询流程要用一个大 state 传递？
#    因为 LangGraph 的节点之间，本质上就是不断读写同一个状态对象。
#    每个节点只关心自己要读的字段、要写的字段。
#    这种设计比“一个函数把几十个参数层层传下去”更适合复杂工作流。
# ===========================================================


# 用户意图类型
IntentType = Literal[
    "qa",                # 通用问答：事实性知识查询
    "comparison",        # 对比分析：多产品/多参数横向比较
    "operation_guide",   # 操作指导：配置、安装、使用步骤
    "troubleshooting",   # 故障排查：异常诊断与修复
    "parameter_query",   # 参数查询：规格参数、指标数据
    "unknown",           # 未识别意图，需进一步澄清
]

# 任务路由类型：决定查询走哪条处理链路
TaskType = Literal[
    "kb_only",          # 仅知识库检索
    "kb_with_web",      # 知识库 + 网络搜索补充
    "kg_only",          # 仅知识图谱检索
    "kb_with_kg",       # 知识库 + 知识图谱联合检索
    "full_agentic",     # 全量智能体模式：多工具自主编排
    "clarification",    # 澄清模式：信息不足，反问用户
]

# 证据充分性状态：评估检索结果是否足以回答问题
EvidenceStatus = Literal[
    "unknown",          # 未评估
    "sufficient",       # 证据充分，可直接生成答案
    "insufficient",     # 证据不足，需补充检索
    "conflicting",      # 证据冲突，需消歧或综合判断
]

# 可调用的检索工具名称
ToolName = Literal[
    "embedding",        # 密集向量嵌入检索
    "hyde",             # 假设文档嵌入（HyDE）检索
    "kg",               # 知识图谱检索
    "web_search",       # 网络搜索
]


class RetrievalPlan(TypedDict, total=False):
    """
    检索计划：由 Agent 规划节点生成，指导后续检索策略。

    total=False 表示所有字段均为可选，允许在不同阶段渐进填充。
    """
    intent_type: IntentType           # 识别出的用户意图
    task_type: TaskType               # 路由到的任务类型
    selected_tools: List[ToolName]    # 本轮计划调用的检索工具列表
    need_clarify: bool                # 是否需要向用户追问澄清
    success_criteria: str             # 检索成功的判别标准（供反思节点使用）
    notes: str                        # 规划备注，记录决策理由


class QueryGraphState(TypedDict, total=False):
    """
    查询流程状态定义，贯穿"规划→检索→融合→反思→生成"全链路。

    数据流向：基础输入 → Agent规划 → 多轮检索 → 融合重排 → 证据反思 → 答案生成
    total=False 表示所有字段均为可选，支持不同节点按需读写。
    """

    # 读这份 state 的最好方式不是死记字段，而是按链路看数据如何演进：
    # 1. original_query 进入系统
    # 2. rewritten_query / item_names 在前置节点中被改写和补充
    # 3. selected_tools / run_xxx 决定检索支路
    # 4. embedding_chunks / hyde_embedding_chunks / web_search_docs 等承载原始召回结果
    # 5. rrf_chunks / reranked_docs 承载融合与精排后的高质量证据
    # 6. evidence_status / missing_facts 决定是直接回答还是补检索
    # 7. prompt / answer 是最终输出层结果
    # --- 基础输入：用户的原始请求与输出控制 ---
    session_id: str                   # 会话唯一标识，用于关联上下文
    original_query: str               # 用户原始问题
    rewritten_query: str              # 改写后的问题（优化检索效果）
    is_stream: bool                   # 是否流式输出

    # --- 会话上下文：历史对话与产品信息 ---
    history: List[Dict[str, Any]]     # 历史对话记录，格式 [{"role": ..., "content": ...}]
    item_names: List[str]             # 从问题中提取的产品/设备名称，用于检索增强

    # --- Agent 规划层：意图识别与检索策略 ---
    intent_type: IntentType           # 识别出的用户意图类型
    task_type: TaskType               # 路由到的任务处理类型
    retrieval_plan: RetrievalPlan     # 完整的检索计划
    selected_tools: List[ToolName]    # 本轮选中的检索工具
    need_clarify: bool                # 是否需要向用户追问
    clarification_question: str       # 追问用户时的澄清问题


    # --- Router 执行开关：由路由节点决定各检索通道是否执行 ---
    run_embedding: bool               # 是否执行密集向量嵌入检索
    run_hyde: bool                    # 是否执行 HyDE（假设文档嵌入）检索
    run_kg: bool                      # 是否执行知识图谱检索
    run_web_search: bool              # 是否执行网络搜索
    router_reason: str                # 路由决策理由，记录为何选择/跳过各通道


    # --- 检索轮次控制：多轮检索的进度与约束 ---
    retrieval_round: int              # 当前检索轮次（从1开始）
    max_retrieval_rounds: int         # 最大允许检索轮次，防止无限循环
    followup_query: str               # 上一轮反思后生成的补充检索查询
    retry_intent: str                 # 本轮补检索的重点说明

    # --- 多路检索结果：各检索通道返回的原始切片 ---
    embedding_chunks: list            # 密集向量嵌入检索返回的切片
    hyde_embedding_chunks: list       # HyDE（假设文档嵌入）检索返回的切片
    kg_chunks: list                   # 知识图谱检索返回的切片
    web_search_docs: list             # 网络搜索返回的文档

    # --- 融合/重排结果：多路检索结果的合并与排序 ---
    rrf_chunks: list                  # RRF（倒数排名融合）合并后的切片
    reranked_docs: list               # 重排序后的最终 Top-K 文档

    # --- 证据反思层：评估检索结果质量与充分性 ---
    evidence_status: EvidenceStatus   # 证据充分性评估结果
    reflection_reason: str            # 反思推理过程说明
    missing_facts: List[str]          # 识别出的缺失事实点（用于补充检索）
    citations: List[Dict[str, Any]]   # 引用来源列表，格式 [{"source": ..., "chunk_id": ...}]
    final_confidence: float           # 最终答案的置信度（0.0~1.0）
    support_score: float              # 支持度评分：证据是否直接支撑答案
    coverage_score: float             # 覆盖度评分：是否覆盖问题关键点
    consistency_score: float          # 一致性评分：多条证据之间是否一致

    # --- 生成层：Prompt 组装与答案输出 ---
    prompt: str                       # 组装好的完整 Prompt
    answer: str                       # 最终生成的回答
    cache_stats: Dict[str, Any]       # 查询链缓存命中明细与 backend 统计

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/agent/nodes/node_planner.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
