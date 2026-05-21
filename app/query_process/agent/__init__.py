"""
`query_process.agent` 负责查询智能体编排。

这里关注两件事：
1. `state.py` 定义所有节点共享的状态字段；
2. `main_graph.py` 决定节点之间如何串联、分叉和回环。
"""
