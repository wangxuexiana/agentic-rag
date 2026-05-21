"""
Neo4j 图数据库客户端工具模块

功能概述：
    封装 Neo4j 图数据库的连接逻辑，提供单例驱动实例。
    知识图谱（KG）查询节点（node_query_kg）通过本模块获取驱动实例执行 Cypher 查询。

当前状态：
    该模块为基础封装，KG 查询功能尚在预留阶段，后续可扩展为完整的图查询工具。

配置依赖（.env）：
    - NEO4J_URI: Neo4j 服务端连接地址（如 bolt://localhost:7687）
    - NEO4J_USERNAME: 数据库用户名
    - NEO4J_PASSWORD: 数据库密码
"""
import os
from neo4j import GraphDatabase

# 全局单例：Neo4j驱动实例，避免重复创建连接
_neo4j_driver = None

def get_neo4j_driver() -> GraphDatabase:
    """
    获取 Neo4j 数据库驱动单例实例

    初始化策略：
        全局仅创建一次驱动实例，后续调用直接返回缓存。
        驱动实例管理连接池，可安全在多线程环境中使用。

    Returns:
        GraphDatabase.driver: Neo4j 驱动实例，可通过 session() 方法创建会话执行 Cypher 查询

    典型用法：
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN n LIMIT 10")
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),  # Neo4j连接地址（bolt://host:port）
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))  # 认证信息
        )
    return _neo4j_driver


# ──────────────────────────────────────────────────────────
# 📖 阅读导航
# 上一篇: app/clients/mongo_history_utils.py
# 下一篇: app/lm/llm_utils.py
# ──────────────────────────────────────────────────────────