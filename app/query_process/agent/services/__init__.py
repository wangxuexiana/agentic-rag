"""Query agent service helpers.

这个目录放“可复用的领域服务”，不直接参与图路由，只承载具体业务逻辑。
节点文件负责：
1. 从 state 里取字段
2. 调服务
3. 把结果写回 state
"""
