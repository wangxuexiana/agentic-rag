"""
BGE-Reranker-Large 模型下载工具

本脚本用于从 ModelScope 下载 BAAI/bge-reranker-large 重排序模型到本地缓存目录。
该模型是本项目检索流程中 Rerank 节点使用的核心模型，对多路检索结果进行二次精排，
显著提升最终答案的相关性。

运行方式：
    python app/tool/download_reranker.py

模型缓存路径：D:\\ai_models\\modelscope_cache\\models\\rerank
"""

from modelscope.hub.snapshot_download import snapshot_download

local_dir = r"D:\ai_models\modelscope_cache\models\rerank"

snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=local_dir,
)

print("下载完成，模型目录：", local_dir)