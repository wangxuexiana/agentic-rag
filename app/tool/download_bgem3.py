"""
BGE-M3 模型下载工具

本脚本用于从 ModelScope 下载 BAAI/bge-m3 多语言嵌入模型到本地缓存目录。
BGE-M3 是本项目核心的嵌入模型，支持稠密向量（Dense）和稀疏向量（Sparse）双向量输出，
用于 Milvus 混合检索的向量化环节。

运行方式：
    python app/tool/download_bgem3.py

模型缓存路径：D:/ai_models/modelscope_cache/models
"""

from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到当前目录下的 models/bge-m3 文件夹
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache/models')
print(f"模型已下载到: {model_dir}")