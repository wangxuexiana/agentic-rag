"""
BGE-M3 模型下载/续传测试脚本

本脚本用于从 ModelScope 下载或续传 BAAI/bge-m3 嵌入模型。
支持断点续传：如果模型文件已部分下载，会自动识别已有文件并只下载缺失部分。

运行方式：
    python test/download_model.py
"""

from modelscope.hub.snapshot_download import snapshot_download

# 指定模型名称
model_id = 'BAAI/bge-m3'

# 指定你报错日志里的缓存根目录
# 注意：这里写到 modelscope_cache 即可，底层会自动拼出 models\BAAI\bge-m3
cache_directory = 'D:\\ai_models\\modelscope_cache'

print("开始检查并继续下载 BGE-M3 模型...")

# 执行下载（它会自动识别已有的文件，只下缺少的）
model_dir = snapshot_download(
    model_id,
    cache_dir=cache_directory,
    revision='master' # 默认拉取最新版
)

print(f"✅ 模型下载或校验完成！完整路径为: {model_dir}")