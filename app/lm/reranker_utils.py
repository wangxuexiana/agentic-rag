"""
BGE-Reranker 重排序模型工具模块

功能概述：
    封装 BAAI/bge-reranker-large 重排序模型的加载与调用逻辑。
    重排序模型用于对检索结果进行二次精排，提升最终返回文档的相关性。

核心设计：
    - 单例模式：模型全局仅加载一次，避免重复初始化浪费显存/时间
    - 本地优先：优先加载本地模型路径，路径无效时回退到HuggingFace仓库ID自动下载
    - 配置驱动：模型路径、设备、精度等参数均从环境变量读取

使用方式：
    from app.lm.reranker_utils import get_reranker_model
    reranker = get_reranker_model()
    scores = reranker.compute_score([("问题", "候选文档1"), ("问题", "候选文档2")])

依赖模型：
    BAAI/bge-reranker-large（BGE系列重排序模型，支持中英文）
"""
import os
import shutil
import zipfile

from FlagEmbedding import FlagReranker
from safetensors import safe_open
from app.config.reranker_config import reranker_config
from app.core.logger import logger

# 全局单例：存储FlagReranker模型实例，避免重复加载
_reranker_model = None
# 默认模型仓库标识（HuggingFace/ModelScope），本地路径无效时回退使用
DEFAULT_RERANKER_REPO = "BAAI/bge-reranker-large"
_RERANKER_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
)
_RERANKER_CONFIG_FILES = (
    "config.json",
    "configuration.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


def _has_reranker_weights(path: str | None) -> bool:
    if not path or not os.path.isdir(path):
        return False

    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.isfile(safetensors_path):
        try:
            with safe_open(safetensors_path, framework="pt"):
                return True
        except Exception:
            pass

    pytorch_bin_path = os.path.join(path, "pytorch_model.bin")
    if os.path.isfile(pytorch_bin_path) and zipfile.is_zipfile(pytorch_bin_path):
        return True

    return any(
        os.path.isfile(os.path.join(path, name))
        for name in ("tf_model.h5", "model.ckpt.index", "flax_model.msgpack")
    )


def _has_reranker_config(path: str | None) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "config.json"))


def _copy_missing_files(src_dir: str | None, dst_dir: str, filenames: tuple[str, ...]) -> None:
    if not src_dir or not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for name in filenames:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def _copy_missing_subdir(src_dir: str | None, dst_dir: str, subdir_name: str) -> None:
    if not src_dir or not os.path.isdir(src_dir):
        return
    src = os.path.join(src_dir, subdir_name)
    dst = os.path.join(dst_dir, subdir_name)
    if os.path.isdir(src) and not os.path.isdir(dst):
        shutil.copytree(src, dst)


def _build_hydrated_reranker_dir(primary_path: str, secondary_path: str) -> str | None:
    primary_has_config = _has_reranker_config(primary_path)
    primary_has_weights = _has_reranker_weights(primary_path)
    secondary_has_config = _has_reranker_config(secondary_path)
    secondary_has_weights = _has_reranker_weights(secondary_path)

    if primary_has_config and primary_has_weights:
        return primary_path
    if secondary_has_config and secondary_has_weights:
        return secondary_path

    can_hydrate_from_primary = primary_has_config and secondary_has_weights
    can_hydrate_from_secondary = secondary_has_config and primary_has_weights
    if not can_hydrate_from_primary and not can_hydrate_from_secondary:
        return None

    base_path = primary_path if can_hydrate_from_primary else secondary_path
    extra_path = secondary_path if can_hydrate_from_primary else primary_path
    hydrated_dir = os.path.join(os.path.dirname(base_path), ".hydrated", os.path.basename(base_path))

    if os.path.isdir(hydrated_dir):
        shutil.rmtree(hydrated_dir)

    _copy_missing_files(base_path, hydrated_dir, _RERANKER_CONFIG_FILES + _RERANKER_WEIGHT_FILES)
    _copy_missing_files(extra_path, hydrated_dir, _RERANKER_CONFIG_FILES + _RERANKER_WEIGHT_FILES)
    _copy_missing_subdir(base_path, hydrated_dir, "onnx")
    _copy_missing_subdir(extra_path, hydrated_dir, "onnx")

    if _has_reranker_config(hydrated_dir) and _has_reranker_weights(hydrated_dir):
        logger.info("Hydrated split reranker cache into usable dir: {}", hydrated_dir)
        return hydrated_dir
    return None


def _resolve_local_reranker_path(path: str | None) -> str | None:
    if not path:
        return None

    if _has_reranker_weights(path) and _has_reranker_config(path):
        return path

    normalized = os.path.normpath(path)
    parent_dir = os.path.dirname(os.path.dirname(normalized))
    vendor_name = os.path.basename(os.path.dirname(normalized))
    model_name = os.path.basename(normalized)
    temp_candidate = os.path.join(parent_dir, "._____temp", vendor_name, model_name)

    hydrated_candidate = _build_hydrated_reranker_dir(normalized, temp_candidate)
    if hydrated_candidate:
        return hydrated_candidate

    if _has_reranker_weights(temp_candidate) and _has_reranker_config(temp_candidate):
        logger.warning(
            "Configured reranker dir is incomplete, switch to temp dir: {}",
            temp_candidate,
        )
        return temp_candidate

    return path

def get_reranker_model():
    """
    获取BGE-Reranker-Large重排序模型单例实例

    初始化策略：
        1. 从配置读取本地模型路径，若路径有效（目录存在）则加载本地模型
        2. 若本地路径不存在，回退到DEFAULT_RERANKER_REPO（自动从远程仓库下载）
        3. 模型全局仅加载一次，后续调用直接返回缓存实例

    Returns:
        FlagReranker: 初始化完成的重排序模型实例，可调用compute_score方法

    典型用法：
        reranker = get_reranker_model()
        scores = reranker.compute_score([("query", "passage1"), ("query", "passage2")])
        # scores为浮点数列表，分数越高表示query与passage越相关
    """
    global _reranker_model
    if _reranker_model is None:
        # 从配置读取本地模型路径
        model_name_or_path = reranker_config.bge_reranker_large
        # 本地路径校验：配置了路径但目录不存在，回退到远程仓库ID
        model_name_or_path = _resolve_local_reranker_path(model_name_or_path)
        if model_name_or_path and not os.path.isdir(model_name_or_path):
            logger.warning(
                "Configured reranker path does not exist, fallback to repo id: {}",
                model_name_or_path,
            )
            model_name_or_path = DEFAULT_RERANKER_REPO
        elif model_name_or_path and (
            not _has_reranker_weights(model_name_or_path) or not _has_reranker_config(model_name_or_path)
        ):
            logger.info(
                "Configured reranker path is incomplete, fallback to repo id: {}",
                model_name_or_path,
            )
            model_name_or_path = DEFAULT_RERANKER_REPO

        candidate_paths = [model_name_or_path]
        if model_name_or_path != DEFAULT_RERANKER_REPO:
            candidate_paths.append(DEFAULT_RERANKER_REPO)

        last_error = None
        for candidate in candidate_paths:
            try:
                _reranker_model = FlagReranker(
                    model_name_or_path=candidate,  # 模型路径或仓库ID
                    device=reranker_config.bge_reranker_device,  # 运行设备（cuda:0/cpu）
                    use_fp16=reranker_config.bge_reranker_fp16  # 是否开启半精度推理（节省显存）
                )
                logger.info(f"BGE-Reranker模型初始化成功，模型路径：{candidate}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"BGE-Reranker模型初始化失败，候选路径：{candidate} | 错误：{str(e)}")

        if _reranker_model is None and last_error is not None:
            raise last_error
    return _reranker_model


# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/core/logger.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
