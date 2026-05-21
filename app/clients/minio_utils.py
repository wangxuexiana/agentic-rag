import json

from minio import Minio

from app.config.minio_config import minio_config
from app.core.logger import logger

minio_client = None

try:
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure,
    )
    bucket_name = minio_config.bucket_name

    if not minio_client.bucket_exists(bucket_name):
        logger.info(f"MinIO存储桶[{bucket_name}]不存在，开始创建")
        minio_client.make_bucket(bucket_name)
        logger.info(f"MinIO存储桶[{bucket_name}]创建成功")
    else:
        logger.info(f"MinIO存储桶[{bucket_name}]已存在，无需重复创建")

    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
        }],
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
    logger.info(f"MinIO存储桶[{bucket_name}]已配置公网只读策略，支持匿名URL访问")

except Exception as e:
    logger.error(f"MinIO客户端初始化失败，错误信息：{str(e)}", exc_info=True)
    minio_client = None


def get_minio_client():
    return minio_client
