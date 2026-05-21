"""
PyTorch 与 CUDA 环境检测脚本

本脚本用于验证 PyTorch 是否正确安装以及 CUDA GPU 是否可用。
在部署本项目前运行此脚本，可快速排查 GPU 推理环境是否就绪。

运行方式：
    python test/pytorch.py

预期输出：
    - PyTorch 版本号
    - CUDA 是否可用（CPU 版本显示 False 为正常）
    - GPU 设备数量和名称
"""

try:
    import torch
    print(f"✅ PyTorch 加载成功！版本：{torch.__version__}")
    print(f"✅ CUDA 状态：{torch.cuda.is_available()}（CPU版显示False正常）")
    print(f"✅ CUDA 设备数：{torch.cuda.device_count()}")
    print(f"✅ CUDA 设备名称：{torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch 加载失败：{e}")