import gc
from enum import Enum
import torch
import psutil
from qwen_tts_webui.logger import get_logger

logger = get_logger()


class CPUState(Enum):
    """设备状态"""

    GPU = 0
    CPU = 1
    MPS = 2


OutOfMemoryError = torch.cuda.OutOfMemoryError

try:
    _ = torch.xpu.device_count()
    xpu_available = torch.xpu.is_available()
except:
    xpu_available = False

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

try:
    import torch_npu  # noqa: F401 # pylint: disable=import-error,unused-import

    _ = torch.npu.device_count()
    npu_available = torch.npu.is_available()
except:
    npu_available = False

try:
    import torch_mlu  # noqa: F401 # pylint: disable=import-error,unused-import

    _ = torch.mlu.device_count()
    mlu_available = torch.mlu.is_available()
except:
    mlu_available = False

try:
    ixuca_available = hasattr(torch, "corex")
except:
    ixuca_available = False

cpu_state = CPUState.GPU


def is_intel_xpu() -> bool:
    """设置是否为 Intel XPU

    :return `bool`: 检测结果
    """
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def is_ascend_npu():
    """设备是否为 NPU

    :return `bool`: 检测结果
    """
    global npu_available
    if npu_available:
        return True
    return False


def is_mlu():
    """设备是否为 MLU

    :return `bool`: 检测结果
    """
    global mlu_available
    if mlu_available:
        return True
    return False


def cleanup_models_gc() -> None:
    """清理模型缓存"""
    gc.collect()
    soft_empty_cache()


def soft_empty_cache() -> None:
    """清理模型缓存"""
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif is_ascend_npu():
        torch.npu.empty_cache()
    elif is_mlu():
        torch.mlu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_torch_device() -> torch.device:
    """获取 PyTorch 使用的设备

    :return `torch.device`: PyTorch 使用的设备类型
    """
    global cpu_state
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_free_memory(dev: torch.device | None = None, torch_free_too: bool | None = False) -> int | tuple[int, int]:
    """获取当前设备空闲内存

    :param dev`(torch.device|None)`: PyTorch 使用的设备
    :param torch_free_too`(bool|None)`: 显示 PyTorch 已保留的内存
    :return `int|tuple[int,int]`: 可用内存 / (可用内存 | PyTorch 已保留的内存)
    """
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_npu, _ = torch.npu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_npu + mem_free_torch
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_mlu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cleanup_models() -> None:
    """回收模型占用的内存"""
    cleanup_models_gc()
    logger.info("当前可用的内存: %s MB", get_free_memory() / (1024 * 1024))
