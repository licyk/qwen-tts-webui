"""模型内存管理器"""

import gc
from enum import Enum

import torch
import psutil

from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
)

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


class CPUState(Enum):
    """设备状态枚举"""

    GPU = 0
    CPU = 1
    MPS = 2


class DeviceManager:
    """设备管理器，封装所有设备检测和管理逻辑"""

    def __init__(
        self,
    ) -> None:
        """初始化设备管理器并自动检测可用后端"""
        self._xpu_available: bool = self._check_xpu()
        self._npu_available: bool = self._check_npu()
        self._mlu_available: bool = self._check_mlu()
        self._mps_available: bool = self._check_mps()
        self._cpu_state: CPUState = self._init_cpu_state()

    def _check_xpu(
        self,
    ) -> bool:
        """检测 Intel XPU 是否可用"""
        try:
            import torch.xpu  # pylint: disable=redefined-outer-name

            return torch.xpu.is_available()
        except (ImportError, AttributeError, Exception):
            logger.debug("Intel XPU 不可用")
            return False

    def _check_npu(
        self,
    ) -> bool:
        """检测华为昇腾 NPU 是否可用"""
        try:
            import torch_npu  # noqa: F401  # pylint: disable=redefined-outer-name,unused-import # type: ignore

            return torch.npu.is_available()
        except (ImportError, AttributeError, Exception):
            logger.debug("华为昇腾 NPU 不可用")
            return False

    def _check_mlu(
        self,
    ) -> bool:
        """检测寒武纪 MLU 是否可用"""
        try:
            import torch_mlu  # noqa: F401  # pylint: disable=redefined-outer-name,unused-import # type: ignore

            return torch.mlu.is_available()
        except (ImportError, AttributeError, Exception):
            logger.debug("寒武纪 MLU 不可用")
            return False

    def _check_mps(
        self,
    ) -> bool:
        """检测 Apple Silicon MPS 是否可用"""
        try:
            return torch.backends.mps.is_available()
        except (AttributeError, Exception):
            logger.debug("Apple Silicon MPS 不可用")
            return False

    def _init_cpu_state(
        self,
    ) -> CPUState:
        """初始化默认设备状态"""
        if self._mps_available:
            return CPUState.MPS
        return CPUState.GPU

    @property
    def cpu_state(
        self,
    ) -> CPUState:
        """获取当前设备状态"""
        return self._cpu_state

    @cpu_state.setter
    def cpu_state(
        self,
        state: CPUState,
    ) -> None:
        """设置当前设备状态"""
        self._cpu_state = state

    def is_intel_xpu(
        self,
    ) -> bool:
        """判断当前是否正在使用 Intel XPU"""
        return self._cpu_state == CPUState.GPU and self._xpu_available

    def is_ascend_npu(
        self,
    ) -> bool:
        """判断 NPU 是否可用"""
        return self._npu_available

    def is_mlu(
        self,
    ) -> bool:
        """判断 MLU 是否可用"""
        return self._mlu_available

    def get_torch_device(
        self,
    ) -> torch.device:
        """获取当前首选的 PyTorch 设备

        Returns:
            torch.device:
                当前使用的 PyTorch 设备对象
        """
        if self._cpu_state == CPUState.MPS:
            return torch.device("mps")
        if self._cpu_state == CPUState.CPU:
            return torch.device("cpu")

        if self.is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        if self.is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        if self.is_mlu():
            return torch.device("mlu", torch.mlu.current_device())

        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())

        return torch.device("cpu")

    def get_available_devices(
        self,
    ) -> list[torch.device]:
        """获取所有可用的计算设备列表

        Returns:
            list[torch.device]:
                包含所有可用设备的列表
        """
        devices: list[torch.device] = []

        # CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda", i))

        # XPU
        if self._xpu_available:
            try:
                for i in range(torch.xpu.device_count()):
                    devices.append(torch.device("xpu", i))
            except Exception:
                pass

        # NPU
        if self._npu_available:
            try:
                for i in range(torch.npu.device_count()):
                    devices.append(torch.device("npu", i))
            except Exception:
                pass

        # MLU
        if self._mlu_available:
            try:
                for i in range(torch.mlu.device_count()):
                    devices.append(torch.device("mlu", i))
            except Exception:
                pass

        # MPS
        if self._mps_available:
            devices.append(torch.device("mps"))

        # CPU
        devices.append(torch.device("cpu"))
        return devices

    def empty_cache(
        self,
    ) -> None:
        """清理当前设备的显存/内存缓存"""
        import torch  # pylint: disable=redefined-outer-name,import-error,unused-import,reimported  # noqa: F401  # type: ignore

        if self._cpu_state == CPUState.MPS:
            import torch.mps  # pylint: disable=redefined-outer-name

            torch.mps.empty_cache()
        elif self.is_intel_xpu():
            import torch.xpu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

            torch.xpu.empty_cache()
        elif self.is_ascend_npu():
            import torch_npu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

            torch.npu.empty_cache()
        elif self.is_mlu():
            import torch_mlu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

            torch.mlu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def get_free_memory(
        self,
        dev: torch.device | None = None,
        torch_free_too: bool | None = False,
    ) -> int | tuple[int, int]:
        """获取指定设备的空闲内存信息

        Args:
            dev (torch.device | None):
                要查询的设备, 如果为 None 则查询当前默认设备
            torch_free_too (bool | None):
                是否同时返回 PyTorch 已保留但未使用的内存

        Returns:
            (int | tuple[int, int]):
                空闲内存字节数, 或者 (总空闲, PyTorch保留空闲) 元组
        """
        if dev is None:
            dev = self.get_torch_device()

        if dev.type in ["cpu", "mps"]:
            mem_free_total = psutil.virtual_memory().available
            mem_free_torch = mem_free_total
        else:
            try:
                if dev.type == "xpu":
                    import torch.xpu  # pylint: disable=redefined-outer-name

                    stats = torch.xpu.memory_stats(dev)
                    mem_active = stats["active_bytes.all.current"]
                    mem_reserved = stats["reserved_bytes.all.current"]
                    mem_free_backend = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
                elif dev.type == "npu":
                    import torch_npu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

                    stats = torch.npu.memory_stats(dev)
                    mem_active = stats["active_bytes.all.current"]
                    mem_reserved = stats["reserved_bytes.all.current"]
                    mem_free_backend, _ = torch.npu.mem_get_info(dev)
                elif dev.type == "mlu":
                    import torch_mlu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

                    stats = torch.mlu.memory_stats(dev)
                    mem_active = stats["active_bytes.all.current"]
                    mem_reserved = stats["reserved_bytes.all.current"]
                    mem_free_backend, _ = torch.mlu.mem_get_info(dev)
                else:  # cuda
                    import torch  # pylint: disable=redefined-outer-name,import-error,unused-import,reimported  # noqa: F401  # type: ignore

                    stats = torch.cuda.memory_stats(dev)
                    mem_active = stats["active_bytes.all.current"]
                    mem_reserved = stats["reserved_bytes.all.current"]
                    mem_free_backend, _ = torch.cuda.mem_get_info(dev)

                mem_free_torch = mem_reserved - mem_active
                mem_free_total = mem_free_backend + mem_free_torch
            except Exception:
                mem_free_total = psutil.virtual_memory().available
                mem_free_torch = 0

        return (mem_free_total, mem_free_torch) if torch_free_too else mem_free_total


device_manager = DeviceManager()
"""设备管理器"""

try:
    OutOfMemoryError = torch.cuda.OutOfMemoryError
    """内存溢出错误"""
except Exception as _:
    OutOfMemoryError = Exception
    """内存溢出错误"""

MODEL_PRECISION_LIST = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
"""模型精度列表"""


def get_torch_device() -> torch.device:
    """获取当前使用的 PyTorch 设备

    Returns:
        torch.device:
            当前使用的 PyTorch 设备
    """
    return device_manager.get_torch_device()


def get_available_devices() -> list[torch.device]:
    """获取所有可用的计算设备列表

    Returns:
        list[torch.device]:
            所有可用的计算设备列表
    """
    return device_manager.get_available_devices()


def is_intel_xpu() -> bool:
    """判断是否为 Intel XPU

    Returns:
        bool:
            为 Intel XPU 时返回 True
    """
    return device_manager.is_intel_xpu()


def is_ascend_npu() -> bool:
    """判断是否为华为昇腾 NPU

    Returns:
        bool:
            为华为昇腾 NPU 时返回 True
    """
    return device_manager.is_ascend_npu()


def is_mlu() -> bool:
    """判断是否为寒武纪 MLU

    Returns:
        bool:
            为寒武纪 MLU 时返回 True
    """
    return device_manager.is_mlu()


def cleanup_models_gc() -> None:
    """执行垃圾回收并清理显存缓存"""
    gc.collect()
    device_manager.empty_cache()


def get_free_memory(
    dev: torch.device | None = None,
    torch_free_too: bool | None = False,
) -> int | tuple[int, int]:
    """获取空闲内存"""
    return device_manager.get_free_memory(dev, torch_free_too)


def cleanup_models() -> None:
    """回收模型占用的内存并记录日志"""
    cleanup_models_gc()
    logger.info("当前可用的内存: %.2f MB", get_free_memory() / (1024 * 1024))


def get_device_name(
    device: torch.device,
) -> str:
    """获取设备对应的名字

    Args:
        device (torch.device):
            PyTorch 设备对象

    Returns:
        str:
            设备的名字
    """
    try:
        if device.type == "cuda":
            import torch  # pylint: disable=redefined-outer-name,reimported

            return f"({torch.cuda.get_device_name(device)})"
        elif device.type == "xpu":
            import torch.xpu  # pylint: disable=redefined-outer-name

            return f"({torch.xpu.get_device_properties(device).name})"
        elif device.type == "npu":
            import torch_npu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

            return f"({torch.npu.get_device_name(device)})"
        elif device.type == "mlu":
            import torch_mlu  # pylint: disable=redefined-outer-name,import-error,unused-import  # noqa: F401  # type: ignore

            return f"({torch.mlu.get_device_name(device)})"
    except Exception:
        return str(device)
