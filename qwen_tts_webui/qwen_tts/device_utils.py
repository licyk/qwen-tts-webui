"""模型计算设备工具

通过 `device_map` 加载模型时, 如果显存不足, Accelerate 会将部分模块卸载到 CPU / 硬盘. 被卸载的模块的参数在模型中只是位于
`meta` 设备上的占位张量, 实际权重在前向计算时才由 Hook 拷贝到计算设备上.

HuggingFace 的 `PreTrainedModel.device` 以及 `weight.device` 返回的是参数所在的设备, 对于被卸载的模块会得到 `meta`,
在其上创建的张量没有数据 (例如 `Tensor.item() cannot be called on meta tensors`). 因此需要通过 Accelerate Hook 获取模块实际的计算设备.
"""

import torch
from torch import nn

SINGLE_DEVICE_MAP_STRATEGIES = ("auto", "balanced", "balanced_low_0", "sequential")
"""会让 Accelerate 自动分配设备 (可能拆分 / 卸载模型) 的 `device_map` 取值"""


def _to_device(
    device: int | str | torch.device,
) -> torch.device:
    """将 Accelerate 中的设备表示转换为 `torch.device`

    Args:
        device (int | str | torch.device):
            设备, 整数表示加速器的序号

    Returns:
        torch.device: 设备
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        for name in ("cuda", "xpu", "npu", "mlu"):
            backend = getattr(torch, name, None)
            if backend is not None and hasattr(backend, "is_available") and backend.is_available():
                return torch.device(name, device)
        return torch.device("cuda", device)
    return torch.device(device)


def _hook_execution_device(
    module: nn.Module,
) -> torch.device | None:
    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return None
    for h in getattr(hook, "hooks", None) or [hook]:
        device = getattr(h, "execution_device", None)
        if device is not None:
            return _to_device(device)
    return None


def execution_device(
    module: nn.Module,
) -> torch.device:
    """获取模块实际进行计算的设备

    依次使用: 模块自身 Accelerate Hook 的计算设备 -> 第一个不在 `meta` 设备上的参数 / Buffer 所在的设备 -> 子模块 Hook 的计算设备 -> CPU.

    Args:
        module (nn.Module):
            模块

    Returns:
        torch.device: 计算设备
    """
    device = _hook_execution_device(module)
    if device is not None:
        return device
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    for submodule in module.modules():
        device = _hook_execution_device(submodule)
        if device is not None:
            return device
    return torch.device("cpu")


def is_single_device_map(
    device_map: object,
) -> bool:
    """判断 `device_map` 是否表示将模型完整放在一个设备上

    Args:
        device_map (object):
            `from_pretrained()` 的 `device_map` 参数

    Returns:
        bool: 为单个设备 (例如 `cuda:0`, `cpu`, `0`) 时返回 True, 为自动分配策略或者设备映射字典时返回 False
    """
    if device_map is None or isinstance(device_map, (int, torch.device)):
        return True
    return isinstance(device_map, str) and device_map not in SINGLE_DEVICE_MAP_STRATEGIES
