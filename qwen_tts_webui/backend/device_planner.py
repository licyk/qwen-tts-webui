"""显存不足时的模型设备分配方案

`device_map="auto"` 由 Accelerate 按模块顺序将模型填入显卡, 放不下的部分卸载到 CPU. Qwen3-TTS 中每一帧都会使用的模块
(Code Predictor, Codec Embedding / Head 等) 位于模型的末尾, 因此会被最先卸载, 导致每一帧都需要从内存中多次拷贝权重,
生成速度极慢 (实测实时率约 31).

该模块根据模型各部分的大小和每张显卡的空闲显存生成设备映射:
1. 每一帧都会使用的模块与语音解码器放在空闲显存最多的显卡上.
2. Talker 的 Decoder Layer 按顺序填入该显卡, 然后是其他显卡, 放不下的部分卸载到 CPU.
3. 如果连第 1 步中的模块都放不下, 则完全使用 CPU 进行推理.

文本嵌入 (约 0.58 GB) 只在预填充阶段使用一次, 但 Accelerate 在调用被卸载的模块时会将其权重临时拷贝到显卡上,
卸载它并不能降低显存峰值, 因此与其他常驻模块一起放在显卡上.
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch

from qwen_tts_webui.config_manager.config import (
    LOGGER_COLOR,
    LOGGER_LEVEL,
)
from qwen_tts_webui.logger import get_logger

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)

RUNTIME_RESERVE_BYTES = int(1.6 * 1024**3)
"""主显卡上为生成过程预留的显存 (KV Cache, 中间激活值, 语音解码). 实测批量大小为 4 或者生成 60 秒音频时峰值约为 1.3 ~ 1.6 GB,
更大的批量在显存不足时会自动减小批量大小"""

SECONDARY_RESERVE_BYTES = int(0.5 * 1024**3)
"""其他显卡上预留的显存 (KV Cache 与中间激活值)"""

LAYER_PREFIX = "talker.model.layers."
"""Talker Decoder Layer 的模块名前缀"""

_SAFETENSORS_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


@dataclass
class DevicePlan:
    """设备分配方案"""

    device_map: str | dict[str, int | str]
    """传给 `from_pretrained()` 的 `device_map`"""

    description: str
    """方案说明"""

    offloaded_layers: list[str] = field(default_factory=list)
    """被卸载到 CPU 的模块"""


def _floating_numel(
    safetensors_path: Path,
) -> tuple[int, int]:
    """读取 safetensors 文件头, 统计浮点参数与非浮点参数的字节数 (不读取权重)

    Args:
        safetensors_path (Path):
            safetensors 文件路径

    Returns:
        tuple[int, int]: 浮点参数的元素数量, 非浮点参数的字节数
    """
    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    numel, other_bytes = 0, 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in info["shape"]:
            count *= dim
        if info["dtype"].startswith(("F", "BF")):
            numel += count
        else:
            other_bytes += count * _SAFETENSORS_DTYPE_BYTES.get(info["dtype"], 4)
    return numel, other_bytes


def speech_tokenizer_bytes(
    model_path: Path,
    dtype: torch.dtype,
) -> int:
    """估算语音解码器以指定精度加载后占用的显存

    Args:
        model_path (Path):
            模型路径
        dtype (torch.dtype):
            加载精度

    Returns:
        int: 字节数
    """
    total = 0
    for file in (model_path / "speech_tokenizer").glob("*.safetensors"):
        numel, other_bytes = _floating_numel(file)
        total += numel * torch.finfo(dtype).bits // 8 + other_bytes
    return total


def module_sizes(
    model_path: Path,
    dtype: torch.dtype,
) -> tuple[dict[str, int], list[str], list[str]]:
    """在 meta 设备上构建模型结构 (不加载权重), 计算各个模块以指定精度加载后的大小

    Args:
        model_path (Path):
            模型路径
        dtype (torch.dtype):
            加载精度

    Returns:
        tuple[dict[str, int], list[str], list[str]]: 模块大小 (`""` 为总大小), 常驻模块列表, Decoder Layer 列表
    """
    from accelerate import init_empty_weights
    from accelerate.utils import compute_module_sizes

    from qwen_tts_webui.qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration

    config = Qwen3TTSConfig.from_pretrained(model_path)
    with init_empty_weights():
        skeleton = Qwen3TTSForConditionalGeneration(config)
    sizes = dict(compute_module_sizes(skeleton, dtype=dtype))

    resident = [name for name, _ in skeleton.named_children() if name != "talker"]
    resident += [f"talker.{name}" for name, _ in skeleton.talker.named_children() if name != "model"]
    resident += [f"talker.model.{name}" for name, _ in skeleton.talker.model.named_children() if name != "layers"]
    layers = [f"{LAYER_PREFIX}{i}" for i in range(len(skeleton.talker.model.layers))]
    return sizes, resident, layers


def cpu_supports_fast_bf16() -> bool:
    """判断 CPU 是否支持原生 bfloat16 计算 (AMX / AVX512-BF16)

    不支持时 CPU 上的 bfloat16 / float16 矩阵乘法非常慢 (实测 Ryzen 9 5900X 上 bfloat16 比 float32 慢约 5.7 倍)

    Returns:
        bool: 是否支持
    """
    cpu_ops = getattr(torch._C, "_cpu", None)
    for name in ("_is_amx_tile_supported", "_is_avx512_bf16_supported"):
        check = getattr(cpu_ops, name, None)
        try:
            if check is not None and check():
                return True
        except RuntimeError:
            continue
    return False


def cpu_inference_dtype(
    dtype: torch.dtype | None,
) -> torch.dtype | None:
    """获取在 CPU 上推理时实际使用的精度

    Args:
        dtype (torch.dtype | None):
            设置的精度

    Returns:
        (torch.dtype | None): CPU 不支持原生半精度计算时将 bfloat16 / float16 替换为 float32, 否则保持不变
    """
    if dtype in (torch.bfloat16, torch.float16) and not cpu_supports_fast_bf16():
        return torch.float32
    return dtype


def gpu_free_memory() -> list[tuple[int, int]]:
    """获取所有 CUDA 显卡的空闲显存

    Returns:
        list[tuple[int, int]]: (空闲显存, 显卡序号) 列表, 按空闲显存从大到小排序
    """
    return sorted(((torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())), reverse=True)


def plan_device_map(
    model_path: str | Path,
    dtype: torch.dtype,
    candidates: list[tuple[int, int]],
) -> DevicePlan:
    """根据显卡的空闲显存生成模型的设备分配方案

    Args:
        model_path (str | Path):
            模型路径
        dtype (torch.dtype):
            加载精度
        candidates (list[tuple[int, int]]):
            可用显卡的 (空闲显存, 显卡序号) 列表, 第一项为主显卡

    Returns:
        DevicePlan: 设备分配方案
    """
    model_path = Path(model_path)
    gib = 1024**3
    sizes, resident, layers = module_sizes(model_path, dtype)
    tokenizer = speech_tokenizer_bytes(model_path, dtype)
    total = sizes[""] + tokenizer
    main_free, main_index = candidates[0]

    if total + RUNTIME_RESERVE_BYTES <= main_free:
        return DevicePlan(f"cuda:{main_index}", f"模型完整加载到 cuda:{main_index} (需要约 {total / gib:.2f} GB + 运行预留 {RUNTIME_RESERVE_BYTES / gib:.1f} GB)")

    max_layer = max(sizes[name] for name in layers)
    resident_bytes = sum(sizes.get(name, 0) for name in resident) + tokenizer
    budgets = [main_free - RUNTIME_RESERVE_BYTES - resident_bytes - max_layer] + [free - SECONDARY_RESERVE_BYTES for free, _ in candidates[1:]]
    if budgets[0] < 0:
        return DevicePlan(
            "cpu",
            f"空闲显存 ({main_free / gib:.2f} GB) 不足以放下每帧都需要使用的模块 (约 {(resident_bytes + max_layer + RUNTIME_RESERVE_BYTES) / gib:.2f} GB), 使用 CPU 进行推理",
        )

    device_map: dict[str, int | str] = {name: main_index for name in resident}
    slot = 0
    offloaded = []
    for name in layers:
        while slot < len(budgets) and budgets[slot] < sizes[name]:
            slot += 1
        if slot < len(budgets):
            device_map[name] = candidates[slot][1]
            budgets[slot] -= sizes[name]
        else:
            device_map[name] = "cpu"
            offloaded.append(name)

    used = sorted({str(v) for v in device_map.values()}, key=str)
    description = f"单张显卡放不下完整模型 (需要约 {total / gib:.2f} GB + 运行预留 {RUNTIME_RESERVE_BYTES / gib:.1f} GB, cuda:{main_index} 空闲 {main_free / gib:.2f} GB), 使用设备: {', '.join(used)}"
    if offloaded:
        description += f", 其中 {len(offloaded)}/{len(layers)} 层 Talker 卸载到 CPU, 生成速度会明显降低"
    return DevicePlan(device_map, description, offloaded)
