"""Qwen TTS 推理后端"""

import hashlib
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeAlias,
)

import numpy as np
import soundfile as sf
import torch
from modelscope import snapshot_download

from qwen_tts_webui.backend.device_planner import cpu_inference_dtype, gpu_free_memory, plan_device_map
from qwen_tts_webui.backend.errors import GenerationInterruptedError
from qwen_tts_webui.backend.memory_manager import (
    OutOfMemoryError,
    cleanup_models,
    get_free_memory,
    get_torch_device,
    is_out_of_memory_error,
)
from qwen_tts_webui.config_manager.config import (
    LOGGER_COLOR,
    LOGGER_LEVEL,
    OUTPUT_PATH,
)
from qwen_tts_webui.hub import HubManager
from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from qwen_tts_webui.qwen_tts.fast_generation import GenerationStoppedError
from qwen_tts_webui.utils import generate_filename

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)

AttnImpl: TypeAlias = Literal["eager", "sdpa", "flash_attention_2"]
"""注意力加速方案"""

AudioInput: TypeAlias = Path | str | tuple[np.ndarray, int]
"""参考音频输入, 可以为音频文件路径或者 (音频数据, 采样率)"""

VOICE_CLONE_PROMPT_CACHE_SIZE = 16
"""声音克隆提示缓存数量上限"""

StreamCallback: TypeAlias = Callable[[np.ndarray, int], None]
"""流式输出回调, 参数为音频片段和采样率"""


class StreamingAudioDecoder:
    """将逐帧生成的音频码本增量解码为音频片段

    语音解码器为因果结构, 与上游的分块解码 (`chunked_decode`) 相同, 每次解码时带上之前的帧作为上下文, 再丢弃上下文对应的音频.
    上下文长度默认与解码器 Transformer 的滑动窗口 (72 帧) 一致, 与整段解码相比信噪比约为 35dB (上游使用的 25 帧约为 21dB).
    """

    def __init__(
        self,
        model: Qwen3TTSModel,
        emit: StreamCallback,
        prefix_codes: torch.Tensor | None = None,
        first_chunk_frames: int = 12,
        chunk_frames: int = 24,
        left_context_frames: int = 72,
    ) -> None:
        """初始化增量解码器

        Args:
            model (Qwen3TTSModel):
                已加载的模型
            emit (StreamCallback):
                输出音频片段的回调函数
            prefix_codes (torch.Tensor | None):
                位于生成内容之前的码本 (声音克隆 ICL 模式下的参考音频码本), 仅作为解码上下文
            first_chunk_frames (int):
                第一个音频片段的帧数, 越小首包延迟越低
            chunk_frames (int):
                后续每个音频片段的帧数
            left_context_frames (int):
                解码时使用的上下文帧数
        """
        tokenizer = model.model.speech_tokenizer
        self.tokenizer = tokenizer
        self.emit = emit
        self.upsample = int(tokenizer.get_decode_upsample_rate())
        self.sample_rate = int(tokenizer.get_output_sample_rate())
        self.eos_token_id = model.model.config.talker_config.codec_eos_token_id
        self.left_context_frames = left_context_frames
        self.prefix = prefix_codes[-left_context_frames:] if prefix_codes is not None and len(prefix_codes) > 0 else None
        self.first_chunk_frames = first_chunk_frames
        self.chunk_frames = chunk_frames
        self.emitted = 0
        self.finished = False

    def __call__(
        self,
        codes: torch.Tensor,
        final: bool,
    ) -> None:
        """接收已生成的码本, 积累到足够帧数时解码并输出音频

        Args:
            codes (torch.Tensor):
                目前已生成的码本, 形状为 `[1, T, num_code_groups]`
            final (bool):
                是否为最后一次调用
        """
        if self.finished:
            return
        available = codes.shape[1]
        threshold = self.first_chunk_frames if self.emitted == 0 else self.chunk_frames
        if not final and available - self.emitted < threshold:
            return

        frames = codes[0]
        is_eos = frames[:, 0] == self.eos_token_id
        valid = int(is_eos.int().argmax()) if bool(is_eos.any()) else available
        if valid > self.emitted:
            if self.emitted == 0:
                context = self.prefix.to(frames.device) if self.prefix is not None else frames[:0]
            else:
                context = frames[max(0, self.emitted - self.left_context_frames) : self.emitted]
            chunk = torch.cat([context, frames[self.emitted : valid]], dim=0)
            wavs, _ = self.tokenizer.decode([{"audio_codes": chunk}])
            self.emit(wavs[0][context.shape[0] * self.upsample :], self.sample_rate)
            self.emitted = valid
        self.finished = final


@dataclass
class TTSResult:
    """音频生成结果"""

    wavs: list[np.ndarray]
    """生成的音频数据"""

    sample_rate: int
    """音频采样率"""

    paths: list[Path] = field(default_factory=list)
    """音频文件保存路径, 未保存时为空"""

    elapsed: float = 0.0
    """生成耗时 (秒)"""

    @property
    def audio_seconds(self) -> float:
        """生成的音频总时长 (秒)"""
        return sum(len(w) for w in self.wavs) / self.sample_rate if self.sample_rate else 0.0


class QwenTTSBackend:
    """Qwen TTS 推理后端"""

    def __init__(
        self,
    ) -> None:
        """Qwen TTS 后端初始化"""
        self.model_name = None
        self.model: Qwen3TTSModel | None = None
        self.load_signature: tuple[Any, ...] | None = None
        self.cuda_graph_disabled = False
        """当前模型在显存不足时已关闭 CUDA Graph, 重新加载模型前不再启用"""
        self.voice_clone_prompts: OrderedDict[tuple[Any, ...], list[VoiceClonePromptItem]] = OrderedDict()
        self.hub = HubManager(
            hf_token=os.getenv("HF_TOKEN"),
            ms_token=os.getenv("MODELSCOPE_API_TOKEN"),
        )

    def unload_model(
        self,
    ) -> None:
        """卸载模型"""
        logger.info("卸载 %s 模型", self.model_name)
        if self.model is not None:
            self.model.release_fast_generation()
        self.voice_clone_prompts.clear()
        self.model = None
        self.model_name = None
        self.load_signature = None
        self.cuda_graph_disabled = False
        # cuBLAS 的工作区由 PyTorch 显存分配器在首次矩阵乘法时分配, 会长期占用并导致其所在的大块显存 (模型权重释放后留下的) 无法被回收
        if torch.cuda.is_available() and hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()  # pylint: disable=protected-access
        cleanup_models()
        logger.info("卸载模型完成")

    def _download_model(
        self,
        model_name: str,
        api_type: Literal["huggingface", "modelscope", "local"] | None,
        force: bool = False,
    ) -> str:
        """获取模型的本地路径, 本地不存在时从网络下载

        Args:
            model_name (str):
                模型名称或本地路径
            api_type (Literal["huggingface", "modelscope", "local"] | None):
                下载模型时使用的 API 类型
            force (bool):
                跳过本地缓存直接从网络下载 (用于修复损坏的模型文件)

        Returns:
            str: 模型的本地路径

        Raises:
            ValueError: 使用错误的 API 类型时
            FileNotFoundError: 本地路径不存在时
        """
        if api_type == "local":
            if Path(model_name).exists():
                logger.info("从本地路径加载模型: %s", model_name)
                return model_name
            raise FileNotFoundError(f"模型路径 '{model_name}' 不存在\n提示: 如果使用的不是本地路径的模型, 请将 API 类型切换到 modelscope 或者是 huggingface")

        if api_type == "huggingface":

            def download(local_files_only: bool) -> str:
                return self.hub.hf_api.snapshot_download(repo_id=model_name, repo_type="model", local_files_only=local_files_only)

            source = "HuggingFace"
        elif api_type == "modelscope":

            def download(local_files_only: bool) -> str:
                return snapshot_download(repo_id=model_name, repo_type="model", local_files_only=local_files_only)

            source = "ModelScope"
        else:
            raise ValueError(f"未知的 API 类型: {api_type}")

        if not force:
            try:
                # 优先使用本地缓存
                model_path = download(local_files_only=True)
                logger.info("从本地缓存加载 %s", model_name)
                return model_path
            except Exception:
                logger.info("本地未找到模型, 从 %s 下载 %s 中", source, model_name)
        else:
            logger.info("从 %s 重新下载 %s 中", source, model_name)
        model_path = download(local_files_only=False)
        logger.info("%s 模型从 %s 下载完成", model_name, source)
        return model_path

    def _resolve_device_map(
        self,
        device_map: str | None,
        model_path: str,
        dtype: torch.dtype | None,
    ) -> str | dict[str, int | str] | None:
        """解析推理设备

        - `auto`: 单张显卡能放下模型时, 使用空闲显存最多的显卡 (跨设备拆分只会增加开销); 放不下时, 将每帧都会使用的模块
          放在该显卡上, 其余 Decoder Layer 依次放到其他显卡或者卸载到 CPU (Accelerate 默认的分配方式会最先卸载这些模块, 生成速度极慢).
        - 指定 CUDA 显卡时, 显存不足则按同样的方式将部分 Decoder Layer 卸载到 CPU, 而不是直接加载失败.

        Args:
            device_map (str | None):
                用户设置的推理设备
            model_path (str):
                模型的本地路径
            dtype (torch.dtype | None):
                加载精度

        Returns:
            (str | dict[str, int | str] | None): 实际使用的推理设备或者设备映射
        """
        explicit_cuda = isinstance(device_map, str) and device_map.startswith("cuda") and torch.cuda.is_available()
        if device_map not in (None, "auto") and not explicit_cuda:
            return device_map

        if explicit_cuda:
            index = torch.device(device_map).index
            index = torch.cuda.current_device() if index is None else index
            candidates = [(torch.cuda.mem_get_info(index)[0], index)]
        else:
            device = get_torch_device()
            if device.type == "cpu":
                return device_map
            if device.type != "cuda":
                logger.info("推理设备自动选择: %s", device)
                return str(device)
            candidates = gpu_free_memory()

        try:
            plan = plan_device_map(model_path, dtype or torch.bfloat16, candidates)
        except Exception as e:
            logger.warning("生成设备分配方案失败, 使用 %s 加载模型: %s", device_map, e)
            return device_map
        if isinstance(plan.device_map, dict) or plan.device_map == "cpu":
            logger.warning("%s", plan.description)
        else:
            logger.info("推理设备自动选择: %s", plan.description)
        return plan.device_map

    def load_model(
        self,
        model_name: str | None = None,
        device_map: str | None = "auto",
        dtype: torch.dtype | None = torch.bfloat16,
        attn_implementation: AttnImpl | None = None,
        api_type: Literal["huggingface", "modelscope", "local"] | None = "modelscope",
        fast_generation: bool = True,
        cuda_graph: bool = True,
    ) -> None:
        """加载 Qwen TTS 模型

        Args:
            model_name (str | None):
                加载的 Qwen TTS 模型名称或本地路径
            device_map (str | None):
                加载模型使用的设备
            dtype (torch.dtype | None):
                加载模型使用的精度
            attn_implementation (AttnImpl | None):
                加载模型时使用的加速方案
            api_type (Literal["huggingface", "modelscope", "local"] | None):
                下载 Qwen TTS 模型时使用的 API 类型, 设置为 "local" 时直接从本地路径加载
            fast_generation (bool):
                是否使用快速解码 (静态 KV Cache + GPU 采样 + 融合算子)
            cuda_graph (bool):
                快速解码时是否使用 CUDA Graph (仅 CUDA 设备有效)

        Raises:
            ValueError:
                当未设置任何模型名称时或者使用错误的 API 类型下载模型时
            OutOfMemoryError:
                内存不足以加载模型时
            RuntimeError:
                加载模型发生未知错误时
            FileNotFoundError:
                模型路径不存在时
        """
        if self.model_name is None and model_name is None:
            raise ValueError("未指定 Qwen TTS 模型名称")
        model_name = model_name or self.model_name
        signature = (model_name, device_map, dtype, attn_implementation, api_type)

        if self.model is not None and self.load_signature == signature:
            logger.info("Qwen TTS 模型 %s 已加载", self.model_name)
            self._apply_fast_generation(fast_generation, cuda_graph)
            return

        if self.model is not None:
            if self.model_name != model_name:
                logger.info("切换 Qwen TTS 模型: %s -> %s", self.model_name, model_name)
            else:
                logger.info("模型加载设置已修改, 重新加载 Qwen TTS 模型 %s", model_name)
            self.unload_model()

        self.model_name = model_name
        logger.info("加载 Qwen TTS 模型: %s", self.model_name)
        model_path = self._download_model(model_name, api_type)

        def load(path: str) -> Qwen3TTSModel:
            resolved = self._resolve_device_map(device_map, path, dtype)
            load_dtype = dtype
            if resolved == "cpu" and cpu_inference_dtype(dtype) != dtype:
                load_dtype = cpu_inference_dtype(dtype)
                logger.info("CPU 不支持原生 %s 计算, 使用 %s 精度在 CPU 上进行推理 (速度更快)", dtype, load_dtype)
            return Qwen3TTSModel.from_pretrained(
                pretrained_model_name_or_path=path,
                device_map=resolved,
                dtype=load_dtype,
                attn_implementation=attn_implementation,
                fast_generation=fast_generation,
                use_cuda_graph=cuda_graph,
            )

        try:
            self.model = load(model_path)
        except OSError as e:  # pylint: disable=bad-except-order
            logger.warning("加载 Qwen TTS 模型 %s 时因模型文件损坏发生错误: %s", self.model_name, e)
            tmp_name = self.model_name
            if api_type == "local":
                # 本地路径模式无法重新下载
                self.unload_model()
                raise OSError(f"加载 Qwen TTS 模型 {tmp_name} 时因模型文件损坏发生错误: {e}\n\n提示: 本地路径模式无法自动重新下载, 请手动检查模型文件 '{model_path}'") from e

            logger.info("尝试重新下载模型 %s", self.model_name)
            try:
                model_path = self._download_model(model_name, api_type, force=True)
                logger.info("重新加载模型 %s", self.model_name)
                self.model = load(model_path)
                logger.info("%s 模型重新加载成功", self.model_name)
            except OSError as retry_e:
                logger.error("重新下载并加载模型 %s 失败: %s", self.model_name, retry_e)
                self.unload_model()
                raise OSError(f"加载 Qwen TTS 模型 {tmp_name} 时因模型文件损坏发生错误, 重新下载后仍然失败: {retry_e}\n\n可尝试手动将 '{model_path}' 删除后再试") from retry_e
            except Exception as retry_e:
                logger.error("重新下载模型 %s 时发生错误: %s", self.model_name, retry_e)
                self.unload_model()
                raise RuntimeError(f"重新下载 Qwen TTS 模型 {tmp_name} 时发生错误: {retry_e}") from retry_e
        except Exception as e:  # pylint: disable=duplicate-except
            tmp_name = self.model_name
            if is_out_of_memory_error(e):
                logger.error("加载 Qwen TTS 模型 %s 时发生内存不足: %s", self.model_name, e)
                self.unload_model()
                raise OutOfMemoryError(f"加载 Qwen TTS 模型 {tmp_name} 时发生内存不足: {e}") from e
            logger.error("加载 Qwen TTS 模型 %s 时发生未知错误: %s", self.model_name, e)
            try:
                self.unload_model()
            except Exception:
                logger.warning("卸载模型发生错误: %s", e)
            raise RuntimeError(f"加载 Qwen TTS 模型 {tmp_name} 时发生未知错误: {e}") from e

        self.load_signature = signature
        logger.info("%s 模型加载完成, 当前剩余的显存: %.2f MB", self.model_name, get_free_memory() / (1024 * 1024))

    def _apply_fast_generation(
        self,
        fast_generation: bool,
        cuda_graph: bool,
    ) -> None:
        """在不重新加载模型的情况下切换快速解码设置

        Args:
            fast_generation (bool):
                是否使用快速解码
            cuda_graph (bool):
                是否使用 CUDA Graph
        """
        cuda_graph = cuda_graph and not self.cuda_graph_disabled
        generator = getattr(self.model.model, "fast_generator", None)
        current = (generator is not None, generator is not None and generator.use_cuda_graph)
        wanted_graph = cuda_graph and (generator.single_device and generator.main_device.type == "cuda" if generator is not None else True)
        if current[0] == fast_generation and (not fast_generation or current[1] == wanted_graph):
            return
        logger.info("切换快速解码设置: fast_generation=%s, cuda_graph=%s", fast_generation, cuda_graph)
        self.model.set_fast_generation(fast_generation, use_cuda_graph=cuda_graph)

    def _disable_cuda_graph(self) -> bool:
        """显存不足时关闭当前模型的 CUDA Graph (CUDA Graph 需要额外的显存池)

        Returns:
            bool: 是否关闭了 CUDA Graph (未使用 CUDA Graph 时返回 False)
        """
        generator = getattr(self.model.model, "fast_generator", None) if self.model is not None else None
        if generator is None or not generator.use_cuda_graph:
            return False
        self.cuda_graph_disabled = True
        self.model.set_fast_generation(True, use_cuda_graph=False)
        cleanup_models()
        return True

    def get_supported_speakers(
        self,
    ) -> list[str] | None:
        """获取 Qwen TTS 支持的说话者列表

        Returns:
            (list[str] | None): 说话者列表, 如果模型不支持指定任何说话者或者模型未加载时则返回 None
        """
        if self.model is None:
            return None
        return self.model.get_supported_speakers()

    def get_supported_languages(
        self,
    ) -> list[str] | None:
        """获取 Qwen TTS 支持的语言列表

        Returns:
            (list[str] | None): 语言列表, 如果模型不支持指定任何语言或者模型未加载时则返回 None
        """
        if self.model is None:
            return None
        return self.model.get_supported_languages()

    def get_model_type(
        self,
    ) -> str | None:
        """获取当前模型的类型

        Returns:
            (str | None): `custom_voice` / `voice_design` / `base`, 模型未加载时返回 None
        """
        if self.model is None:
            return None
        return self.model.model.tts_model_type

    def count_tokens(
        self,
        text: str,
    ) -> int:
        """计算文本的 token 数量

        Args:
            text (str):
                要计算 token 数量的文本

        Returns:
            int: token 数量

        Raises:
            ValueError:
                当模型未加载时
        """
        if self.model is None:
            raise ValueError("模型未加载, 无法计算 token 数量")

        # 使用 processor 进行 tokenize
        inputs = self.model.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]

        # 返回 token 数量
        token_count = input_ids.shape[-1]
        logger.debug("文本 '%s' 的 token 数量: %d", text[:50] + "..." if len(text) > 50 else text, token_count)
        return token_count

    def _run_batched(
        self,
        texts: list[str],
        call: Callable[[list[str], dict[str, Any]], tuple[list[np.ndarray], int]],
        batch_size: int,
        should_stop: Callable[[], bool] | None,
        stream_callback: StreamCallback | None = None,
        prefix_codes: torch.Tensor | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """按批次生成音频, 显存不足时自动减小批次大小重试

        Args:
            texts (list[str]):
                待合成的文本列表
            call (Callable[[list[str], dict[str, Any]], tuple[list[np.ndarray], int]]):
                实际调用模型的函数, 参数为当前批次的文本和额外的生成参数
            batch_size (int):
                批次大小
            should_stop (Callable[[], bool] | None):
                返回 True 时中断生成
            stream_callback (StreamCallback | None):
                流式输出回调, 设置后按顺序逐段生成, 并在生成过程中增量输出音频片段
            prefix_codes (torch.Tensor | None):
                流式解码时使用的前置码本 (声音克隆 ICL 模式下的参考音频码本)

        Returns:
            tuple[list[np.ndarray], int]: 音频数据列表与采样率

        Raises:
            GenerationInterruptedError: 生成被中断时
            OutOfMemoryError: 批次大小为 1 时仍然显存不足
        """
        wavs: list[np.ndarray] = []
        sample_rate = 0
        batch_size = 1 if stream_callback is not None else max(1, int(batch_size))
        index = 0
        while index < len(texts):
            if should_stop is not None and should_stop():
                raise GenerationInterruptedError("任务已中断")
            chunk = texts[index : index + batch_size]
            if len(texts) > 1:
                logger.info("生成音频中 (%d-%d / %d)", index + 1, index + len(chunk), len(texts))
            decoder = None
            extra: dict[str, Any] = {}
            if stream_callback is not None:
                decoder = StreamingAudioDecoder(self.model, stream_callback, prefix_codes=prefix_codes)
                extra["on_frames"] = decoder
            try:
                chunk_wavs, sample_rate = call(chunk, extra)
            except GenerationStoppedError as e:
                raise GenerationInterruptedError("任务已中断") from e
            except Exception as e:
                if not is_out_of_memory_error(e):
                    raise
                self.model.release_fast_generation()
                cleanup_models()
                # 流式输出时已经输出的音频无法撤回, 不能重试
                retry = decoder is None or decoder.emitted == 0
                if retry and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.warning("生成音频时显存不足, 将批量大小减小为 %d 后重试", batch_size)
                    continue
                if retry and self._disable_cuda_graph():
                    logger.warning("生成音频时显存不足, 关闭 CUDA Graph 后重试")
                    continue
                logger.error("调用 Qwen TTS 模型 %s 进行音频合成时发生内存不足: %s", self.model_name, e)
                self.unload_model()
                if isinstance(e, OutOfMemoryError) and OutOfMemoryError is not Exception:
                    raise
                raise OutOfMemoryError(f"生成音频时显存不足: {e}") from e
            if should_stop is not None and should_stop():
                raise GenerationInterruptedError("任务已中断")
            if decoder is not None and not decoder.finished:
                # 未使用快速解码时无法增量输出, 生成完成后一次性输出
                stream_callback(chunk_wavs[0], sample_rate)
            wavs.extend(chunk_wavs)
            index += len(chunk)
        return wavs, sample_rate

    def _finish(
        self,
        wavs: list[np.ndarray],
        sample_rate: int,
        start_time: float,
        save: bool,
    ) -> TTSResult:
        result = TTSResult(wavs=wavs, sample_rate=sample_rate, elapsed=time.perf_counter() - start_time)
        if save:
            result.paths = [self.save_audio(w, sample_rate) for w in wavs]
        rtf = result.elapsed / result.audio_seconds if result.audio_seconds > 0 else 0.0
        logger.info(
            "音频生成完成, 共 %d 段, 音频时长: %.2fs, 耗时: %.2fs, 实时率 (RTF): %.3f",
            len(wavs),
            result.audio_seconds,
            result.elapsed,
            rtf,
        )
        generator = getattr(self.model.model, "fast_generator", None) if self.model is not None else None
        if generator is not None:
            stats = generator.last_stats
            logger.debug(
                "快速解码统计: batch=%d, steps=%d, prefill=%.1fms, decode=%.2fs (%.1fms/帧), cuda_graph=%s, graph_captures=%d (%.2fs)",
                stats.batch_size,
                stats.steps,
                stats.prefill_seconds * 1000,
                stats.decode_seconds,
                stats.decode_seconds * 1000 / max(stats.steps, 1),
                stats.used_cuda_graph,
                stats.graph_captures,
                stats.capture_seconds,
            )
        return result

    @staticmethod
    def _sampling_kwargs(
        do_sample: bool | None,
        top_k: int | None,
        top_p: float | None,
        temperature: float | None,
        repetition_penalty: float | None,
        subtalker_dosample: bool | None,
        subtalker_top_k: int | None,
        subtalker_top_p: float | None,
        subtalker_temperature: float | None,
        max_new_tokens: int | None,
        should_stop: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        return {
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "max_new_tokens": max_new_tokens,
            "should_stop": should_stop,
        }

    def _check_loaded(self) -> Qwen3TTSModel:
        if self.model is None:
            raise ValueError("Qwen TTS 模型未加载")
        return self.model

    def generate_custom_voice(
        self,
        text: str | list[str],
        speaker: str,
        language: str | None = None,
        instruct: str | None = None,
        do_sample: bool | None = True,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        temperature: float | None = 0.9,
        repetition_penalty: float | None = 1.05,
        subtalker_dosample: bool | None = True,
        subtalker_top_k: int | None = 50,
        subtalker_top_p: float | None = 1.0,
        subtalker_temperature: float | None = 0.9,
        max_new_tokens: int | None = 2048,
        batch_size: int = 1,
        should_stop: Callable[[], bool] | None = None,
        stream_callback: StreamCallback | None = None,
        save: bool = True,
    ) -> TTSResult:
        """使用提示词生成音频

        Args:
            text (str | list[str]):
                要合成的文本, 传入列表时按批次生成多段音频
            speaker (str):
                说话人的名字
            language (str | None):
                合成文本使用的语言, 为 None 时自动根据要合成的文本语言进行配置
            instruct (str | None):
                描述合成的音频的特征
            do_sample (bool | None):
                是否使用采样, 建议设置为 `True` 以适用于大多数用例
            top_k (int | None):
                Top-k 采样参数
            top_p (float | None):
                Top-p 采样参数
            temperature (float | None):
                采样温度, 越高越随机
            repetition_penalty (float | None):
                减少重复标记 / 代码的惩罚系数
            subtalker_dosample (bool | None):
                子说话者的采样开关 (仅对 qwen3-tts-tokenizer-v2 有效) (如适用)
            subtalker_top_k (int | None):
                子说话者 Top-k 采样 (仅对 qwen3-tts-tokenizer-v2 有效)
            subtalker_top_p (float | None):
                子说话者 Top-p 采样 (仅对 qwen3-tts-tokenizer-v2 有效)
            subtalker_temperature (float | None):
                子说话者采样温度 (仅对 qwen3-tts-tokenizer-v2 有效)
            max_new_tokens (int | None):
                要生成的最大新编解码标记数
            batch_size (int):
                同时生成的文本段数
            should_stop (Callable[[], bool] | None):
                返回 True 时中断生成
            stream_callback (StreamCallback | None):
                流式输出回调, 设置后按顺序逐段生成, 并在生成过程中增量输出音频片段
            save (bool):
                是否将音频保存到输出文件夹

        Returns:
            TTSResult: 生成结果

        Raises:
            OutOfMemoryError: 推理时发生内存不足
            GenerationInterruptedError: 生成被中断时
        """
        model = self._check_loaded()
        texts = [text] if isinstance(text, str) else list(text)
        kwargs = self._sampling_kwargs(
            do_sample,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            subtalker_dosample,
            subtalker_top_k,
            subtalker_top_p,
            subtalker_temperature,
            max_new_tokens,
            should_stop,
        )
        logger.debug("调用 QwenTTSBackend.generate_custom_voice() 的参数: text=%s, speaker=%s, language=%s, instruct=%s, %s", texts, speaker, language, instruct, kwargs)

        def call(chunk: list[str], extra: dict[str, Any]) -> tuple[list[np.ndarray], int]:
            return model.generate_custom_voice(text=chunk, speaker=speaker, language=language, instruct=instruct, **kwargs, **extra)

        logger.info("生成音频中")
        start_time = time.perf_counter()
        wavs, sample_rate = self._run_batched(texts, call, batch_size, should_stop, stream_callback)
        return self._finish(wavs, sample_rate, start_time, save)

    def generate_voice_design(
        self,
        text: str | list[str],
        instruct: str,
        language: str | None = None,
        do_sample: bool | None = True,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        temperature: float | None = 0.9,
        repetition_penalty: float | None = 1.05,
        subtalker_dosample: bool | None = True,
        subtalker_top_k: int | None = 50,
        subtalker_top_p: float | None = 1.0,
        subtalker_temperature: float | None = 0.9,
        max_new_tokens: int | None = 2048,
        batch_size: int = 1,
        should_stop: Callable[[], bool] | None = None,
        stream_callback: StreamCallback | None = None,
        save: bool = True,
    ) -> TTSResult:
        """使用提示词生成音频, 并且使用提示词描述声音

        Args:
            text (str | list[str]):
                要合成的文本, 传入列表时按批次生成多段音频
            instruct (str):
                描述合成的音频的特征
            language (str | None):
                合成文本使用的语言, 为 None 时自动根据要合成的文本语言进行配置
            do_sample (bool | None):
                是否使用采样, 建议设置为 `True` 以适用于大多数用例
            top_k (int | None):
                Top-k 采样参数
            top_p (float | None):
                Top-p 采样参数
            temperature (float | None):
                采样温度, 越高越随机
            repetition_penalty (float | None):
                减少重复标记 / 代码的惩罚系数
            subtalker_dosample (bool | None):
                子说话者的采样开关 (仅对 qwen3-tts-tokenizer-v2 有效) (如适用)
            subtalker_top_k (int | None):
                子说话者 Top-k 采样 (仅对 qwen3-tts-tokenizer-v2 有效)
            subtalker_top_p (float | None):
                子说话者 Top-p 采样 (仅对 qwen3-tts-tokenizer-v2 有效)
            subtalker_temperature (float | None):
                子说话者采样温度 (仅对 qwen3-tts-tokenizer-v2 有效)
            max_new_tokens (int | None):
                要生成的最大新编解码标记数
            batch_size (int):
                同时生成的文本段数
            should_stop (Callable[[], bool] | None):
                返回 True 时中断生成
            stream_callback (StreamCallback | None):
                流式输出回调, 设置后按顺序逐段生成, 并在生成过程中增量输出音频片段
            save (bool):
                是否将音频保存到输出文件夹

        Returns:
            TTSResult: 生成结果

        Raises:
            OutOfMemoryError: 推理时发生内存不足
            GenerationInterruptedError: 生成被中断时
        """
        model = self._check_loaded()
        texts = [text] if isinstance(text, str) else list(text)
        kwargs = self._sampling_kwargs(
            do_sample,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            subtalker_dosample,
            subtalker_top_k,
            subtalker_top_p,
            subtalker_temperature,
            max_new_tokens,
            should_stop,
        )
        logger.debug("调用 QwenTTSBackend.generate_voice_design() 的参数: text=%s, instruct=%s, language=%s, %s", texts, instruct, language, kwargs)

        def call(chunk: list[str], extra: dict[str, Any]) -> tuple[list[np.ndarray], int]:
            return model.generate_voice_design(text=chunk, instruct=instruct, language=language, **kwargs, **extra)

        logger.info("生成音频中")
        start_time = time.perf_counter()
        wavs, sample_rate = self._run_batched(texts, call, batch_size, should_stop, stream_callback)
        return self._finish(wavs, sample_rate, start_time, save)

    def get_voice_clone_prompt(
        self,
        ref_audio: AudioInput,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
    ) -> list[VoiceClonePromptItem]:
        """提取参考音频的声音克隆提示, 结果会被缓存, 同一段参考音频不需要重复提取

        Args:
            ref_audio (AudioInput):
                参考音频文件路径或者 (音频数据, 采样率)
            ref_text (str | None):
                参考音频对应的文本
            x_vector_only_mode (bool):
                是否只使用说话人特征

        Returns:
            list[VoiceClonePromptItem]: 声音克隆提示

        Raises:
            ValueError: 当 `x_vector_only_mode=False` 并且 `ref_text` 为空时
        """
        model = self._check_loaded()
        if not x_vector_only_mode and (ref_text is None or ref_text.strip() == ""):
            raise ValueError("当 `x_vector_only_mode=False` 时, 需要提供参考音频对应的参考文本, 而当前的参考文本 `ref_text` 为空")

        digest = hashlib.sha256()
        if isinstance(ref_audio, tuple):
            wav, sample_rate = ref_audio
            digest.update(np.ascontiguousarray(wav, dtype=np.float32).tobytes())
            digest.update(str(int(sample_rate)).encode())
            audio_arg: Any = (np.asarray(wav, dtype=np.float32), int(sample_rate))
        else:
            digest.update(Path(ref_audio).read_bytes())
            audio_arg = str(ref_audio)
        key = (digest.hexdigest(), "" if x_vector_only_mode else ref_text, bool(x_vector_only_mode))

        items = self.voice_clone_prompts.get(key)
        if items is not None:
            self.voice_clone_prompts.move_to_end(key)
            logger.debug("使用已缓存的声音克隆提示")
            return items

        start_time = time.perf_counter()
        items = model.create_voice_clone_prompt(ref_audio=audio_arg, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
        logger.info("提取参考音频特征完成, 耗时: %.2fs", time.perf_counter() - start_time)
        self.voice_clone_prompts[key] = items
        while len(self.voice_clone_prompts) > VOICE_CLONE_PROMPT_CACHE_SIZE:
            self.voice_clone_prompts.popitem(last=False)
        return items

    def generate_voice_clone(
        self,
        text: str | list[str],
        language: str | None = None,
        ref_audio: AudioInput | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool | None = False,
        do_sample: bool | None = True,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        temperature: float | None = 0.9,
        repetition_penalty: float | None = 1.05,
        subtalker_dosample: bool | None = True,
        subtalker_top_k: int | None = 50,
        subtalker_top_p: float | None = 1.0,
        subtalker_temperature: float | None = 0.9,
        max_new_tokens: int | None = 2048,
        batch_size: int = 1,
        should_stop: Callable[[], bool] | None = None,
        stream_callback: StreamCallback | None = None,
        save: bool = True,
    ) -> TTSResult:
        """使用提示词生成音频, 并利用一段音频克隆声音

        Args:
            text (str | list[str]):
                要合成的文本, 传入列表时按批次生成多段音频
            language (str | None):
                合成文本使用的语言, 为 None 时自动根据要合成的文本语言进行配置
            ref_audio (AudioInput | None):
                用于克隆声音的参考音频文件路径或者 (音频数据, 采样率)
            ref_text (str | None):
                在 ICL 模式下使用的参考文本 (当 `x_vector_only_mode=False` 时必须设置)
            x_vector_only_mode (bool | None):
                - 如果为 True, 则只使用 `ref_audio` 提供的提示 (忽略 `ref_text`)
                - 如果为 False, 则自动使用 ICL 模式
            do_sample (bool | None):
                是否使用采样, 建议设置为 `True` 以适用于大多数用例
            top_k (int | None):
                Top-k 采样参数
            top_p (float | None):
                Top-p 采样参数
            temperature (float | None):
                采样温度, 越高越随机
            repetition_penalty (float | None):
                减少重复标记 / 代码的惩罚系数
            subtalker_dosample (bool | None):
                子说话者的采样开关 (仅对 qwen3-tts-tokenizer-v2 有效) (如适用)
            subtalker_top_k (int | None):
                子说话者 Top-k 采样 (仅对 qwen3-tts-tokenizer-v2 有效)
            subtalker_top_p (float | None):
                子说话者 Top-p 采样 (仅对 qwen3-tts-tokenizer-v2 有效)
            subtalker_temperature (float | None):
                子说话者采样温度 (仅对 qwen3-tts-tokenizer-v2 有效)
            max_new_tokens (int | None):
                要生成的最大新编解码标记数
            batch_size (int):
                同时生成的文本段数
            should_stop (Callable[[], bool] | None):
                返回 True 时中断生成
            stream_callback (StreamCallback | None):
                流式输出回调, 设置后按顺序逐段生成, 并在生成过程中增量输出音频片段
            save (bool):
                是否将音频保存到输出文件夹

        Returns:
            TTSResult: 生成结果

        Raises:
            OutOfMemoryError: 推理时发生内存不足
            ValueError: 当 `x_vector_only_mode=False` 并且 `ref_text` 为空时, 或者未提供参考音频时
            GenerationInterruptedError: 生成被中断时
        """
        model = self._check_loaded()
        if ref_audio is None:
            raise ValueError("未提供用于克隆声音的参考音频")
        texts = [text] if isinstance(text, str) else list(text)
        kwargs = self._sampling_kwargs(
            do_sample,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            subtalker_dosample,
            subtalker_top_k,
            subtalker_top_p,
            subtalker_temperature,
            max_new_tokens,
            should_stop,
        )
        logger.debug(
            "调用 QwenTTSBackend.generate_voice_clone() 的参数: text=%s, language=%s, ref_text=%s, x_vector_only_mode=%s, %s",
            texts,
            language,
            ref_text,
            x_vector_only_mode,
            kwargs,
        )

        logger.info("生成音频中")
        start_time = time.perf_counter()
        prompt = self.get_voice_clone_prompt(ref_audio, ref_text, bool(x_vector_only_mode))

        def call(chunk: list[str], extra: dict[str, Any]) -> tuple[list[np.ndarray], int]:
            return model.generate_voice_clone(text=chunk, language=language, voice_clone_prompt=prompt, **kwargs, **extra)

        prefix_codes = prompt[0].ref_code if prompt[0].icl_mode else None
        wavs, sample_rate = self._run_batched(texts, call, batch_size, should_stop, stream_callback, prefix_codes)
        return self._finish(wavs, sample_rate, start_time, save)

    def save_audio(
        self,
        wav: Any,
        samplerate: int,
    ) -> Path:
        """将音频数据保存为音频文件

        Args:
            wav (Any):
                音频数据
            samplerate (int):
                音频的采样率

        Returns:
            Path: 保存音频的路径
        """
        save_path = OUTPUT_PATH / f"{generate_filename()}.wav"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(save_path, wav, samplerate)
        logger.info("音频文件保存到 '%s'", save_path)
        return save_path
