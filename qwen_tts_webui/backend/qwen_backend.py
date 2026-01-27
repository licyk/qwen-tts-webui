"""Qwen TTS 推理后端"""

import os
from typing import (
    Literal,
    Any,
    TypeAlias,
)
from pathlib import Path

import torch
import soundfile as sf
from modelscope import snapshot_download
from qwen_tts import Qwen3TTSModel

from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
    OUTPUT_PATH,
)
from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.backend.memory_manager import (
    cleanup_models,
    get_free_memory,
    OutOfMemoryError,
)
from qwen_tts_webui.hub import HubManager
from qwen_tts_webui.utils import generate_filename

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)

AttnImpl: TypeAlias = Literal["eager", "sdpa", "flash_attention_2"]
"""注意力加速方案"""


class QwenTTSBackend:
    """Qwen TTS 推理后端"""

    def __init__(
        self,
    ) -> None:
        """Qwen TTS 后端初始化"""
        self.model_name = None
        self.model = None
        self.hub = HubManager(
            hf_token=os.getenv("HF_TOKEN"),
            ms_token=os.getenv("MODELSCOPE_API_TOKEN"),
        )

    def unload_model(
        self,
    ) -> None:
        """卸载模型"""
        logger.info("卸载 %s 模型", self.model_name)
        try:
            del self.model_name
        except NameError:
            pass
        try:
            del self.model
        except NameError:
            pass
        cleanup_models()
        self.model_name = None
        self.model = None
        logger.info("卸载模型完成")

    def load_model(
        self,
        model_name: str | None = None,
        device_map: str | None = "auto",
        dtype: torch.dtype | None = torch.bfloat16,
        attn_implementation: AttnImpl | None = None,
        api_type: Literal["huggingface", "modelscope"] | None = "modelscope",
    ) -> None:
        """加载 Qwen TTS 模型

        Args:
            model_name (str | None):
                加载的 Qwen TTS 模型名称
            device_map (str | None):
                加载模型使用的设备
            dtype (torch.dtype | None):
                加载模型使用的精度
            attn_implementation (AttnImpl | None):
                加载模型时使用的加速方案
            api_type (Literal["huggingface", "modelscope"] | None):
                下载 Qwen TTS 模型时使用的 API 类型

        Raises:
            ValueError:
                当未设置任何模型名称时或者使用错误的 API 类型下载模型时
            OutOfMemoryError:
                内存不足以加载模型时
        """
        if self.model is not None and self.model_name == model_name:
            logger.info("Qwen TTS 模型 %s 已加载", self.model_name)
            return

        if self.model_name is None and model_name is None:
            raise ValueError("未指定 Qwen TTS 模型名称")

        if self.model_name is not None and model_name is not None and self.model_name != model_name:
            logger.info("切换 Qwen TTS 模型: %s -> %s", self.model_name, model_name)
            self.unload_model()
            self.model_name = model_name
        else:
            self.model_name = model_name
            logger.info("加载 Qwen TTS 模型: %s", self.model_name)

        if api_type == "huggingface":
            logger.info("从 HuggingFace 下载 %s 中", self.model_name)
            model_path = self.hub.hf_api.snapshot_download(
                repo_id=model_name,
                repo_type="model",
            )
        elif api_type == "modelscope":
            logger.info("从 ModelScope 下载 %s 中", self.model_name)
            model_path = snapshot_download(
                repo_id=model_name,
                repo_type="model",
            )
        else:
            raise ValueError(f"未知的 API 类型: {api_type}")

        logger.info("%s 模型从 %s 下载完成", self.model_name, api_type)

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        except OutOfMemoryError as e:
            logger.error("加载 Qwen TTS 模型 %s 时发生内存不足: %s", self.model_name, e)
            self.unload_model()
            raise e

        logger.info("%s 模型加载完成, 当前剩余的显存: %.2f MB", self.model_name, get_free_memory() / (1024 * 1024))

    def get_supported_speakers(
        self,
    ) -> list[str] | None:
        """获取 Qwen TTS 支持的说话者列表

        Returns:
            (list[str] | None): 说话者列表, 如果模型不支持指定任何说话者时则返回 None
        """
        return self.model.get_supported_speakers()

    def get_supported_languages(
        self,
    ) -> list[str] | None:
        """获取 Qwen TTS 支持的语言列表

        Returns:
            (list[str] | None): 语言列表, 如果模型不支持指定任何语言时则返回 None
        """
        return self.model.get_supported_languages()

    def generate_custom_voice(
        self,
        text: str,
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
    ) -> Path:
        """使用提示词生成音频

        Args:
            text (str):
                要合成的文本
            speaker (str):
                说话人的名字
            language (str | None):
                合成文本使用的语言, 可设置为 `auto` 自动根据要合成的文本语言进行配置
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

        Returns:
            Path: 生成音频的路径

        Raises:
            OutOfMemoryError: 推理时发生内存不足
        """
        kwargs = {
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
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
        }
        logger.debug("调用 QwenTTSBackend.generate_custom_voice() 的参数: %s", kwargs)
        try:
            logger.info("生成音频中")
            wavs, sr = self.model.generate_custom_voice(**kwargs)
        except OutOfMemoryError as e:
            logger.error("调用 Qwen TTS 模型 %s 进行音频合成时发生内存不足: %s", self.model_name, e)
            self.unload_model()
            raise e

        return self.save_audio(wavs[0], sr)

    def generate_voice_design(
        self,
        text: str,
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
    ) -> Path:
        """使用提示词生成音频, 并且使用提示词描述声音

        Args:
            text (str):
                要合成的文本
            instruct (str | None):
                描述合成的音频的特征
            language (str | None):
                合成文本使用的语言, 可设置为 `auto` 自动根据要合成的文本语言进行配置
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

        Returns:
            Path: 生成音频的路径

        Raises:
            OutOfMemoryError: 推理时发生内存不足
        """
        kwargs = {
            "text": text,
            "instruct": instruct,
            "language": language,
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
        }
        logger.debug("调用 QwenTTSBackend.generate_voice_design() 的参数: %s", kwargs)
        try:
            wavs, sr = self.model.generate_voice_design(**kwargs)
        except OutOfMemoryError as e:
            logger.error("调用 Qwen TTS 模型 %s 进行音频合成时发生内存不足: %s", self.model_name, e)
            self.unload_model()
            raise e

        return self.save_audio(wavs[0], sr)

    def generate_voice_clone(
        self,
        text: str,
        language: str | None = None,
        ref_audio: Path | None = None,
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
    ) -> Path:
        """使用提示词生成音频, 并利用一段音频克隆声音

        Args:
            text (str):
                要合成的文本
            language (str | None):
                合成文本使用的语言, 可设置为 `auto` 自动根据要合成的文本语言进行配置
            ref_audio (Path | None):
                用于构建提示的参考音频。如果为 None, 则 `voice_clone_prompt` 必须设置
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

        Returns:
            Path: 生成音频的路径

        Raises:
            OutOfMemoryError: 推理时发生内存不足
            ValueError: 当 `x_vector_only_mode=False` 并且 `ref_text` 为空时
        """
        kwargs = {
            "text": text,
            "language": language,
            "ref_audio": str(ref_audio),
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
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
        }
        logger.debug("调用 QwenTTSBackend.generate_voice_clone() 的参数: %s", kwargs)
        if not x_vector_only_mode and (ref_text is None or ref_text.strip() == ""):
            raise ValueError("当 `x_vector_only_mode=False` 时, 需要提供参考音频对应的参考文本, 而当前的参考文本 `ref_text` 为空")

        try:
            wavs, sr = self.model.generate_voice_clone(**kwargs)
        except OutOfMemoryError as e:
            logger.error("调用 Qwen TTS 模型 %s 进行音频合成时发生内存不足: %s", self.model_name, e)
            self.unload_model()
            raise e

        return self.save_audio(wavs[0], sr)

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
