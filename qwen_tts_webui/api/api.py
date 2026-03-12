"""API 接口实现"""

import base64
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException

from qwen_tts_webui.api import models
from qwen_tts_webui.backend.memory_manager import OutOfMemoryError
from qwen_tts_webui.config_manager.config import (
    QWEN_TTS_BASE_MODEL_LIST,
    QWEN_TTS_CUSTOM_VOICE_MODEL_LIST,
    QWEN_TTS_VOICE_DESIGN_MODEL_LIST,
    OUTPUT_PATH,
)
from qwen_tts_webui.config_manager.shared import (
    opts,
    state,
    get_backend,
)


def encode_audio_to_base64(
    audio_path: str,
) -> str:
    """将音频文件编码为 Base64 字符串

    Args:
        audio_path (str):
            音频文件路径

    Returns:
        str:
            Base64 编码的音频数据
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def decode_base64_to_audio(
    base64_str: str,
    output_path: Path,
) -> Path:
    """将 Base64 字符串解码为音频文件

    Args:
        base64_str (str):
            Base64 编码的音频数据
        output_path (Path):
            输出文件路径

    Returns:
        Path:
            保存的音频文件路径
    """
    audio_bytes = base64.b64decode(base64_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path


class Api:
    """API 类"""

    def __init__(
        self,
        app: FastAPI,
        queue_lock: Lock,
    ) -> None:
        """初始化 API

        Args:
            app (FastAPI):
                FastAPI 应用实例
            queue_lock (Lock):
                队列锁
        """
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock

        # 注册 API 路由
        self.add_api_route(
            "/qwenapi/v1/custom-voice",
            self.custom_voice_api,
            methods=["POST"],
            response_model=models.TTSResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/voice-design",
            self.voice_design_api,
            methods=["POST"],
            response_model=models.TTSResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/voice-clone",
            self.voice_clone_api,
            methods=["POST"],
            response_model=models.TTSResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/models",
            self.get_models,
            methods=["GET"],
            response_model=models.ModelsResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/speakers",
            self.get_speakers,
            methods=["GET"],
            response_model=models.SpeakersResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/languages",
            self.get_languages,
            methods=["GET"],
            response_model=models.LanguagesResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/options",
            self.get_options,
            methods=["GET"],
            response_model=models.OptionsResponse,
        )
        self.add_api_route(
            "/qwenapi/v1/interrupt",
            self.interrupt,
            methods=["POST"],
            response_model=models.InterruptResponse,
        )

    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        **kwargs: Any,
    ) -> None:
        """添加 API 路由

        Args:
            path (str):
                路由路径
            endpoint (Callable[..., Any]):
                端点函数
            **kwargs (Any):
                其他参数
        """
        self.app.add_api_route(path, endpoint, **kwargs)

    def custom_voice_api(
        self,
        req: models.CustomVoiceRequest,
    ) -> models.TTSResponse:
        """声音生成 API

        Args:
            req (models.CustomVoiceRequest):
                请求参数

        Returns:
            models.TTSResponse:
                响应数据
        """
        output_paths: list[str] = []
        if req.segment_gen:
            text_list = [x.strip() for x in req.text.splitlines() if x.strip() != ""]
        else:
            text_list = [req.text]

        try:
            start_time = time.perf_counter()
            with self.queue_lock:
                # 加载模型
                get_backend().load_model(
                    model_name=req.model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )

                # 获取实际的 speaker 和 language
                speakers = get_backend().get_supported_speakers() or []
                actual_speaker = req.speaker
                if req.speaker is None or req.speaker == "default":
                    actual_speaker = speakers[0] if speakers else None

                actual_language = None if req.language == "auto" or req.language is None else req.language

                # 生成音频
                for t in text_list:
                    output_path = get_backend().generate_custom_voice(
                        text=t,
                        speaker=actual_speaker,
                        language=actual_language,
                        instruct=req.instruct,
                        do_sample=req.do_sample if req.do_sample is not None else opts.do_sample,
                        top_k=req.top_k if req.top_k is not None else opts.top_k,
                        top_p=req.top_p if req.top_p is not None else opts.top_p,
                        temperature=req.temperature if req.temperature is not None else opts.temperature,
                        repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else opts.repetition_penalty,
                        subtalker_dosample=req.subtalker_dosample if req.subtalker_dosample is not None else opts.subtalker_dosample,
                        subtalker_top_k=req.subtalker_top_k if req.subtalker_top_k is not None else opts.subtalker_top_k,
                        subtalker_top_p=req.subtalker_top_p if req.subtalker_top_p is not None else opts.subtalker_top_p,
                        subtalker_temperature=req.subtalker_temperature if req.subtalker_temperature is not None else opts.subtalker_temperature,
                        max_new_tokens=req.max_new_tokens if req.max_new_tokens is not None else opts.max_new_tokens,
                    )
                    if state.interrupted:
                        raise HTTPException(status_code=500, detail="任务已中断")
                    output_paths.append(str(output_path))

            # 将音频文件编码为 Base64
            audio_files_base64 = [encode_audio_to_base64(path) for path in output_paths]
            elapsed_time = time.perf_counter() - start_time

            return models.TTSResponse(
                audio_files_base64=audio_files_base64,
                info=f"成功生成 {len(audio_files_base64)} 个音频文件, 耗时: {elapsed_time:.2f}s",
            )

        except OutOfMemoryError as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"生成音频所需的显存不足: {str(e)}") from e
        except Exception as e:  # pylint: disable=duplicate-except
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}") from e

    def voice_design_api(
        self,
        req: models.VoiceDesignRequest,
    ) -> models.TTSResponse:
        """声音设计 API

        Args:
            req (models.VoiceDesignRequest):
                请求参数

        Returns:
            models.TTSResponse:
                响应数据
        """
        output_paths: list[str] = []
        if req.segment_gen:
            text_list = [x.strip() for x in req.text.splitlines() if x.strip() != ""]
        else:
            text_list = [req.text]

        try:
            start_time = time.perf_counter()
            with self.queue_lock:
                # 加载模型
                get_backend().load_model(
                    model_name=req.model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )

                actual_language = None if req.language == "auto" or req.language is None else req.language

                # 生成音频
                for t in text_list:
                    output_path = get_backend().generate_voice_design(
                        text=t,
                        instruct=req.instruct,
                        language=actual_language,
                        do_sample=req.do_sample if req.do_sample is not None else opts.do_sample,
                        top_k=req.top_k if req.top_k is not None else opts.top_k,
                        top_p=req.top_p if req.top_p is not None else opts.top_p,
                        temperature=req.temperature if req.temperature is not None else opts.temperature,
                        repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else opts.repetition_penalty,
                        subtalker_dosample=req.subtalker_dosample if req.subtalker_dosample is not None else opts.subtalker_dosample,
                        subtalker_top_k=req.subtalker_top_k if req.subtalker_top_k is not None else opts.subtalker_top_k,
                        subtalker_top_p=req.subtalker_top_p if req.subtalker_top_p is not None else opts.subtalker_top_p,
                        subtalker_temperature=req.subtalker_temperature if req.subtalker_temperature is not None else opts.subtalker_temperature,
                        max_new_tokens=req.max_new_tokens if req.max_new_tokens is not None else opts.max_new_tokens,
                    )
                    if state.interrupted:
                        raise HTTPException(status_code=500, detail="任务已中断")
                    output_paths.append(str(output_path))

            # 将音频文件编码为 Base64
            audio_files_base64 = [encode_audio_to_base64(path) for path in output_paths]
            elapsed_time = time.perf_counter() - start_time

            return models.TTSResponse(
                audio_files_base64=audio_files_base64,
                info=f"成功生成 {len(audio_files_base64)} 个音频文件, 耗时: {elapsed_time:.2f}s",
            )

        except OutOfMemoryError as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"生成音频所需的显存不足: {str(e)}") from e
        except Exception as e:  # pylint: disable=duplicate-except
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}") from e

    def voice_clone_api(
        self,
        req: models.VoiceCloneRequest,
    ) -> models.TTSResponse:
        """声音克隆 API

        Args:
            req (models.VoiceCloneRequest):
                请求参数

        Returns:
            models.TTSResponse:
                响应数据
        """
        output_paths: list[str] = []
        if req.segment_gen:
            text_list = [x.strip() for x in req.text.splitlines() if x.strip() != ""]
        else:
            text_list = [req.text]

        try:
            start_time = time.perf_counter()

            # 解码参考音频
            temp_audio_path = OUTPUT_PATH / "temp" / f"ref_audio_{int(time.time())}.wav"
            ref_audio_path = decode_base64_to_audio(req.ref_audio_base64, temp_audio_path)

            with self.queue_lock:
                # 加载模型
                get_backend().load_model(
                    model_name=req.model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )

                actual_language = None if req.language == "auto" or req.language is None else req.language

                # 生成音频
                for t in text_list:
                    output_path = get_backend().generate_voice_clone(
                        text=t,
                        language=actual_language,
                        ref_audio=ref_audio_path,
                        ref_text=req.ref_text,
                        x_vector_only_mode=(req.ref_text is None or req.ref_text.strip() == ""),
                        do_sample=req.do_sample if req.do_sample is not None else opts.do_sample,
                        top_k=req.top_k if req.top_k is not None else opts.top_k,
                        top_p=req.top_p if req.top_p is not None else opts.top_p,
                        temperature=req.temperature if req.temperature is not None else opts.temperature,
                        repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else opts.repetition_penalty,
                        subtalker_dosample=req.subtalker_dosample if req.subtalker_dosample is not None else opts.subtalker_dosample,
                        subtalker_top_k=req.subtalker_top_k if req.subtalker_top_k is not None else opts.subtalker_top_k,
                        subtalker_top_p=req.subtalker_top_p if req.subtalker_top_p is not None else opts.subtalker_top_p,
                        subtalker_temperature=req.subtalker_temperature if req.subtalker_temperature is not None else opts.subtalker_temperature,
                        max_new_tokens=req.max_new_tokens if req.max_new_tokens is not None else opts.max_new_tokens,
                    )
                    if state.interrupted:
                        raise HTTPException(status_code=500, detail="任务已中断")
                    output_paths.append(str(output_path))

            # 清理临时文件
            if ref_audio_path.exists():
                ref_audio_path.unlink()

            # 将音频文件编码为 Base64
            audio_files_base64 = [encode_audio_to_base64(path) for path in output_paths]
            elapsed_time = time.perf_counter() - start_time

            return models.TTSResponse(
                audio_files_base64=audio_files_base64,
                info=f"成功生成 {len(audio_files_base64)} 个音频文件, 耗时: {elapsed_time:.2f}s",
            )

        except OutOfMemoryError as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"生成音频所需的显存不足: {str(e)}") from e
        except Exception as e:  # pylint: disable=duplicate-except
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}") from e

    def get_models(
        self,
    ) -> models.ModelsResponse:
        """获取可用模型列表

        Returns:
            models.ModelsResponse:
                模型列表
        """
        model_list = []

        # 声音生成模型
        for model_name in QWEN_TTS_CUSTOM_VOICE_MODEL_LIST + opts.extra_custom_voice_models:
            model_list.append(models.ModelInfo(name=model_name, type="custom_voice"))

        # 声音设计模型
        for model_name in QWEN_TTS_VOICE_DESIGN_MODEL_LIST + opts.extra_voice_design_models:
            model_list.append(models.ModelInfo(name=model_name, type="voice_design"))

        # 声音克隆模型
        for model_name in QWEN_TTS_BASE_MODEL_LIST + opts.extra_voice_clone_models:
            model_list.append(models.ModelInfo(name=model_name, type="voice_clone"))

        return models.ModelsResponse(models=model_list)

    def get_speakers(
        self,
    ) -> models.SpeakersResponse:
        """获取支持的发言人列表

        Returns:
            models.SpeakersResponse:
                发言人列表
        """
        speakers = get_backend().get_supported_speakers() or []
        return models.SpeakersResponse(speakers=speakers)

    def get_languages(
        self,
    ) -> models.LanguagesResponse:
        """获取支持的语言列表

        Returns:
            models.LanguagesResponse:
                语言列表
        """
        languages = get_backend().get_supported_languages() or []
        return models.LanguagesResponse(languages=languages)

    def get_options(
        self,
    ) -> models.OptionsResponse:
        """获取配置选项

        Returns:
            models.OptionsResponse:
                配置选项
        """
        return models.OptionsResponse(
            api_type=opts.api_type,
            device_map=str(opts.device_map),
            dtype=opts.dtype,
            attn_implementation=opts.attn_implementation,
            extra_custom_voice_models=opts.extra_custom_voice_models,
            extra_voice_design_models=opts.extra_voice_design_models,
            extra_voice_clone_models=opts.extra_voice_clone_models,
            do_sample=opts.do_sample,
            top_k=opts.top_k,
            top_p=opts.top_p,
            temperature=opts.temperature,
            repetition_penalty=opts.repetition_penalty,
            subtalker_dosample=opts.subtalker_dosample,
            subtalker_top_k=opts.subtalker_top_k,
            subtalker_top_p=opts.subtalker_top_p,
            subtalker_temperature=opts.subtalker_temperature,
            max_new_tokens=opts.max_new_tokens,
        )

    def interrupt(
        self,
    ) -> models.InterruptResponse:
        """中断当前任务

        Returns:
            models.InterruptResponse:
                中断响应
        """
        state.interrupt()
        return models.InterruptResponse(message="任务已中断")

    def launch(
        self,
        server_name: str,
        port: int,
    ) -> None:
        """启动 API 服务器

        Args:
            server_name (str):
                服务器地址
            port (int):
                端口号
        """

        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=server_name,
            port=port,
        )
