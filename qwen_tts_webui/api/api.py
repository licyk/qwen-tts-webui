"""OpenAI 兼容的 API 接口实现

主要接口:
- `POST /v1/audio/speech`: 语音合成 (兼容 OpenAI Audio Speech API, 支持流式输出)
- `GET /v1/models`: 获取可用模型列表
- `GET /v1/audio/voices`: 获取可用声音列表
- `POST /v1/audio/voices`: 注册声音克隆参考音频
- `DELETE /v1/audio/voices/{voice_id}`: 删除注册的声音
- `POST /v1/audio/interrupt`: 中断当前任务
"""

import base64
import binascii
import hmac
import io
import json
import os
import queue
import struct
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute

from qwen_tts_webui.api import models
from qwen_tts_webui.backend.errors import GenerationInterruptedError
from qwen_tts_webui.backend.memory_manager import OutOfMemoryError
from qwen_tts_webui.config_manager.config import (
    LOGGER_COLOR,
    LOGGER_LEVEL,
    OPENAI_MODEL_ALIASES,
    OPENAI_VOICE_ALIASES,
    QWEN_TTS_BASE_MODEL_LIST,
    QWEN_TTS_CUSTOM_VOICE_MODEL_LIST,
    QWEN_TTS_VOICE_DESIGN_MODEL_LIST,
    VOICES_PATH,
)
from qwen_tts_webui.config_manager.shared import (
    get_backend,
    opts,
    state,
)
from qwen_tts_webui.logger import get_logger

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)

QWEN_TTS_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
"""Qwen TTS 声音生成模型的内置发言人 (模型未加载时用于展示)"""

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
"""音频格式对应的 Content-Type"""

PROGRESSIVE_FORMATS = ("pcm", "wav")
"""支持边生成边输出的音频格式"""

CODEC_FRAME_RATE = 12.5
"""音频码本的帧率 (每秒帧数)"""


class OpenAIError(Exception):
    """以 OpenAI 错误格式返回的 API 错误"""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        """初始化 API 错误

        Args:
            status_code (int):
                HTTP 状态码
            message (str):
                错误信息
            error_type (str):
                错误类型
            param (str | None):
                导致错误的参数
            code (str | None):
                错误代码
        """
        super().__init__(message)
        self.status_code = status_code
        self.detail = models.ErrorDetail(message=message, type=error_type, param=param, code=code)

    def to_response(self) -> JSONResponse:
        """转换为 JSON 响应

        Returns:
            JSONResponse: 错误响应
        """
        return JSONResponse(status_code=self.status_code, content=models.ErrorResponse(error=self.detail).model_dump())


class OpenAIRoute(APIRoute):
    """将请求校验错误与 `OpenAIError` 转换为 OpenAI 错误格式的路由, 仅作用于 API 路由, 不影响 WebUI"""

    def get_route_handler(self) -> Callable:
        """获取路由处理函数

        Returns:
            Callable: 路由处理函数
        """
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as e:
                errors = e.errors()
                first = errors[0] if errors else {}
                loc = [str(x) for x in first.get("loc", []) if x != "body"]
                message = "; ".join(f"{'.'.join(str(x) for x in err.get('loc', []) if x != 'body')}: {err.get('msg')}" for err in errors)
                return OpenAIError(400, message or "Invalid request", param=".".join(loc) or None).to_response()
            except OpenAIError as e:
                return e.to_response()

        return route_handler


def decode_base64_audio(
    data: str,
    param: str,
) -> tuple[np.ndarray, int]:
    """解码 Base64 / data URI 格式的音频

    Args:
        data (str):
            Base64 编码的音频或者 data URI
        param (str):
            对应的请求参数名称, 用于错误提示

    Returns:
        tuple[np.ndarray, int]: 单声道 float32 音频数据与采样率

    Raises:
        OpenAIError: 音频无法解码时
    """
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        audio_bytes = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise OpenAIError(400, f"`{param}` is not valid base64 encoded audio: {e}", param=param) from e
    try:
        wav, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError, TypeError) as e:
        raise OpenAIError(400, f"`{param}` could not be decoded as audio (supported: wav / flac / ogg / mp3): {e}", param=param) from e
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if wav.size == 0:
        raise OpenAIError(400, f"`{param}` contains no audio samples", param=param)
    return wav.astype(np.float32), int(sample_rate)


def to_pcm16(
    wav: np.ndarray,
) -> bytes:
    """转换为 16 bit 小端 PCM 数据

    Args:
        wav (np.ndarray):
            float 音频数据

    Returns:
        bytes: PCM 数据
    """
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_stream_header(
    sample_rate: int,
) -> bytes:
    """生成长度未知的流式 WAV 文件头 (16 bit 单声道 PCM)

    Args:
        sample_rate (int):
            采样率

    Returns:
        bytes: WAV 文件头
    """
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF)


def encode_audio(
    wav: np.ndarray,
    sample_rate: int,
    response_format: str,
) -> bytes:
    """将音频编码为指定格式

    Args:
        wav (np.ndarray):
            float 音频数据
        sample_rate (int):
            采样率
        response_format (str):
            音频格式

    Returns:
        bytes: 编码后的音频

    Raises:
        OpenAIError: 不支持的音频格式
    """
    wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    if response_format == "pcm":
        return to_pcm16(wav)
    if response_format == "aac":
        raise OpenAIError(400, "`aac` is not supported by this server, please use one of: mp3, opus, flac, wav, pcm", param="response_format", code="unsupported_format")

    buffer = io.BytesIO()
    if response_format == "wav":
        sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    elif response_format == "flac":
        sf.write(buffer, wav, sample_rate, format="FLAC", subtype="PCM_16")
    elif response_format == "mp3":
        sf.write(buffer, wav, sample_rate, format="MP3", subtype="MPEG_LAYER_III")
    elif response_format == "opus":
        if sample_rate not in (8000, 12000, 16000, 24000, 48000):
            import librosa

            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=48000)
            sample_rate = 48000
        sf.write(buffer, wav, sample_rate, format="OGG", subtype="OPUS")
    else:
        raise OpenAIError(400, f"Unsupported response_format `{response_format}`", param="response_format")
    return buffer.getvalue()


def change_speed(
    wav: np.ndarray,
    speed: float,
) -> np.ndarray:
    """在不改变音调的情况下调整语速

    Args:
        wav (np.ndarray):
            音频数据
        speed (float):
            语速倍率

    Returns:
        np.ndarray: 调整后的音频数据
    """
    if abs(speed - 1.0) < 1e-3 or wav.size == 0:
        return wav
    import librosa

    return librosa.effects.time_stretch(np.asarray(wav, dtype=np.float32), rate=speed)


@dataclass
class SpeechJob:
    """一次语音合成任务"""

    req: models.SpeechRequest
    model_name: str
    model_type: str
    texts: list[str]
    ref_audio: Path | tuple[np.ndarray, int] | None = None
    ref_text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Api:
    """OpenAI 兼容的 API"""

    def __init__(
        self,
        app: FastAPI,
        queue_lock: Lock,
        api_key: str | None = None,
    ) -> None:
        """初始化 API 并注册路由

        Args:
            app (FastAPI):
                FastAPI 应用实例
            queue_lock (Lock):
                队列锁, 与 WebUI 共享, 保证同一时间只有一个生成任务
            api_key (str | None):
                API 访问密钥, 为 None 时不进行鉴权
        """
        self.app = app
        self.queue_lock = queue_lock
        self.api_key = api_key or os.getenv("QWEN_TTS_API_KEY") or None
        self.router = APIRouter(route_class=OpenAIRoute, tags=["OpenAI Compatible API"])

        error_responses: dict[int | str, dict[str, Any]] = {code: {"model": models.ErrorResponse} for code in (400, 401, 404, 500, 503)}
        audio_content = {media: {"schema": {"type": "string", "format": "binary"}} for media in sorted(set(MEDIA_TYPES.values())) + ["text/event-stream"]}
        self.router.add_api_route(
            "/v1/audio/speech",
            self.create_speech,
            methods=["POST"],
            response_class=Response,
            responses={200: {"content": audio_content, "description": "生成的音频"}, **error_responses},
            summary="语音合成 (兼容 OpenAI)",
        )
        self.router.add_api_route("/v1/models", self.list_models, methods=["GET"], response_model=models.ModelList, responses=error_responses, summary="获取模型列表")
        self.router.add_api_route(
            "/v1/models/{model_id:path}",
            self.retrieve_model,
            methods=["GET"],
            response_model=models.ModelObject,
            responses=error_responses,
            summary="获取模型信息",
        )
        self.router.add_api_route("/v1/audio/voices", self.list_voices, methods=["GET"], response_model=models.VoiceList, responses=error_responses, summary="获取声音列表")
        self.router.add_api_route(
            "/v1/audio/voices",
            self.create_voice,
            methods=["POST"],
            response_model=models.VoiceObject,
            responses=error_responses,
            summary="注册声音克隆参考音频",
        )
        self.router.add_api_route(
            "/v1/audio/voices/{voice_id}",
            self.delete_voice,
            methods=["DELETE"],
            response_model=models.DeleteVoiceResponse,
            responses=error_responses,
            summary="删除注册的声音",
        )
        self.router.add_api_route(
            "/v1/audio/interrupt",
            self.interrupt,
            methods=["POST"],
            response_model=models.InterruptResponse,
            responses=error_responses,
            summary="中断当前任务",
        )
        self.app.include_router(self.router)

    # ------------------------------------------------------------------
    # 通用工具
    # ------------------------------------------------------------------
    def check_auth(
        self,
        request: Request,
    ) -> None:
        """校验 API 访问密钥

        Args:
            request (Request):
                请求

        Raises:
            OpenAIError: 密钥错误时
        """
        if not self.api_key:
            return
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), self.api_key.encode()):
            raise OpenAIError(401, "Incorrect API key provided", error_type="invalid_request_error", code="invalid_api_key")

    @staticmethod
    def model_catalog() -> dict[str, str]:
        """获取可用模型及其类型

        Returns:
            dict[str, str]: 模型名称到模型类型的映射
        """
        catalog: dict[str, str] = {}
        for name in QWEN_TTS_CUSTOM_VOICE_MODEL_LIST + opts.extra_custom_voice_models:
            catalog.setdefault(name, "custom_voice")
        for name in QWEN_TTS_VOICE_DESIGN_MODEL_LIST + opts.extra_voice_design_models:
            catalog.setdefault(name, "voice_design")
        for name in QWEN_TTS_BASE_MODEL_LIST + opts.extra_voice_clone_models:
            catalog.setdefault(name, "voice_clone")
        return catalog

    def resolve_model(
        self,
        model: str,
    ) -> tuple[str, str]:
        """解析请求中的模型名称

        Args:
            model (str):
                请求中的模型名称或别名

        Returns:
            tuple[str, str]: 实际的模型名称与模型类型

        Raises:
            OpenAIError: 模型不存在时
        """
        catalog = self.model_catalog()
        name = OPENAI_MODEL_ALIASES.get(model, model) if model not in catalog else model
        if name not in catalog:
            raise OpenAIError(404, f"The model `{model}` does not exist", param="model", code="model_not_found")
        return name, catalog[name]

    @staticmethod
    def voice_paths(
        voice_id: str,
    ) -> tuple[Path, Path]:
        """获取注册声音的音频与元数据路径

        Args:
            voice_id (str):
                声音名称

        Returns:
            tuple[Path, Path]: 音频路径与元数据路径
        """
        return VOICES_PATH / f"{voice_id}.wav", VOICES_PATH / f"{voice_id}.json"

    def load_registered_voice(
        self,
        voice_id: str,
    ) -> tuple[Path, dict[str, Any]] | None:
        """读取注册的声音

        Args:
            voice_id (str):
                声音名称

        Returns:
            (tuple[Path, dict[str, Any]] | None): 音频路径与元数据, 不存在时返回 None
        """
        if not voice_id or "/" in voice_id or "\\" in voice_id or voice_id.startswith("."):
            return None
        wav_path, meta_path = self.voice_paths(voice_id)
        if not wav_path.is_file():
            return None
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("读取声音 %s 的元数据失败", voice_id)
        return wav_path, meta

    def sampling_kwargs(
        self,
        req: models.SpeechRequest,
    ) -> dict[str, Any]:
        """合并请求中的采样参数与 WebUI 设置中的默认值

        Args:
            req (models.SpeechRequest):
                请求

        Returns:
            dict[str, Any]: 采样参数
        """
        names = [
            "do_sample",
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "subtalker_dosample",
            "subtalker_top_k",
            "subtalker_top_p",
            "subtalker_temperature",
            "max_new_tokens",
        ]
        return {name: getattr(req, name) if getattr(req, name) is not None else getattr(opts, name) for name in names}

    # ------------------------------------------------------------------
    # 语音合成
    # ------------------------------------------------------------------
    def prepare_job(
        self,
        req: models.SpeechRequest,
    ) -> SpeechJob:
        """校验请求并生成合成任务 (不需要加载模型的校验在此完成)

        Args:
            req (models.SpeechRequest):
                请求

        Returns:
            SpeechJob: 合成任务

        Raises:
            OpenAIError: 请求参数错误时
        """
        model_name, model_type = self.resolve_model(req.model)
        if req.segment:
            texts = [x.strip() for x in req.input.splitlines() if x.strip()]
        else:
            texts = [req.input.strip()] if req.input.strip() else []
        if not texts:
            raise OpenAIError(400, "`input` must not be empty", param="input")

        job = SpeechJob(req=req, model_name=model_name, model_type=model_type, texts=texts)
        if model_type == "voice_design" and not (req.instructions and req.instructions.strip()):
            raise OpenAIError(400, "Voice design models require `instructions` describing the voice", param="instructions")
        if model_type == "voice_clone":
            if req.ref_audio:
                job.ref_audio = decode_base64_audio(req.ref_audio, "ref_audio")
                job.ref_text = req.ref_text
            else:
                registered = self.load_registered_voice(req.voice or "")
                if registered is None:
                    raise OpenAIError(
                        400,
                        f"Voice clone models require `voice` to be a voice registered via `POST /v1/audio/voices` or `ref_audio`, got voice `{req.voice}`",
                        param="voice",
                        code="voice_not_found",
                    )
                job.ref_audio, meta = registered
                job.ref_text = req.ref_text if req.ref_text is not None else meta.get("ref_text")
        return job

    def resolve_speaker(
        self,
        voice: str | None,
    ) -> str | None:
        """解析声音生成模型使用的发言人

        Args:
            voice (str | None):
                请求中的声音名称

        Returns:
            (str | None): 发言人名称

        Raises:
            OpenAIError: 发言人不存在时
        """
        speakers = get_backend().get_supported_speakers() or []
        name = (voice or "").strip().lower()
        if not name or name == "default":
            return speakers[0] if speakers else None
        if name in speakers:
            return name
        alias = OPENAI_VOICE_ALIASES.get(name)
        if alias in speakers:
            return alias
        supported = ", ".join(speakers + sorted(OPENAI_VOICE_ALIASES))
        raise OpenAIError(400, f"Unsupported voice `{voice}`, supported voices: {supported}", param="voice", code="invalid_voice")

    @staticmethod
    def resolve_language(
        language: str | None,
    ) -> str | None:
        """解析语言参数

        Args:
            language (str | None):
                请求中的语言

        Returns:
            (str | None): 模型使用的语言, 自动检测时为 None

        Raises:
            OpenAIError: 语言不支持时
        """
        if language is None or language.strip().lower() in ("", "auto"):
            return None
        supported = get_backend().get_supported_languages() or []
        if supported and language.lower() not in supported:
            raise OpenAIError(400, f"Unsupported language `{language}`, supported languages: {', '.join(supported)}", param="language", code="invalid_language")
        return language.lower()

    def run_job(
        self,
        job: SpeechJob,
        should_stop: Callable[[], bool],
        stream_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> Any:
        """加载模型并执行合成任务, 调用前需要持有队列锁

        Args:
            job (SpeechJob):
                合成任务
            should_stop (Callable[[], bool]):
                返回 True 时中断生成
            stream_callback (Callable[[np.ndarray, int], None] | None):
                流式输出回调

        Returns:
            TTSResult: 生成结果
        """
        backend = get_backend()
        backend.load_model(
            model_name=job.model_name,
            api_type=opts.api_type,
            device_map=opts.device_map,
            dtype=getattr(torch, opts.dtype.split(".")[-1]),
            attn_implementation=opts.attn_implementation,
            fast_generation=opts.fast_generation,
            cuda_graph=opts.cuda_graph,
        )
        req = job.req
        language = self.resolve_language(req.language)
        if req.seed is not None:
            torch.manual_seed(req.seed)
        common: dict[str, Any] = {
            **self.sampling_kwargs(req),
            "batch_size": opts.batch_size,
            "should_stop": should_stop,
            "stream_callback": stream_callback,
        }
        if job.model_type == "custom_voice":
            speaker = self.resolve_speaker(req.voice)
            instruct = req.instructions.strip() if req.instructions and req.instructions.strip() else None
            return backend.generate_custom_voice(text=job.texts, speaker=speaker, language=language, instruct=instruct, **common)
        if job.model_type == "voice_design":
            return backend.generate_voice_design(text=job.texts, instruct=req.instructions.strip(), language=language, **common)
        ref_text = job.ref_text.strip() if job.ref_text and job.ref_text.strip() else None
        return backend.generate_voice_clone(
            text=job.texts,
            language=language,
            ref_audio=job.ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=ref_text is None,
            **common,
        )

    @staticmethod
    def to_openai_error(
        e: BaseException,
    ) -> OpenAIError:
        """将生成过程中的异常转换为 API 错误

        Args:
            e (BaseException):
                异常

        Returns:
            OpenAIError: API 错误
        """
        if isinstance(e, OpenAIError):
            return e
        if isinstance(e, GenerationInterruptedError):
            return OpenAIError(503, "The generation was interrupted", error_type="server_error", code="generation_interrupted")
        if isinstance(e, OutOfMemoryError):
            return OpenAIError(503, f"Not enough memory to generate audio: {e}", error_type="server_error", code="out_of_memory")
        if isinstance(e, ValueError):
            return OpenAIError(400, str(e))
        traceback.print_exception(e)
        return OpenAIError(500, f"Generation failed: {e}", error_type="server_error")

    def create_speech(
        self,
        req: models.SpeechRequest,
        request: Request,
    ) -> Response:
        """语音合成, 与 OpenAI `POST /v1/audio/speech` 兼容

        Args:
            req (models.SpeechRequest):
                请求参数
            request (Request):
                原始请求

        Returns:
            Response: 音频数据 / SSE 事件流
        """
        self.check_auth(request)
        job = self.prepare_job(req)
        progressive = req.response_format in PROGRESSIVE_FORMATS
        if req.stream_format == "sse" or progressive:
            return self.stream_speech(job, progressive)

        with self.queue_lock:
            state.begin()
            try:
                result = self.run_job(job, should_stop=lambda: state.interrupted)
            except Exception as e:
                raise self.to_openai_error(e) from e
            finally:
                state.end()

        audio = change_speed(np.concatenate(result.wavs), req.speed)
        content = encode_audio(audio, result.sample_rate, req.response_format)
        return Response(content=content, media_type=MEDIA_TYPES[req.response_format], headers=self.result_headers(result, len(audio) / result.sample_rate))

    @staticmethod
    def result_headers(
        result: Any,
        audio_seconds: float,
    ) -> dict[str, str]:
        """生成包含生成统计信息的响应头

        Args:
            result (TTSResult):
                生成结果
            audio_seconds (float):
                音频时长

        Returns:
            dict[str, str]: 响应头
        """
        return {
            "X-Audio-Duration": f"{audio_seconds:.3f}",
            "X-Generation-Time": f"{result.elapsed:.3f}",
            "X-Real-Time-Factor": f"{result.elapsed / audio_seconds:.4f}" if audio_seconds > 0 else "0",
        }

    def stream_speech(
        self,
        job: SpeechJob,
        progressive: bool,
    ) -> Response:
        """流式语音合成

        生成任务在后台线程中执行 (持有队列锁), 每生成一段音频就推送到队列中, 响应从队列中读取并输出.
        在第一个音频片段生成之前发生的错误 (例如发言人不存在) 会以正常的 HTTP 错误返回.

        Args:
            job (SpeechJob):
                合成任务
            progressive (bool):
                音频格式是否支持边生成边输出

        Returns:
            Response: 流式响应
        """
        req = job.req
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        cancel = threading.Event()
        start_time = time.perf_counter()

        def on_chunk(wav: np.ndarray, sample_rate: int) -> None:
            if progressive:
                events.put(("audio", (change_speed(wav, req.speed), sample_rate)))

        def worker() -> None:
            try:
                with self.queue_lock:
                    state.begin()
                    try:
                        result = self.run_job(job, should_stop=lambda: cancel.is_set() or state.interrupted, stream_callback=on_chunk)
                        input_tokens = sum(get_backend().count_tokens(t) for t in job.texts)
                    finally:
                        state.end()
                events.put(("done", (result, input_tokens)))
            except BaseException as e:  # noqa: BLE001 - 所有异常都需要传递给响应线程
                events.put(("error", e))

        threading.Thread(target=worker, name="qwen-tts-api-stream", daemon=True).start()
        first = events.get()
        if first[0] == "error":
            raise self.to_openai_error(first[1])

        sse = req.stream_format == "sse"

        def encode_chunk(wav: np.ndarray, sample_rate: int, is_first: bool) -> bytes:
            if req.response_format == "pcm":
                return to_pcm16(wav)
            if req.response_format == "wav":
                return (wav_stream_header(sample_rate) if is_first else b"") + to_pcm16(wav)
            return encode_audio(wav, sample_rate, req.response_format)

        def sse_event(payload: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

        def body() -> Iterator[bytes]:
            event = first
            sent_audio = False
            first_chunk_at = None
            try:
                while True:
                    kind, payload = event
                    if kind == "audio":
                        wav, sample_rate = payload
                        data = encode_chunk(wav, sample_rate, not sent_audio)
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter() - start_time
                        sent_audio = True
                        yield sse_event({"type": "speech.audio.delta", "audio": base64.b64encode(data).decode()}) if sse else data
                    elif kind == "done":
                        result, input_tokens = payload
                        if not progressive:
                            audio = change_speed(np.concatenate(result.wavs), req.speed)
                            data = encode_audio(audio, result.sample_rate, req.response_format)
                            yield sse_event({"type": "speech.audio.delta", "audio": base64.b64encode(data).decode()}) if sse else data
                        output_tokens = round(result.audio_seconds * CODEC_FRAME_RATE)
                        logger.info(
                            "流式合成完成, 首个音频片段耗时: %s, 总耗时: %.2fs",
                            f"{first_chunk_at:.2f}s" if first_chunk_at is not None else "N/A",
                            time.perf_counter() - start_time,
                        )
                        if sse:
                            usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
                            yield sse_event({"type": "speech.audio.done", "usage": usage})
                        return
                    else:
                        error = self.to_openai_error(payload)
                        logger.error("流式合成失败: %s", error)
                        if sse:
                            yield sse_event({"type": "error", "error": error.detail.model_dump()})
                        return
                    event = events.get()
            finally:
                cancel.set()

        if sse:
            return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        return StreamingResponse(body(), media_type=MEDIA_TYPES[req.response_format])

    # ------------------------------------------------------------------
    # 模型与声音
    # ------------------------------------------------------------------
    def list_models(
        self,
        request: Request,
    ) -> models.ModelList:
        """获取可用模型列表

        Args:
            request (Request):
                原始请求

        Returns:
            models.ModelList: 模型列表
        """
        self.check_auth(request)
        catalog = self.model_catalog()
        data = [models.ModelObject(id=name, type=model_type) for name, model_type in catalog.items()]
        for alias, target in OPENAI_MODEL_ALIASES.items():
            if target in catalog and alias not in catalog:
                data.append(models.ModelObject(id=alias, type=catalog[target], alias_of=target))
        return models.ModelList(data=data)

    def retrieve_model(
        self,
        model_id: str,
        request: Request,
    ) -> models.ModelObject:
        """获取模型信息

        Args:
            model_id (str):
                模型名称
            request (Request):
                原始请求

        Returns:
            models.ModelObject: 模型信息
        """
        self.check_auth(request)
        name, model_type = self.resolve_model(model_id)
        return models.ModelObject(id=model_id, type=model_type, alias_of=name if name != model_id else None)

    def list_voices(
        self,
        request: Request,
    ) -> models.VoiceList:
        """获取可用的声音列表

        Args:
            request (Request):
                原始请求

        Returns:
            models.VoiceList: 声音列表
        """
        self.check_auth(request)
        backend = get_backend()
        speakers = backend.get_supported_speakers() if backend.get_model_type() == "custom_voice" else None
        speakers = speakers or QWEN_TTS_SPEAKERS
        data = [models.VoiceObject(id=s, type="speaker") for s in speakers]
        data += [models.VoiceObject(id=alias, type="alias", speaker=target) for alias, target in OPENAI_VOICE_ALIASES.items() if target in speakers]
        if VOICES_PATH.is_dir():
            for wav_path in sorted(VOICES_PATH.glob("*.wav")):
                registered = self.load_registered_voice(wav_path.stem)
                if registered is not None:
                    meta = registered[1]
                    data.append(models.VoiceObject(id=wav_path.stem, type="clone", ref_text=meta.get("ref_text"), created=meta.get("created")))
        return models.VoiceList(data=data)

    def create_voice(
        self,
        req: models.CreateVoiceRequest,
        request: Request,
    ) -> models.VoiceObject:
        """注册声音克隆参考音频, 注册后可在声音克隆模型中通过 `voice` 参数使用

        Args:
            req (models.CreateVoiceRequest):
                请求参数
            request (Request):
                原始请求

        Returns:
            models.VoiceObject: 注册的声音
        """
        self.check_auth(request)
        wav, sample_rate = decode_base64_audio(req.audio, "audio")
        if len(wav) / sample_rate > 300:
            raise OpenAIError(400, "Reference audio must be shorter than 300 seconds", param="audio")
        wav_path, meta_path = self.voice_paths(req.name)
        VOICES_PATH.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, wav, sample_rate, format="WAV", subtype="PCM_16")
        ref_text = req.ref_text.strip() if req.ref_text and req.ref_text.strip() else None
        created = int(time.time())
        meta_path.write_text(json.dumps({"ref_text": ref_text, "created": created}, ensure_ascii=False, indent=4), encoding="utf-8")
        logger.info("已注册声音 %s (%.2fs)", req.name, len(wav) / sample_rate)
        return models.VoiceObject(id=req.name, type="clone", ref_text=ref_text, created=created)

    def delete_voice(
        self,
        voice_id: str,
        request: Request,
    ) -> models.DeleteVoiceResponse:
        """删除注册的声音

        Args:
            voice_id (str):
                声音名称
            request (Request):
                原始请求

        Returns:
            models.DeleteVoiceResponse: 删除结果
        """
        self.check_auth(request)
        if self.load_registered_voice(voice_id) is None:
            raise OpenAIError(404, f"The voice `{voice_id}` does not exist", param="voice_id", code="voice_not_found")
        for path in self.voice_paths(voice_id):
            path.unlink(missing_ok=True)
        return models.DeleteVoiceResponse(id=voice_id, deleted=True)

    def interrupt(
        self,
        request: Request,
    ) -> models.InterruptResponse:
        """中断当前任务

        Args:
            request (Request):
                原始请求

        Returns:
            models.InterruptResponse: 中断响应
        """
        self.check_auth(request)
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
        uvicorn.run(
            self.app,
            host=server_name,
            port=port,
        )
