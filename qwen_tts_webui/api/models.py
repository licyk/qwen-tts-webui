"""API 数据模型定义 (兼容 OpenAI Audio API 格式)"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
"""音频输出格式"""

ModelType = Literal["custom_voice", "voice_design", "voice_clone"]
"""模型类型"""


class SpeechRequest(BaseModel):
    """语音合成请求, 与 OpenAI `POST /v1/audio/speech` 兼容, 并扩展了 Qwen TTS 的参数"""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(description="模型名称, 可以为 Qwen TTS 模型名称或者 OpenAI 模型别名 (tts-1 / tts-1-hd / gpt-4o-mini-tts)")
    input: str = Field(min_length=1, description="待合成文本")
    voice: str | None = Field(
        default=None,
        description="声音生成模型: 发言人名称或 OpenAI 发言人别名; 声音克隆模型: 通过 `POST /v1/audio/voices` 注册的声音名称; 声音设计模型: 忽略",
    )
    instructions: str | None = Field(default=None, description="声音生成模型: 声音特征描述; 声音设计模型: 声音描述 (必填)")
    response_format: ResponseFormat = Field(default="mp3", description="音频输出格式")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速, 范围为 0.25 ~ 4.0")
    stream_format: Literal["audio", "sse"] = Field(
        default="audio",
        description="`audio`: 直接返回音频 (pcm / wav 格式在生成过程中边生成边返回); `sse`: 以 Server-Sent Events 的方式边生成边返回音频片段",
    )

    # Qwen TTS 扩展参数
    language: str | None = Field(default=None, description="[扩展] 语言, 默认自动检测")
    ref_audio: str | None = Field(default=None, description="[扩展] 声音克隆参考音频 (Base64 或者 data URI), 设置后忽略 `voice`")
    ref_text: str | None = Field(default=None, description="[扩展] 参考音频对应的文本, 为空时仅使用说话人特征进行克隆")
    segment: bool = Field(default=False, description="[扩展] 按行切分文本并批量生成, 结果按顺序拼接为一段音频")
    seed: int | None = Field(default=None, description="[扩展] 随机种子")
    do_sample: bool | None = Field(default=None, description="[扩展] 是否使用采样")
    top_k: int | None = Field(default=None, ge=0, description="[扩展] Top-k 采样参数")
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, description="[扩展] Top-p 采样参数")
    temperature: float | None = Field(default=None, ge=0.0, description="[扩展] 采样温度")
    repetition_penalty: float | None = Field(default=None, gt=0.0, description="[扩展] 重复惩罚系数")
    subtalker_dosample: bool | None = Field(default=None, description="[扩展] 子说话者采样开关")
    subtalker_top_k: int | None = Field(default=None, ge=0, description="[扩展] 子说话者 Top-k 采样")
    subtalker_top_p: float | None = Field(default=None, gt=0.0, le=1.0, description="[扩展] 子说话者 Top-p 采样")
    subtalker_temperature: float | None = Field(default=None, ge=0.0, description="[扩展] 子说话者采样温度")
    max_new_tokens: int | None = Field(default=None, ge=1, description="[扩展] 最大生成 Token 数 (每个 Token 对应 80ms 音频)")


class ModelObject(BaseModel):
    """模型信息"""

    id: str = Field(description="模型名称")
    object: Literal["model"] = "model"
    created: int = Field(default=0, description="创建时间")
    owned_by: str = Field(default="qwen", description="模型所有者")
    type: ModelType = Field(description="[扩展] 模型类型")
    alias_of: str | None = Field(default=None, description="[扩展] 别名对应的模型")


class ModelList(BaseModel):
    """模型列表"""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class VoiceObject(BaseModel):
    """声音信息"""

    id: str = Field(description="声音名称, 用于 `voice` 参数")
    object: Literal["voice"] = "voice"
    type: Literal["speaker", "alias", "clone"] = Field(description="speaker: 声音生成模型的内置发言人; alias: OpenAI 发言人别名; clone: 注册的声音克隆参考音频")
    speaker: str | None = Field(default=None, description="别名对应的发言人")
    ref_text: str | None = Field(default=None, description="声音克隆参考音频对应的文本")
    created: int | None = Field(default=None, description="注册时间")


class VoiceList(BaseModel):
    """声音列表"""

    object: Literal["list"] = "list"
    data: list[VoiceObject]


class CreateVoiceRequest(BaseModel):
    """注册声音克隆参考音频"""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", description="声音名称, 只能包含字母, 数字, `_`, `.`, `-`")
    audio: str = Field(min_length=1, description="参考音频 (Base64 或者 data URI)")
    ref_text: str | None = Field(default=None, description="参考音频对应的文本, 为空时仅使用说话人特征进行克隆")


class DeleteVoiceResponse(BaseModel):
    """删除声音响应"""

    id: str
    object: Literal["voice"] = "voice"
    deleted: bool


class InterruptResponse(BaseModel):
    """中断响应"""

    message: str = Field(description="中断消息")


class ErrorDetail(BaseModel):
    """错误信息"""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """错误响应 (与 OpenAI 错误格式兼容)"""

    error: ErrorDetail
