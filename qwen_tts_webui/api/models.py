"""API 数据模型定义"""

from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


class CustomVoiceRequest(BaseModel):
    """声音生成请求模型"""

    model_name: str = Field(description="模型名称")
    text: str = Field(description="待合成文本")
    instruct: str = Field(default="", description="声音特征描述")
    speaker: Optional[str] = Field(default=None, description="发言人标识")
    language: Optional[str] = Field(default=None, description="语言标识")
    segment_gen: bool = Field(default=False, description="是否分段生成")
    do_sample: Optional[bool] = Field(default=None, description="是否使用采样")
    top_k: Optional[int] = Field(default=None, description="Top-k 采样参数")
    top_p: Optional[float] = Field(default=None, description="Top-p 采样参数")
    temperature: Optional[float] = Field(default=None, description="采样温度")
    repetition_penalty: Optional[float] = Field(default=None, description="重复惩罚系数")
    subtalker_dosample: Optional[bool] = Field(default=None, description="子说话者采样开关")
    subtalker_top_k: Optional[int] = Field(default=None, description="子说话者 Top-k 采样")
    subtalker_top_p: Optional[float] = Field(default=None, description="子说话者 Top-p 采样")
    subtalker_temperature: Optional[float] = Field(default=None, description="子说话者采样温度")
    max_new_tokens: Optional[int] = Field(default=None, description="最大生成 Token 数")


class VoiceDesignRequest(BaseModel):
    """声音设计请求模型"""

    model_name: str = Field(description="模型名称")
    text: str = Field(description="待合成文本")
    instruct: str = Field(description="声音特征描述")
    language: Optional[str] = Field(default=None, description="语言标识")
    segment_gen: bool = Field(default=False, description="是否分段生成")
    do_sample: Optional[bool] = Field(default=None, description="是否使用采样")
    top_k: Optional[int] = Field(default=None, description="Top-k 采样参数")
    top_p: Optional[float] = Field(default=None, description="Top-p 采样参数")
    temperature: Optional[float] = Field(default=None, description="采样温度")
    repetition_penalty: Optional[float] = Field(default=None, description="重复惩罚系数")
    subtalker_dosample: Optional[bool] = Field(default=None, description="子说话者采样开关")
    subtalker_top_k: Optional[int] = Field(default=None, description="子说话者 Top-k 采样")
    subtalker_top_p: Optional[float] = Field(default=None, description="子说话者 Top-p 采样")
    subtalker_temperature: Optional[float] = Field(default=None, description="子说话者采样温度")
    max_new_tokens: Optional[int] = Field(default=None, description="最大生成 Token 数")


class VoiceCloneRequest(BaseModel):
    """声音克隆请求模型"""

    model_name: str = Field(description="模型名称")
    text: str = Field(description="待合成文本")
    language: Optional[str] = Field(default=None, description="语言标识")
    ref_audio_base64: str = Field(description="参考音频文件的 Base64 编码")
    ref_text: Optional[str] = Field(default=None, description="参考音频文本描述")
    segment_gen: bool = Field(default=False, description="是否分段生成")
    do_sample: Optional[bool] = Field(default=None, description="是否使用采样")
    top_k: Optional[int] = Field(default=None, description="Top-k 采样参数")
    top_p: Optional[float] = Field(default=None, description="Top-p 采样参数")
    temperature: Optional[float] = Field(default=None, description="采样温度")
    repetition_penalty: Optional[float] = Field(default=None, description="重复惩罚系数")
    subtalker_dosample: Optional[bool] = Field(default=None, description="子说话者采样开关")
    subtalker_top_k: Optional[int] = Field(default=None, description="子说话者 Top-k 采样")
    subtalker_top_p: Optional[float] = Field(default=None, description="子说话者 Top-p 采样")
    subtalker_temperature: Optional[float] = Field(default=None, description="子说话者采样温度")
    max_new_tokens: Optional[int] = Field(default=None, description="最大生成 Token 数")


class TTSResponse(BaseModel):
    """TTS 响应模型"""

    audio_files_base64: list[str] = Field(description="生成的音频文件列表 (Base64 编码)")
    info: str = Field(description="生成信息")


class ModelInfo(BaseModel):
    """模型信息"""

    name: str = Field(description="模型名称")
    type: str = Field(description="模型类型: custom_voice, voice_design, voice_clone")


class ModelsResponse(BaseModel):
    """模型列表响应"""

    models: list[ModelInfo] = Field(description="可用模型列表")


class SpeakersResponse(BaseModel):
    """发言人列表响应"""

    speakers: list[str] = Field(description="支持的发言人列表")


class LanguagesResponse(BaseModel):
    """语言列表响应"""

    languages: list[str] = Field(description="支持的语言列表")


class OptionsResponse(BaseModel):
    """配置选项响应"""

    api_type: str
    device_map: str
    dtype: str
    attn_implementation: Optional[str]
    extra_custom_voice_models: list[str]
    extra_voice_design_models: list[str]
    extra_voice_clone_models: list[str]
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_dosample: bool
    subtalker_top_k: int
    subtalker_top_p: float
    subtalker_temperature: float
    max_new_tokens: int


class InterruptResponse(BaseModel):
    """中断响应"""

    message: str = Field(description="中断消息")
