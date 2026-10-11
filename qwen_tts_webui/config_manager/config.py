"""配置管理"""

import os
import logging
from pathlib import Path

LOGGER_NAME = os.getenv("QWEN_TTS_WEBUI_LOGGER_NAME", "Qwen TTS WebUI")
"""日志器名字"""

LOGGER_LEVEL = int(os.getenv("QWEN_TTS_WEBUI_LOGGER_LEVEL", str(logging.INFO)))
"""日志等级"""

LOGGER_COLOR = os.getenv("QWEN_TTS_WEBUI_LOGGER_COLOR") not in ["0", "False", "false", "None", "none", "null"]
"""日志颜色"""

ROOT_PATH = Path(__file__).parent.parent.parent
"""Qwen TTS WebUI 根目录"""

OUTPUT_PATH = ROOT_PATH / "outputs"
"""输出文件路径"""

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT_PATH / "config.json"
"""配置文件路径"""

VOICES_PATH = ROOT_PATH / "voices"
"""通过 API 注册的声音克隆参考音频的保存路径"""

QWEN_TTS_BASE_MODEL_LIST = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]
"""Qwen TTS 声音克隆模型"""

QWEN_TTS_CUSTOM_VOICE_MODEL_LIST = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]
"""Qwen TTS 声音生成模型"""

QWEN_TTS_VOICE_DESIGN_MODEL_LIST = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]
"""Qwen TTS 声音设计模型"""

OPENAI_MODEL_ALIASES = {
    "tts-1": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "tts-1-hd": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "gpt-4o-mini-tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
"""OpenAI 模型名称到 Qwen TTS 模型的映射, 方便直接使用 OpenAI 客户端的默认模型名称"""

OPENAI_VOICE_ALIASES = {
    "alloy": "vivian",
    "ash": "ryan",
    "ballad": "ryan",
    "cedar": "uncle_fu",
    "coral": "serena",
    "echo": "aiden",
    "fable": "aiden",
    "marin": "serena",
    "nova": "vivian",
    "onyx": "uncle_fu",
    "sage": "serena",
    "shimmer": "serena",
    "verse": "ryan",
}
"""OpenAI 发言人名称到 Qwen TTS 声音生成模型发言人的映射"""

ATTN_IMPL_LIST = [
    "eager",
    "sdpa",
    "flash_attention_2",
]
"""注意力加速方案"""

PYTORCH_PACKAGES = [
    "torch==2.10.0+cu130",
    "torchvision==0.25.0+cu130",
    "torchaudio==2.10.0+cu130",
]
"""安装所需的 PyTorch 包名列表"""

PYTORCH_MIRROR_NJU = "https://mirror.nju.edu.cn/pytorch/whl/cu130"
"""PyTorch 镜像源 (国内源)"""

PYTORCH_MIRROR = "https://download.pytorch.org/whl/cu130"
"""PyTorch 镜像源"""

PYPI_MIRROR_CERNET = "https://mirrors.cernet.edu.cn/pypi/web/simple"
"""PyPI 镜像源 (国内源)"""

PYPI_MIRROR = "https://pypi.python.org/simple"
"""PyPI 镜像源"""

UV_MINIMUN_VER = "0.9.9"
"""uv 的最低版本"""
