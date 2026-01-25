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

ROOT_PATH = Path(__file__).parent.parent
"""Qwen TTS WebUI 根目录"""

MODEL_PATH = ROOT_PATH / "models"
"""Qwen TTS WebUI 模型路径"""

OUTPUT_PATH = ROOT_PATH / "outputs"
"""输出文件路径"""
