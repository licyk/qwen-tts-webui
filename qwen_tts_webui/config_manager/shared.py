"""全局状态管理"""

import threading

from qwen_tts_webui.config_manager.config import CONFIG_PATH
from qwen_tts_webui.config_manager.options import (
    OptionInfo,
    Options,
)
from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
)

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


class State:
    """全局状态类，用于管理任务运行状态"""

    def __init__(self) -> None:
        self.interrupted = False
        self.job = ""
        self._lock = threading.Lock()

    def interrupt(self) -> None:
        """中断当前任务"""
        logger.debug("已中断当前任务")
        self.interrupted = True

    def begin(self) -> None:
        """开始新任务"""
        logger.debug("开始新的任务")
        self.interrupted = False

    def end(self) -> None:
        """结束任务"""
        logger.debug("任务运行结束")
        self.job = ""


options_templates: dict[str, OptionInfo] = {
    "api_type": OptionInfo("modelscope", "下载模型的 API 类型"),
    "device_map": OptionInfo("auto", "推理设备"),
    "dtype": OptionInfo("torch.bfloat16", "推理精度"),
    "attn_implementation": OptionInfo(None, "加速方案"),
    "do_sample": OptionInfo(True, "是否使用采样"),
    "top_k": OptionInfo(50, "Top-k 采样参数"),
    "top_p": OptionInfo(1.0, "Top-p 采样参数"),
    "temperature": OptionInfo(0.9, "采样温度"),
    "repetition_penalty": OptionInfo(1.05, "重复惩罚系数"),
    "subtalker_dosample": OptionInfo(True, "子说话者采样开关"),
    "subtalker_top_k": OptionInfo(50, "子说话者 Top-k 采样"),
    "subtalker_top_p": OptionInfo(1.0, "子说话者 Top-p 采样"),
    "subtalker_temperature": OptionInfo(0.9, "子说话者采样温度"),
    "max_new_tokens": OptionInfo(2048, "最大生成 Token 数"),
}
"""配置项模板"""

opts = Options(options_templates)
"""全局配置实例"""

opts.load(CONFIG_PATH)

state = State()
"""全局状态实例"""
