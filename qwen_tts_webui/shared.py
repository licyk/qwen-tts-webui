"""全局共享配置"""

from qwen_tts_webui.options import Options, OptionInfo
from qwen_tts_webui.config import CONFIG_PATH

options_templates: dict[str, OptionInfo] = {
    "api_type": OptionInfo("modelscope", "API 类型"),
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
