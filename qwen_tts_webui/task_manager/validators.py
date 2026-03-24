"""任务参数验证函数"""

from typing import (
    Any,
    Tuple,
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


def validate_voice_generation(params: dict[str, Any]) -> Tuple[bool, str]:
    """验证声音生成任务的参数
    
    Args:
        params (dict[str, Any]): 任务参数字典，应包含：
            - model_name (str): 模型名称
            - text (str): 合成文本
            - speaker (str): 说话人
            - language (str): 语言
            - instruct (str, optional): 声音特征描述
            
    Returns:
        Tuple[bool, str]: (是否验证通过，消息)
    """
    # 检查必需参数
    required_fields = ["model_name", "text", "speaker", "language"]
    for field in required_fields:
        if field not in params:
            error_msg = f"缺少必需参数：{field}"
            logger.warning(error_msg)
            return False, error_msg
    
    # 验证文本内容
    text = params.get("text", "").strip()
    if not text:
        error_msg = "合成文本不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证模型名称
    model_name = params.get("model_name", "").strip()
    if not model_name:
        error_msg = "模型名称不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证说话人
    speaker = params.get("speaker", "").strip()
    if not speaker:
        error_msg = "说话人不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 可选参数验证：instruct（声音特征描述）
    instruct = params.get("instruct", "").strip()
    if instruct and len(instruct) > 500:
        error_msg = "声音特征描述过长（最大长度：500 字符）"
        logger.warning(error_msg)
        return False, error_msg
    
    logger.debug("声音生成参数验证通过")
    return True, "参数验证通过"


def validate_voice_design(params: dict[str, Any]) -> Tuple[bool, str]:
    """验证声音设计任务的参数
    
    Args:
        params (dict[str, Any]): 任务参数字典，应包含：
            - model_name (str): 模型名称
            - text (str): 合成文本
            - language (str): 语言
            - instruct (str): 声音特征描述
            
    Returns:
        Tuple[bool, str]: (是否验证通过，消息)
    """
    # 检查必需参数
    required_fields = ["model_name", "text", "language", "instruct"]
    for field in required_fields:
        if field not in params:
            error_msg = f"缺少必需参数：{field}"
            logger.warning(error_msg)
            return False, error_msg
    
    # 验证文本内容
    text = params.get("text", "").strip()
    if not text:
        error_msg = "合成文本不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证模型名称
    model_name = params.get("model_name", "").strip()
    if not model_name:
        error_msg = "模型名称不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证声音特征描述（必需）
    instruct = params.get("instruct", "").strip()
    if not instruct:
        error_msg = "声音特征描述不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    if len(instruct) > 500:
        error_msg = "声音特征描述过长（最大长度：500 字符）"
        logger.warning(error_msg)
        return False, error_msg
    
    logger.debug("声音设计参数验证通过")
    return True, "参数验证通过"


def validate_voice_clone(params: dict[str, Any]) -> Tuple[bool, str]:
    """验证声音克隆任务的参数
    
    Args:
        params (dict[str, Any]): 任务参数字典，应包含：
            - model_name (str): 模型名称
            - text (str): 合成文本
            - language (str): 语言
            - ref_audio (str): 参考音频路径
            - ref_text (str, optional): 参考音频文本描述
            
    Returns:
        Tuple[bool, str]: (是否验证通过，消息)
    """
    # 检查必需参数
    required_fields = ["model_name", "text", "language", "ref_audio"]
    for field in required_fields:
        if field not in params:
            error_msg = f"缺少必需参数：{field}"
            logger.warning(error_msg)
            return False, error_msg
    
    # 验证文本内容
    text = params.get("text", "").strip()
    if not text:
        error_msg = "合成文本不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证模型名称
    model_name = params.get("model_name", "").strip()
    if not model_name:
        error_msg = "模型名称不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证参考音频
    ref_audio = params.get("ref_audio", "").strip()
    if not ref_audio:
        error_msg = "参考音频文件不能为空"
        logger.warning(error_msg)
        return False, error_msg
    
    # 验证参考音频路径是否存在（可选，因为可能是前端上传的临时文件）
    from pathlib import Path
    ref_audio_path = Path(ref_audio)
    if not ref_audio_path.exists():
        error_msg = f"参考音频文件不存在：{ref_audio}"
        logger.warning(error_msg)
        return False, error_msg
    
    # 可选参数验证：ref_text（参考音频文本描述）
    ref_text = params.get("ref_text", "").strip()
    if ref_text and len(ref_text) > 500:
        error_msg = "参考音频文本描述过长（最大长度：500 字符）"
        logger.warning(error_msg)
        return False, error_msg
    
    logger.debug("声音克隆参数验证通过")
    return True, "参数验证通过"


def validate_task_params(
    task_type: str,
    params: dict[str, Any],
) -> Tuple[bool, str]:
    """统一的参数验证入口函数
    
    Args:
        task_type (str): 任务类型，应为以下之一：
            - "voice_generation" (声音生成)
            - "voice_design" (声音设计)
            - "voice_clone" (声音克隆)
        params (dict[str, Any]): 任务参数字典
            
    Returns:
        Tuple[bool, str]: (是否验证通过，消息)
    """
    if task_type == "voice_generation":
        return validate_voice_generation(params)
    elif task_type == "voice_design":
        return validate_voice_design(params)
    elif task_type == "voice_clone":
        return validate_voice_clone(params)
    else:
        error_msg = f"未知的任务类型：{task_type}"
        logger.warning(error_msg)
        return False, error_msg
