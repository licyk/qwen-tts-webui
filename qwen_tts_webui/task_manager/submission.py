"""任务提交包装函数 - 将前端参数转换为队列任务并提交"""

from typing import Any, Optional

import gradio as gr

from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
)
from qwen_tts_webui.config_manager.shared import get_queue_manager
from qwen_tts_webui.task_manager.models import (
    TaskType,
    TaskInfo,
)
from qwen_tts_webui.task_manager.validators import validate_task_params

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


def add_voice_generation_task_to_queue(
    model_name: str,
    text: str,
    instruct: str,
    speaker: str,
    language: str,
    segment_gen: bool,
) -> tuple[list[str] | None, Any, Any]:
    """将声音生成任务添加到队列
    
    Args:
        model_name (str): 模型名称
        text (str): 合成文本
        instruct (str): 声音特征描述
        speaker (str): 说话人
        language (str): 语言
        segment_gen (bool): 分段生成开关
        
    Returns:
        tuple[list[str] | None, Any, Any]: 生成的音频路径（占位）, 发言人组件更新，语言组件更新
    """
    # 收集任务参数
    params = {
        "model_name": model_name,
        "text": text,
        "instruct": instruct,
        "speaker": speaker,
        "language": language,
        "segment_gen": segment_gen,
    }
    
    # 参数验证
    is_valid, msg = validate_task_params("voice_generation", params)
    if not is_valid:
        gr.Warning(f"参数验证失败：{msg}")
        return None, gr.update(), gr.update()
    
    # 创建任务信息
    task_info = TaskInfo.create(
        task_type=TaskType.VOICE_GENERATION,
        params=params,
    )
    
    # 获取队列管理器并添加任务
    queue_mgr = get_queue_manager()
    success, queue_msg = queue_mgr.add_task(task_info)
    
    if not success:
        gr.Warning(f"添加任务失败：{queue_msg}")
        return None, gr.update(), gr.update()
    
    # 显示成功提示
    gr.Info(f"任务已提交到队列，任务 ID: {task_info.task_id}，请前往「队列管理」标签页查看进度")
    
    # 返回空结果（实际结果由队列执行后返回）
    # 注意：不立即更新元数据选项，因为模型可能还未加载
    # 元数据选项将在队列执行完成后通过其他方式更新
    return (
        None,  # 音频列表将在队列执行完成后通过其他方式更新
        gr.update(),  # 保持原发言人选项不变
        gr.update(),  # 保持原语言选项不变
    )


def add_voice_design_task_to_queue(
    model_name: str,
    text: str,
    instruct: str,
    language: str,
    segment_gen: bool,
) -> tuple[list[str] | None, Any]:
    """将声音设计任务添加到队列
    
    Args:
        model_name (str): 模型名称
        text (str): 合成文本
        instruct (str): 声音特征描述
        language (str): 语言
        segment_gen (bool): 分段生成开关
        
    Returns:
        tuple[list[str] | None, Any]: 生成的音频路径（占位）, 语言组件更新
    """
    # 收集任务参数
    params = {
        "model_name": model_name,
        "text": text,
        "instruct": instruct,
        "language": language,
        "segment_gen": segment_gen,
    }
    
    # 参数验证
    is_valid, msg = validate_task_params("voice_design", params)
    if not is_valid:
        gr.Warning(f"参数验证失败：{msg}")
        return None, gr.update()
    
    # 创建任务信息
    task_info = TaskInfo.create(
        task_type=TaskType.VOICE_DESIGN,
        params=params,
    )
    
    # 获取队列管理器并添加任务
    queue_mgr = get_queue_manager()
    success, queue_msg = queue_mgr.add_task(task_info)
    
    if not success:
        gr.Warning(f"添加任务失败：{queue_msg}")
        return None, gr.update()
    
    # 显示成功提示
    gr.Info(f"任务已提交到队列，任务 ID: {task_info.task_id}，请前往「队列管理」标签页查看进度")
    
    # 返回空结果（实际结果由队列执行后返回）
    # 注意：不立即更新元数据选项，因为模型可能还未加载
    return (
        None,  # 音频列表将在队列执行完成后通过其他方式更新
        gr.update(),  # 保持原语言选项不变
    )


def add_voice_clone_task_to_queue(
    model_name: str,
    text: str,
    language: str,
    ref_audio: Optional[str],
    ref_text: str,
    segment_gen: bool,
) -> tuple[list[str] | None, Any]:
    """将声音克隆任务添加到队列
    
    Args:
        model_name (str): 模型名称
        text (str): 合成文本
        language (str): 语言
        ref_audio (Optional[str]): 参考音频路径
        ref_text (str): 参考音频文本描述
        segment_gen (bool): 分段生成开关
        
    Returns:
        tuple[list[str] | None, Any]: 生成的音频路径（占位）, 语言组件更新
    """
    # 检查参考音频是否存在
    if not ref_audio:
        gr.Warning("请先上传参考音频文件")
        return None, gr.update()
    
    # 收集任务参数
    params = {
        "model_name": model_name,
        "text": text,
        "language": language,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "segment_gen": segment_gen,
    }
    
    # 参数验证
    is_valid, msg = validate_task_params("voice_clone", params)
    if not is_valid:
        gr.Warning(f"参数验证失败：{msg}")
        return None, gr.update()
    
    # 创建任务信息
    task_info = TaskInfo.create(
        task_type=TaskType.VOICE_CLONE,
        params=params,
    )
    
    # 获取队列管理器并添加任务
    queue_mgr = get_queue_manager()
    success, queue_msg = queue_mgr.add_task(task_info)
    
    if not success:
        gr.Warning(f"添加任务失败：{queue_msg}")
        return None, gr.update()
    
    # 显示成功提示
    gr.Info(f"任务已提交到队列，任务 ID: {task_info.task_id}，请前往「队列管理」标签页查看进度")
    
    # 返回空结果（实际结果由队列执行后返回）
    # 注意：不立即更新元数据选项，因为模型可能还未加载
    return (
        None,  # 音频列表将在队列执行完成后通过其他方式更新
        gr.update(),  # 保持原语言选项不变
    )
