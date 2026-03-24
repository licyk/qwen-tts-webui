"""任务执行器 - 负责执行队列中的具体任务"""

import time
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
)
from qwen_tts_webui.task_manager.models import (
    TaskType,
    TaskInfo,
)
from qwen_tts_webui.task_manager.validators import validate_task_params
from qwen_tts_webui.config_manager.shared import get_backend
from qwen_tts_webui.backend.memory_manager import (
    cleanup_models,
    OutOfMemoryError,
)

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


class TaskExecutor:
    """任务执行器 - 负责执行队列中的具体任务"""
    
    def __init__(
        self,
    ) -> None:
        """初始化任务执行器"""
        logger.info("任务执行器初始化完成")
    
    def execute_task(
        self,
        task_info: TaskInfo,
    ) -> None:
        """执行单个任务
        
        Args:
            task_info (TaskInfo): 任务信息对象
        """
        logger.info("开始执行任务 %s（类型：%s）", task_info.task_id, task_info.task_type)
        
        # 1. 参数验证
        validation_success, validation_msg = validate_task_params(
            task_info.task_type,
            task_info.params,
        )
        if not validation_success:
            error_msg = f"参数验证失败：{validation_msg}"
            logger.error(error_msg)
            task_info.mark_failed(error_msg)
            return
        
        try:
            # 2. 加载模型（检查是否需要切换）
            model_name = task_info.params.get("model_name")
            if model_name:
                logger.info("任务 %s: 正在加载模型 %s...", task_info.task_id, model_name)
                backend = get_backend()
                
                # 从配置中获取加载参数
                from qwen_tts_webui.config_manager.shared import opts
                import torch
                
                try:
                    backend.load_model(
                        model_name=model_name,
                        api_type=opts.api_type,
                        device_map=opts.device_map,
                        dtype=getattr(torch, opts.dtype.split(".")[-1]),
                        attn_implementation=opts.attn_implementation,
                    )
                    logger.info("任务 %s: 模型 %s 加载成功", task_info.task_id, model_name)
                except Exception as e:
                    error_msg = f"模型加载失败 ({model_name}): {str(e)}"
                    logger.error("任务 %s: %s", task_info.task_id, error_msg, exc_info=True)
                    task_info.mark_failed(error_msg)
                    return
            
            # 3. 根据任务类型调用对应的生成函数
            logger.info("任务 %s: 正在执行生成逻辑...", task_info.task_id)
            result_path = self._execute_by_task_type(task_info)
            
            if not result_path or not result_path.exists():
                error_msg = "生成失败：未生成有效的音频文件"
                logger.error("任务 %s: %s", task_info.task_id, error_msg)
                task_info.mark_failed(error_msg)
                return

            # 4. 标记任务成功
            task_info.mark_success(result=str(result_path))
            logger.info("任务 %s: 执行成功，音频路径：%s", task_info.task_id, result_path)
            
        except OutOfMemoryError as e:
            # 显存不足错误处理
            error_msg = f"显存不足 (OOM)：{str(e)}"
            logger.error("任务 %s: %s", task_info.task_id, error_msg)
            task_info.mark_failed(error_msg)
            # 清理模型释放显存
            cleanup_models()
            
        except Exception as e:
            # 其他异常处理
            error_msg = f"执行过程中发生未知错误：{str(e)}"
            logger.error("任务 %s: %s", task_info.task_id, error_msg, exc_info=True)
            task_info.mark_failed(error_msg)
            # 清理模型释放显存
            cleanup_models()
        
        finally:
            # 5. 任务完成后清理显存
            cleanup_models()
            logger.info("任务 %s 执行结束，已清理显存", task_info.task_id)
    
    def _execute_by_task_type(
        self,
        task_info: TaskInfo,
    ) -> Optional[Path]:
        """根据任务类型执行对应的生成函数
        
        Args:
            task_info (TaskInfo): 任务信息对象
            
        Returns:
            Optional[Path]: 生成的音频文件路径
            
        Raises:
            ValueError: 未知的任务类型
            Exception: 生成过程中的异常
        """
        backend = get_backend()
        
        if task_info.task_type == TaskType.VOICE_GENERATION:
            return self._execute_voice_generation(backend, task_info.params)
        elif task_info.task_type == TaskType.VOICE_DESIGN:
            return self._execute_voice_design(backend, task_info.params)
        elif task_info.task_type == TaskType.VOICE_CLONE:
            return self._execute_voice_clone(backend, task_info.params)
        else:
            raise ValueError(f"未知的任务类型：{task_info.task_type}")
    
    def _execute_voice_generation(
        self,
        backend: Any,
        params: dict[str, Any],
    ) -> Path:
        """执行声音生成任务
        
        Args:
            backend: QwenTTSBackend 实例
            params (dict[str, Any]): 任务参数
            
        Returns:
            Path: 生成的音频文件路径
        """
        
        logger.info("执行声音生成任务")
        
        # 提取参数
        text = params.get("text", "")
        speaker = params.get("speaker", "default")
        language = params.get("language", "auto")
        instruct = params.get("instruct", None)
        
        # 处理 "default" 说话人：替换为第一个可用的说话人
        if speaker == "default":
            supported_speakers = backend.get_supported_speakers() or []
            if supported_speakers:
                speaker = supported_speakers[0]
                logger.info("使用默认说话人：%s", speaker)
            else:
                logger.warning("没有可用的说话人，将使用模型默认值")
        
        # 调用后端生成函数
        result_path = backend.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            **self._get_common_opts(),
        )
        
        logger.info("声音生成任务完成：%s", result_path)
        return result_path
    
    def _execute_voice_design(
        self,
        backend: Any,
        params: dict[str, Any],
    ) -> Path:
        """执行声音设计任务
        
        Args:
            backend: QwenTTSBackend 实例
            params (dict[str, Any]): 任务参数
            
        Returns:
            Path: 生成的音频文件路径
        """
        
        logger.info("执行声音设计任务")
        
        # 提取参数
        text = params.get("text", "")
        instruct = params.get("instruct", "")
        language = params.get("language", "auto")
        
        # 调用后端生成函数
        result_path = backend.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            **self._get_common_opts(),
        )
        
        logger.info("声音设计任务完成：%s", result_path)
        return result_path
    
    def _execute_voice_clone(
        self,
        backend: Any,
        params: dict[str, Any],
    ) -> Path:
        """执行声音克隆任务
        
        Args:
            backend: QwenTTSBackend 实例
            params (dict[str, Any]): 任务参数
            
        Returns:
            Path: 生成的音频文件路径
        """
        
        logger.info("执行声音克隆任务")
        
        # 提取参数
        text = params.get("text", "")
        language = params.get("language", "auto")
        ref_audio = params.get("ref_audio", "")
        ref_text = params.get("ref_text", None)
        
        # 判断是否使用 x_vector_only_mode
        x_vector_only = (ref_text is None or ref_text.strip() == "")
        
        # 调用后端生成函数
        result_path = backend.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=Path(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            **self._get_common_opts(),
        )
        
        logger.info("声音克隆任务完成：%s", result_path)
        return result_path

    def _get_common_opts(self) -> dict[str, Any]:
        """获取通用的生成参数"""
        from qwen_tts_webui.config_manager.shared import opts
        
        return {
            "do_sample": opts.do_sample,
            "top_k": opts.top_k,
            "top_p": opts.top_p,
            "temperature": opts.temperature,
            "repetition_penalty": opts.repetition_penalty,
            "subtalker_dosample": opts.subtalker_dosample,
            "subtalker_top_k": opts.subtalker_top_k,
            "subtalker_top_p": opts.subtalker_top_p,
            "subtalker_temperature": opts.subtalker_temperature,
            "max_new_tokens": opts.max_new_tokens,
        }
