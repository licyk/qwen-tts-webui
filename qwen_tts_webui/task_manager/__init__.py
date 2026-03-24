"""任务管理器模块 - 支持队列管理、任务执行和状态跟踪"""

from qwen_tts_webui.task_manager.fifo_lock import FIFOLock
from qwen_tts_webui.task_manager.call_queue import queue_lock, wrap_queued_call, wrap_gradio_call
from qwen_tts_webui.task_manager.models import (
    TaskType,
    TaskStatus,
    TaskInfo,
    QueueItemResponse,
)
from qwen_tts_webui.task_manager.queue_manager import QueueManager

__all__ = [
    "FIFOLock",
    "queue_lock",
    "wrap_queued_call",
    "wrap_gradio_call",
    "TaskType",
    "TaskStatus",
    "TaskInfo",
    "QueueItemResponse",
    "QueueManager",
]