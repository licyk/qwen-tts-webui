"""任务调用队列管理"""

import functools
from typing import (
    Any,
    Callable,
)

from qwen_tts_webui.task_manager.fifo_lock import FIFOLock
from qwen_tts_webui.config_manager.shared import state

queue_lock = FIFOLock()
"""全局队列锁"""


def wrap_queued_call(
    func: Callable,
) -> Callable:
    """包装函数以支持队列锁

    Args:
        func (Callable): 要包装的函数

    Returns:
        Callable: 包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with queue_lock:
            res = func(*args, **kwargs)
        return res

    return wrapper


def wrap_gradio_call(
    func: Callable,
) -> Callable:
    """包装 Gradio 调用以支持状态管理和中断

    Args:
        func (Callable): 要包装的函数

    Returns:
        Callable: 包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        state.begin()
        try:
            res = func(*args, **kwargs)
        finally:
            state.end()
        return res

    return wrapper
