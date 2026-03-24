"""队列管理器核心实现"""

import threading
import time
from typing import (
    Optional,
)
from collections import deque

from qwen_tts_webui.logger import get_logger  # type: ignore
from qwen_tts_webui.config_manager.config import (  # type: ignore
    LOGGER_LEVEL,
    LOGGER_COLOR,
)
from qwen_tts_webui.task_manager.models import (  # type: ignore
    TaskType,
    TaskStatus,
    TaskInfo,
)
from qwen_tts_webui.task_manager.fifo_lock import FIFOLock  # type: ignore
from qwen_tts_webui.task_manager.task_executor import TaskExecutor  # type: ignore

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


class QueueManager:
    """队列管理器 - 负责任务的添加、调度、执行和状态管理"""

    MAX_QUEUE_SIZE = 50  # 最大队列容量

    def __init__(
        self,
    ) -> None:
        """初始化队列管理器"""
        self._queue: deque[TaskInfo] = deque()
        self._queue_lock = FIFOLock()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_flag = False
        self._current_task: Optional[TaskInfo] = None
        self._executor = TaskExecutor()  # 初始化任务执行器

        logger.info("队列管理器初始化完成")

    def add_task(
        self,
        task_info: TaskInfo,
    ) -> tuple[bool, str]:
        """添加任务到队列
        
        Args:
            task_info (TaskInfo): 任务信息对象
            
        Returns:
            tuple[bool, str]: (是否成功，消息)
        """
        with self._queue_lock:
            # 检查队列容量
            if len(self._queue) >= self.MAX_QUEUE_SIZE:
                error_msg = f"队列已满（最大容量：{self.MAX_QUEUE_SIZE}）"
                logger.warning(error_msg)
                return False, error_msg

            # 添加任务到队列
            self._queue.append(task_info)
            logger.info(
                "任务 %s 已添加到队列，当前队列长度：%d",
                task_info.task_id,
                len(self._queue),
            )

            # 确保工作线程正在运行
            self._ensure_worker_running()

            return True, "任务已成功添加到队列"

    def cancel_task(
        self,
        task_id: str,
    ) -> tuple[bool, str]:
        """取消等待中的任务
        
        Args:
            task_id (str): 任务 ID
            
        Returns:
            tuple[bool, str]: (是否成功，消息)
        """
        with self._queue_lock:
            # 查找任务
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    # 检查任务状态
                    if task.status == TaskStatus.WAITING:
                        task.mark_cancelled()
                        logger.info("任务 %s 已取消", task_id)
                        return True, "任务已成功取消"
                    else:
                        error_msg = f"任务 {task_id} 正在执行中，无法取消"
                        logger.warning(error_msg)
                        return False, error_msg

            error_msg = f"未找到任务 {task_id}"
            logger.warning(error_msg)
            return False, error_msg

    def move_task_up(
        self,
        task_id: str,
    ) -> tuple[bool, str]:
        """上移任务位置（仅适用于等待中的任务）
        
        Args:
            task_id (str): 任务 ID
            
        Returns:
            tuple[bool, str]: (是否成功，消息)
        """
        with self._queue_lock:
            # 查找任务索引
            target_index = -1
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    target_index = i
                    break

            if target_index == -1:
                error_msg = f"未找到任务 {task_id}"
                logger.warning(error_msg)
                return False, error_msg

            # 检查是否是第一个任务
            if target_index == 0:
                error_msg = "任务已在队首，无法继续上移"
                logger.info(error_msg)
                return False, error_msg

            # 检查任务状态
            if self._queue[target_index].status != TaskStatus.WAITING:
                error_msg = f"任务 {task_id} 正在执行中，无法移动"
                logger.warning(error_msg)
                return False, error_msg

            # 交换位置 (解决 Pyre2 不支持 deque 索引赋值的问题)
            task_to_move = self._queue[target_index]
            self._queue.remove(task_to_move)
            self._queue.insert(target_index - 1, task_to_move)
            logger.info("任务 %s 已上移", task_id)
            return True, "任务已上移"

    def move_task_down(
        self,
        task_id: str,
    ) -> tuple[bool, str]:
        """下移任务位置（仅适用于等待中的任务）
        
        Args:
            task_id (str): 任务 ID
            
        Returns:
            tuple[bool, str]: (是否成功，消息)
        """
        with self._queue_lock:
            # 查找任务索引
            target_index = -1
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    target_index = i
                    break

            if target_index == -1:
                error_msg = f"未找到任务 {task_id}"
                logger.warning(error_msg)
                return False, error_msg

            # 检查是否是最后一个任务
            if target_index == len(self._queue) - 1:
                error_msg = "任务已在队尾，无法继续下移"
                logger.info(error_msg)
                return False, error_msg

            # 检查任务状态
            if self._queue[target_index].status != TaskStatus.WAITING:
                error_msg = f"任务 {task_id} 正在执行中，无法移动"
                logger.warning(error_msg)
                return False, error_msg

            # 交换位置 (解决 Pyre2 不支持 deque 索引赋值的问题)
            task_to_move = self._queue[target_index]
            self._queue.remove(task_to_move)
            self._queue.insert(target_index + 1, task_to_move)
            logger.info("任务 %s 已下移", task_id)
            return True, "任务已下移"

    def get_queue_status(
        self,
    ) -> list:
        """获取队列状态
        
        Returns:
            list: 队列中所有任务的 QueueItemResponse 列表（包含队列位置）
        """
        from qwen_tts_webui.task_manager.models import QueueItemResponse
        
        with self._queue_lock:
            result = []
            for idx, task in enumerate(list(self._queue)):
                # 为等待中的任务设置队列位置
                response = QueueItemResponse.from_task_info(task)
                if task.status == TaskStatus.WAITING:
                    response.queue_position = idx
                result.append(response)
            return result

    def _ensure_worker_running(
        self,
    ) -> None:
        """确保工作线程正在运行"""
        worker = self._worker_thread
        if worker is None or not worker.is_alive():
            logger.info("启动队列工作线程")
            self._shutdown_flag = False
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
            )
            self._worker_thread.start()

    def _worker_loop(
        self,
    ) -> None:
        """后台工作线程循环，处理队列中的任务"""
        logger.info("队列工作线程启动")

        while not self._shutdown_flag:
            task_to_process: Optional[TaskInfo] = None

            # 从队列中获取下一个任务
            with self._queue_lock:
                if len(self._queue) > 0:
                    # 获取第一个等待中的任务
                    for task in self._queue:
                        if task.status == TaskStatus.WAITING:
                            task_to_process = task
                            task.mark_running()
                            self._current_task = task
                            logger.info(
                                "开始处理任务 %s（类型：%s）",
                                task.task_id,
                                task.task_type,
                            )
                            break

            # 如果没有任务需要处理，退出循环
            if task_to_process is None:
                logger.debug("队列为空，工作线程进入等待")
                break

            # 执行任务（使用 TaskExecutor）
            try:
                logger.info("任务 %s: 开始执行 (由 TaskExecutor 处理)...", task_to_process.task_id)
                
                # 使用任务执行器执行任务
                self._executor.execute_task(task_to_process)
                
                # 任务执行完成后，状态已经被 execute_task 更新
                if task_to_process.status == TaskStatus.SUCCESS:
                    logger.info("任务 %s: [成功] 耗时：%.2f 秒", task_to_process.task_id, task_to_process.duration)
                elif task_to_process.status == TaskStatus.FAILED:
                    logger.error("任务 %s: [失败] 错误信息：%s", task_to_process.task_id, task_to_process.error_message)
                elif task_to_process.status == TaskStatus.CANCELLED:
                    logger.info("任务 %s: [已取消]", task_to_process.task_id)

            except Exception as e:
                logger.error("任务 %s: 执行器抛出异常：%s", task_to_process.task_id, e, exc_info=True)
                task_to_process.mark_failed(f"系统异常：{str(e)}")

            finally:
                # 清理当前任务引用
                with self._queue_lock:
                    self._current_task = None

        logger.info("队列工作线程停止")

    def shutdown(
        self,
    ) -> None:
        """关闭队列管理器"""
        logger.info("正在关闭队列管理器...")
        self._shutdown_flag = True

        # 等待工作线程结束
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=5)
            if worker.is_alive():
                logger.warning("工作线程未能正常关闭")

        logger.info("队列管理器已关闭")

    def set_executor(
        self,
        executor,
    ) -> None:
        """设置任务执行器（用于自定义执行器）
        
        Args:
            executor: 任务执行器对象
        """
        self._executor = executor
        logger.info("任务执行器已设置")
