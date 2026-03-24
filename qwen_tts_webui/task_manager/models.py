"""任务管理相关的数据模型和枚举类型"""

import time
from enum import Enum
from typing import (
    Any,
    Optional,
)

from pydantic import (  # type: ignore
    BaseModel,
    Field,
)


class TaskType(str, Enum):
    """任务类型枚举"""

    VOICE_GENERATION = "voice_generation"  # 声音生成
    VOICE_DESIGN = "voice_design"  # 声音设计
    VOICE_CLONE = "voice_clone"  # 声音克隆


class TaskStatus(str, Enum):
    """任务状态枚举"""

    WAITING = "waiting"  # 等待中
    RUNNING = "running"  # 生成中
    SUCCESS = "success"  # 成功
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class TaskInfo(BaseModel):
    """任务信息模型"""

    task_id: str = Field(description="任务 ID（使用时间戳作为唯一标识）")
    task_type: TaskType = Field(description="任务类型")
    status: TaskStatus = Field(default=TaskStatus.WAITING, description="任务状态")
    submit_time: float = Field(description="提交时间（时间戳）")
    start_time: Optional[float] = Field(default=None, description="开始执行时间")
    end_time: Optional[float] = Field(default=None, description="完成时间")
    duration: Optional[float] = Field(default=None, description="耗时（秒）")
    
    # 任务参数
    params: dict[str, Any] = Field(default_factory=dict, description="任务参数")
    
    # 结果信息
    result: Optional[Any] = Field(default=None, description="任务结果")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    
    # 进度信息
    progress: int = Field(default=0, description="进度百分比（0-100）")
    message: str = Field(default="", description="进度消息")

    class Config:
        """Pydantic 配置"""
        use_enum_values = True

    @classmethod
    def create(
        cls,
        task_type: TaskType,
        params: dict[str, Any],
    ) -> "TaskInfo":
        """创建新任务实例
        
        Args:
            task_type (TaskType): 任务类型
            params (dict[str, Any]): 任务参数
            
        Returns:
            TaskInfo: 新创建的任务实例
        """
        kwargs = {
            "task_id": str(int(time.time() * 1000)),
            "task_type": task_type,
            "params": params,
            "submit_time": time.time(),
        }
        return cls(**kwargs)

    def mark_running(self) -> None:
        """标记任务为运行中"""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.progress = 0
        self.message = "任务开始执行"

    def mark_success(
        self,
        result: Any = None,
    ) -> None:
        """标记任务为成功
        
        Args:
            result (Any): 任务结果
        """
        self.status = TaskStatus.SUCCESS
        self.end_time = time.time()
        self.result = result
        self.progress = 100
        self.message = "任务完成"
        
        start = self.start_time
        end = self.end_time
        if start is not None and end is not None:
            self.duration = end - start

    def mark_failed(
        self,
        error_message: str,
    ) -> None:
        """标记任务为失败
        
        Args:
            error_message (str): 错误信息
        """
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.error_message = error_message
        self.progress = -1
        self.message = f"任务失败：{error_message}"
        
        start = self.start_time
        end = self.end_time
        if start is not None and end is not None:
            self.duration = end - start

    def mark_cancelled(self) -> None:
        """标记任务为已取消"""
        self.status = TaskStatus.CANCELLED
        self.end_time = time.time()
        self.progress = -1
        self.message = "任务已取消"

    def update_progress(
        self,
        progress: int,
        message: str = "",
    ) -> None:
        """更新任务进度
        
        Args:
            progress (int): 进度百分比（0-100）
            message (str): 进度消息
        """
        self.progress = progress
        self.message = message


class QueueItemResponse(BaseModel):
    """队列项响应模型（用于前端显示）"""

    task_id: str = Field(description="任务 ID")
    task_type: str = Field(description="任务类型")
    status: str = Field(description="任务状态")
    submit_time: str = Field(description="提交时间（格式化字符串）")
    start_time: Optional[str] = Field(default=None, description="开始时间（格式化字符串）")
    end_time: Optional[str] = Field(default=None, description="完成时间（格式化字符串）")
    duration: Optional[float] = Field(default=None, description="耗时（秒）")
    progress: int = Field(description="进度百分比")
    message: str = Field(description="进度消息")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    queue_position: Optional[int] = Field(default=None, description="队列位置（索引从 0 开始）")

    @classmethod
    def from_task_info(
        cls,
        task_info: TaskInfo,
    ) -> "QueueItemResponse":
        """从 TaskInfo 创建响应对象
        
        Args:
            task_info (TaskInfo): 任务信息对象
            
        Returns:
            QueueItemResponse: 响应对象
        """
        # 格式化时间为可读字符串
        def format_timestamp(timestamp: Optional[float]) -> Optional[str]:
            if timestamp is None:
                return None
            return time.strftime("%H:%M:%S", time.localtime(timestamp))

        # 获取队列位置（等待中的任务才有位置）
        queue_position = None
        if task_info.status == TaskStatus.WAITING:
            # 这里需要从队列管理器获取实际位置，暂时设为 None
            # 实际位置将在 get_queue_status 中设置
            pass
        
        dur = task_info.duration
        kwargs = {
            "task_id": task_info.task_id,
            "task_type": task_info.task_type.value if isinstance(task_info.task_type, TaskType) else task_info.task_type,
            "status": task_info.status.value if isinstance(task_info.status, TaskStatus) else task_info.status,
            "submit_time": format_timestamp(task_info.submit_time),
            "start_time": format_timestamp(task_info.start_time),
            "end_time": format_timestamp(task_info.end_time),
            "duration": round(dur, 2) if dur is not None else None,
            "progress": task_info.progress,
            "message": task_info.message,
            "error_message": task_info.error_message,
            "queue_position": queue_position,
        }
        return cls(**kwargs)
