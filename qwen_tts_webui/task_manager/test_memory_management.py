"""显存管理性能测试脚本 - 验证 cleanup_models 的调用和显存监控"""

import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# 设置 PYTHONPATH
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "core"))

# Mock Backend and Memory Manager
mock_backend = MagicMock()
mock_memory_mgr = MagicMock()

# Setup get_backend and MemoryManager patches
patcher_backend = patch('qwen_tts_webui.task_manager.task_executor.get_backend', return_value=mock_backend)
patcher_backend.start()

from qwen_tts_webui.task_manager.models import TaskType, TaskStatus, TaskInfo
from qwen_tts_webui.task_manager.queue_manager import QueueManager
from qwen_tts_webui.task_manager.task_executor import TaskExecutor

class TestMemoryManagement(unittest.TestCase):
    """显存管理测试"""

    def setUp(self):
        self.queue_mgr = QueueManager()
        mock_backend.reset_mock()
        
        # 补丁 Path.exists
        self.path_exists_patcher = patch.object(Path, 'exists', return_value=True)
        self.path_exists_patcher.start()

    def tearDown(self):
        self.queue_mgr.shutdown()
        self.path_exists_patcher.stop()

    def test_cleanup_called_after_each_task(self):
        """验证每个任务执行后都会调用 cleanup_models"""
        print("\n--- [测试] 验证显存清理调用 ---")
        
        # 补丁 cleanup_models 以追踪调用
        with patch('qwen_tts_webui.task_manager.task_executor.cleanup_models') as mock_cleanup:
            params = {
                "text": "测试文本", 
                "speaker": "default", 
                "language": "zh", 
                "model_name": "test_model"
            }
            
            # 提交 3 个任务
            tasks = []
            for i in range(3):
                t = TaskInfo.create(TaskType.VOICE_GENERATION, params)
                tasks.append(t)
                self.queue_mgr.add_task(t)
                time.sleep(0.01)

            # 等待所有任务完成
            start_wait = time.time()
            while any(t.status != TaskStatus.SUCCESS for t in tasks) and time.time() - start_wait < 10:
                time.sleep(0.1)

            # 验证 cleanup_models 被调用了 3 次（每个任务结束各一次）
            # 注意：TaskExecutor.execute_task 里的 finally 块会调用 cleanup_models()
            print(f"cleanup_models 被调用次数: {mock_cleanup.call_count}")
            self.assertGreaterEqual(mock_cleanup.call_count, 3)
            print("✓ 每个任务执行后成功触发了显存清理逻辑")

if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        patcher_backend.stop()
