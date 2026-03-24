"""全链路集成测试脚本 - 验证从任务提交到执行完成的完整流程"""

import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# 设置 PYTHONPATH
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "core"))

# removed module level mocks

from qwen_tts_webui.task_manager.models import TaskType, TaskStatus, TaskInfo
from qwen_tts_webui.task_manager.queue_manager import QueueManager
from qwen_tts_webui.task_manager.task_executor import TaskExecutor

class TestIntegrationFull(unittest.TestCase):
    """全链路集成测试"""

    def setUp(self):
        """测试准备"""
        self.mock_backend = MagicMock()
        patcher = patch('qwen_tts_webui.task_manager.task_executor.get_backend', return_value=self.mock_backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        
        self.queue_mgr = QueueManager()
        
        # 重置 Mock 状态
        self.mock_backend.reset_mock()
        self.mock_backend.generate_custom_voice.side_effect = None
        self.mock_backend.generate_custom_voice.return_value = Path("mock_gen.wav")
        self.mock_backend.get_supported_speakers.return_value = ["default_speaker"]
        
        # 补丁 Path.exists 确保返回 True
        self.path_exists_patcher = patch.object(Path, 'exists', return_value=True)
        self.path_exists_patcher.start()

    def tearDown(self):
        """测试清理"""
        self.queue_mgr.shutdown()
        self.path_exists_patcher.stop()

    def _create_task(self, name, model="m1"):
        """创建测试任务"""
        params = {
            "text": f"测试文本-{name}", 
            "speaker": "default", 
            "language": "zh", 
            "model_name": model
        }
        # 使用唯一 ID
        task = TaskInfo.create(TaskType.VOICE_GENERATION, params)
        time.sleep(0.02) # 强行确保时间戳不同
        return task

    def _wait_for_status(self, task, status, timeout=5):
        """等待任务状态改变"""
        start = time.time()
        while task.status != status and time.time() - start < timeout:
            time.sleep(0.1)
        return task.status == status

    def test_full_flow_success(self):
        """测试完整成功流程"""
        print("\n--- [测试] 完整成功流程 ---")
        task = self._create_task("success")
        
        success, msg = self.queue_mgr.add_task(task)
        self.assertTrue(success, f"添加任务失败: {msg}")
        
        completed = self._wait_for_status(task, TaskStatus.SUCCESS)
        self.assertTrue(completed, f"任务未在预期时间内成功。当前状态: {task.status}")
        self.assertIsNotNone(task.result)
        print(f"[OK] 任务成功完成: {task.task_id}")

    def test_error_recovery_flow(self):
        """测试错误恢复流程：失败的任务不应阻塞后续任务"""
        print("\n--- [测试] 错误恢复流程 ---")
        
        # 1. 第一个任务注定失败
        self.mock_backend.generate_custom_voice.side_effect = Exception("故意抛出的生成异常")
        task1 = self._create_task("fail_first")
        self.queue_mgr.add_task(task1)
        
        # 2. 等待第一个任务失败
        f1 = self._wait_for_status(task1, TaskStatus.FAILED)
        self.assertTrue(f1, "第一个任务没能按预期失败")
        
        # 3. 恢复正常，提交第二个任务
        self.mock_backend.generate_custom_voice.side_effect = None
        self.mock_backend.generate_custom_voice.return_value = Path("recovered.wav")
        task2 = self._create_task("success_second")
        self.queue_mgr.add_task(task2)
        
        # 4. 等待第二个任务成功
        f2 = self._wait_for_status(task2, TaskStatus.SUCCESS)
        self.assertTrue(f2, f"第二个任务没能成功。状态: {task2.status}")
        
        self.assertEqual(task1.status, TaskStatus.FAILED)
        self.assertEqual(task2.status, TaskStatus.SUCCESS)
        print("[OK] 错误恢复验证通过：失败任务后，后续任务正常执行")

    def test_queue_operations_during_execution(self):
        """测试执行期间的队列操作：上移与取消"""
        print("\n--- [测试] 执行期间的队列操作 ---")
        
        # 模拟耗时任务
        def long_gen(*args, **kwargs):
            time.sleep(2.0)
            return Path("long.wav")
        self.mock_backend.generate_custom_voice.side_effect = long_gen
        
        # 连续提交三个任务
        t_run = self._create_task("run")
        t_wait1 = self._create_task("wait1")
        t_wait2 = self._create_task("wait2")
        
        self.queue_mgr.add_task(t_run)
        time.sleep(0.5) # 确保 t_run 进入 RUNNING
        self.queue_mgr.add_task(t_wait1)
        self.queue_mgr.add_task(t_wait2)
        
        self.assertEqual(t_run.status, TaskStatus.RUNNING)
        
        # 执行操作
        print(f"正在测试上移 ID: {t_wait2.task_id[-4:]}")
        up_ok, up_msg = self.queue_mgr.move_task_up(t_wait2.task_id)
        self.assertTrue(up_ok, up_msg)
        
        print(f"正在测试取消 ID: {t_wait1.task_id[-4:]}")
        can_ok, can_msg = self.queue_mgr.cancel_task(t_wait1.task_id)
        self.assertTrue(can_ok, can_msg)
        
        # 验证结果顺序
        # 等待 t_run 完成
        self._wait_for_status(t_run, TaskStatus.SUCCESS)
        # 此时 t_wait2 应该是下一个开始的
        self._wait_for_status(t_wait2, TaskStatus.SUCCESS, timeout=10)
        
        self.assertEqual(t_run.status, TaskStatus.SUCCESS)
        self.assertEqual(t_wait2.status, TaskStatus.SUCCESS)
        self.assertEqual(t_wait1.status, TaskStatus.CANCELLED)
        print("[OK] 队列操作（上移、取消）验证通过")

if __name__ == "__main__":
    unittest.main(verbosity=2)
