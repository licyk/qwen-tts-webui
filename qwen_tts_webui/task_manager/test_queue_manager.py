"""队列管理器基础功能测试脚本"""

import time
import threading
from qwen_tts_webui.task_manager.models import TaskType, TaskInfo
from qwen_tts_webui.task_manager.queue_manager import QueueManager


def create_unique_task(task_type: TaskType, data: str) -> TaskInfo:
    """创建具有唯一 ID 的任务"""
    task = TaskInfo.create(
        task_type=task_type,
        params={"test": data},
    )
    # 确保任务 ID 唯一性
    time.sleep(0.01)  # 10ms 间隔
    return task


def test_add_task() -> None:
    """测试任务添加功能"""
    print("\n=== 测试任务添加功能 ===")
    
    queue_mgr = QueueManager()
    
    # 创建测试任务
    task1 = create_unique_task(TaskType.VOICE_GENERATION, "data1")
    task2 = create_unique_task(TaskType.VOICE_DESIGN, "data2")
    
    # 测试添加任务
    success1, msg1 = queue_mgr.add_task(task1)
    print(f"添加任务 1: {success1}, {msg1}")
    assert success1, f"添加任务 1 失败：{msg1}"
    
    success2, msg2 = queue_mgr.add_task(task2)
    print(f"添加任务 2: {success2}, {msg2}")
    assert success2, f"添加任务 2 失败：{msg2}"
    
    # 验证队列状态
    status = queue_mgr.get_queue_status()
    print(f"当前队列长度：{len(status)}")
    assert len(status) == 2, "队列长度应为 2"
    
    # 等待任务执行完成（临时占位逻辑会执行 2 秒）
    time.sleep(3)
    
    # 再次检查队列状态
    status = queue_mgr.get_queue_status()
    print(f"任务执行后队列长度：{len(status)}")
    
    queue_mgr.shutdown()
    print("✓ 任务添加功能测试通过")


def test_cancel_task() -> None:
    """测试任务取消功能"""
    print("\n=== 测试任务取消功能 ===")
    
    queue_mgr = QueueManager()
    
    # 创建并添加任务
    task1 = create_unique_task(TaskType.VOICE_CLONE, "data1")
    task2 = create_unique_task(TaskType.VOICE_GENERATION, "data2")
    
    queue_mgr.add_task(task1)
    queue_mgr.add_task(task2)
    
    # 立即尝试取消 task2（应该在等待中）
    success, msg = queue_mgr.cancel_task(task2.task_id)
    print(f"取消任务 2: {success}, {msg}")
    
    # 尝试取消不存在的任务
    success_nonexist, msg_nonexist = queue_mgr.cancel_task("non_existent_id")
    print(f"取消不存在的任务：{success_nonexist}, {msg_nonexist}")
    assert not success_nonexist, "取消不存在的任务应该失败"
    
    queue_mgr.shutdown()
    print("✓ 任务取消功能测试完成")


def test_move_tasks() -> None:
    """测试任务上移/下移功能"""
    print("\n=== 测试任务移动功能 ===")
    
    queue_mgr = QueueManager()
    
    # 创建 3 个任务，使用不同的任务类型以避免模型切换
    task1 = create_unique_task(TaskType.VOICE_GENERATION, "data1")
    task2 = create_unique_task(TaskType.VOICE_GENERATION, "data2")
    task3 = create_unique_task(TaskType.VOICE_GENERATION, "data3")
    
    queue_mgr.add_task(task1)
    # 不立即添加 task2 和 task3，让 task1 先开始执行
    time.sleep(0.5)
    queue_mgr.add_task(task2)
    time.sleep(0.1)
    queue_mgr.add_task(task3)
    
    # 验证初始顺序（此时 task1 应该正在执行，task2 和 task3 在等待）
    status = queue_mgr.get_queue_status()
    print(f"当前队列任务数：{len(status)}")
    print(f"任务顺序：{[(t.task_id[-4:], t.status) for t in status]}")
    
    # 测试上移 task3（应该在 task2 之后）
    success_up, msg_up = queue_mgr.move_task_up(task3.task_id)
    print(f"上移 task3: {success_up}, {msg_up}")
    
    # 测试下移 task1（应该在队首，无法下移 - 因为第一个任务是 WAITING 状态）
    # 等待一小段时间，确保任务状态稳定
    time.sleep(0.2)
    status_after_up = queue_mgr.get_queue_status()
    print(f"上移后的顺序：{[(t.task_id[-4:], t.status) for t in status_after_up]}")
    
    # 测试上移队首的等待任务（应该失败）
    waiting_tasks = [t for t in status_after_up if t.status == "waiting"]
    if waiting_tasks:
        first_waiting = waiting_tasks[0]
        success_first_up, msg_first_up = queue_mgr.move_task_up(first_waiting.task_id)
        print(f"上移队首等待任务：{success_first_up}, {msg_first_up}")
        # 如果队首等待任务不是队列的第一个，应该可以上移
    
    queue_mgr.shutdown()
    print("✓ 任务移动功能测试完成")


def test_queue_capacity() -> None:
    """测试队列容量限制"""
    print("\n=== 测试队列容量限制 ===")
    
    queue_mgr = QueueManager()
    print(f"队列最大容量：{queue_mgr.MAX_QUEUE_SIZE}")
    
    # 尝试添加超过容量限制的任务数
    tasks = []
    success_count = 0
    for i in range(queue_mgr.MAX_QUEUE_SIZE + 5):
        task = create_unique_task(TaskType.VOICE_GENERATION, f"index_{i}")
        tasks.append(task)
        time.sleep(0.005)  # 确保 ID 唯一
        success, msg = queue_mgr.add_task(task)
        if success:
            success_count += 1
        else:
            print(f"第 {i+1} 个任务添加失败：{msg}")
            break
    
    print(f"成功添加任务数：{success_count}")
    assert success_count <= queue_mgr.MAX_QUEUE_SIZE, "队列不应超过最大容量"
    
    queue_mgr.shutdown()
    print("✓ 队列容量限制测试通过")


def test_thread_safety() -> None:
    """测试线程锁机制"""
    print("\n=== 测试线程安全性 ===")
    
    queue_mgr = QueueManager()
    
    # 使用线程池快速并发添加多个任务
    tasks_added = []
    results_lock = threading.Lock()
    
    def add_task_thread(index: int) -> None:
        task = create_unique_task(TaskType.VOICE_GENERATION, f"thread_{index}")
        success, msg = queue_mgr.add_task(task)
        with results_lock:
            tasks_added.append(success)
    
    # 创建并启动 10 个线程
    threads = []
    for i in range(10):
        t = threading.Thread(target=add_task_thread, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    success_count = sum(tasks_added)
    print(f"快速添加 10 个任务，成功：{success_count}")
    assert success_count == 10, f"应该有 10 个任务成功添加，实际：{success_count}"
    
    # 等待一小段时间让工作线程处理
    time.sleep(1)
    
    status = queue_mgr.get_queue_status()
    print(f"当前队列中的任务数：{len(status)}")
    
    queue_mgr.shutdown()
    print("✓ 线程安全性测试通过")


def run_all_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("队列管理器基础功能测试")
    print("=" * 60)
    
    try:
        test_add_task()
        test_cancel_task()
        test_move_tasks()
        test_queue_capacity()
        test_thread_safety()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败：{e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常：{e}")
        raise


if __name__ == "__main__":
    run_all_tests()
