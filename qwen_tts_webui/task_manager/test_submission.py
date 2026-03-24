"""阶段三：任务提交接口功能测试"""

from qwen_tts_webui.task_manager.submission import (
    add_voice_generation_task_to_queue,
    add_voice_design_task_to_queue,
    add_voice_clone_task_to_queue,
)
from qwen_tts_webui.config_manager.shared import get_queue_manager
from qwen_tts_webui.task_manager.queue_manager import QueueManager


def test_voice_generation_submission() -> None:
    """测试声音生成任务提交"""
    print("\n=== 测试声音生成任务提交 ===")
    
    # 重置队列管理器
    from qwen_tts_webui.config_manager import shared
    shared.queue_manager = None
    queue_mgr = get_queue_manager()
    
    # 清空队列（通过创建新的队列管理器）
    shared.queue_manager = QueueManager()
    queue_mgr = get_queue_manager()
    
    # 提交一个有效的声音生成任务
    result = add_voice_generation_task_to_queue(
        model_name="Qwen/Qwen-TTS-Custom",
        text="你好，这是一个测试。",
        instruct="用温柔的语气说。",
        speaker="default",
        language="auto",
        segment_gen=False,
    )
    
    print(f"提交结果：{result[0]}")
    print(f"发言人更新：{result[1]}")
    print(f"语言更新：{result[2]}")
    
    # 验证队列中有任务
    status = queue_mgr.get_queue_status()
    print(f"当前队列中的任务数：{len(status)}")
    assert len(status) == 1, "应该有 1 个任务在队列中"
    print("✓ 声音生成任务提交成功")
    
    queue_mgr.shutdown()


def test_voice_design_submission() -> None:
    """测试声音设计任务提交"""
    print("\n=== 测试声音设计任务提交 ===")
    
    # 重置队列管理器
    from qwen_tts_webui.config_manager import shared
    shared.queue_manager = QueueManager()
    queue_mgr = get_queue_manager()
    
    # 提交一个有效的声音设计任务
    result = add_voice_design_task_to_queue(
        model_name="Qwen/Qwen-TTS-Design",
        text="你好，这是一个测试。",
        instruct="体现撒娇稚嫩的女声，音调偏高且起伏明显。",
        language="auto",
        segment_gen=False,
    )
    
    print(f"提交结果：{result[0]}")
    print(f"语言更新：{result[1]}")
    
    # 验证队列中有任务
    status = queue_mgr.get_queue_status()
    print(f"当前队列中的任务数：{len(status)}")
    assert len(status) == 1, "应该有 1 个任务在队列中"
    print("✓ 声音设计任务提交成功")
    
    queue_mgr.shutdown()


def test_voice_clone_submission() -> None:
    """测试声音克隆任务提交"""
    print("\n=== 测试声音克隆任务提交 ===")
    
    # 重置队列管理器
    from qwen_tts_webui.config_manager import shared
    shared.queue_manager = QueueManager()
    queue_mgr = get_queue_manager()
    
    # 创建一个临时文件用于测试
    import tempfile
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # 提交一个有效的声音克隆任务
        result = add_voice_clone_task_to_queue(
            model_name="Qwen/Qwen-TTS-Base",
            text="你好，这是一个测试。",
            language="auto",
            ref_audio=str(tmp_path),
            ref_text="这是参考音频的文本。",
            segment_gen=False,
        )
        
        print(f"提交结果：{result[0]}")
        print(f"语言更新：{result[1]}")
        
        # 验证队列中有任务
        status = queue_mgr.get_queue_status()
        print(f"当前队列中的任务数：{len(status)}")
        assert len(status) == 1, "应该有 1 个任务在队列中"
        print("✓ 声音克隆任务提交成功")
    finally:
        # 清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()
    
    queue_mgr.shutdown()


def test_invalid_params_submission() -> None:
    """测试无效参数提交被拒绝"""
    print("\n=== 测试无效参数提交 ===")
    
    # 重置队列管理器
    from qwen_tts_webui.config_manager import shared
    shared.queue_manager = QueueManager()
    queue_mgr = get_queue_manager()
    
    # 尝试提交空文本的声音生成任务（应该失败）
    result = add_voice_generation_task_to_queue(
        model_name="Qwen/Qwen-TTS-Custom",
        text="",  # 空文本
        instruct="用温柔的语气说。",
        speaker="default",
        language="auto",
        segment_gen=False,
    )
    
    print(f"空文本提交结果：{result[0]}")
    assert result[0] is None, "空文本应该被拒绝"
    
    # 验证队列为空
    status = queue_mgr.get_queue_status()
    print(f"当前队列中的任务数：{len(status)}")
    assert len(status) == 0, "队列应该为空（无效参数被拒绝）"
    print("✓ 无效参数被正确拒绝")
    
    queue_mgr.shutdown()


def test_queue_capacity_limit() -> None:
    """测试队列容量限制"""
    print("\n=== 测试队列容量限制 ===")
    
    # 重置队列管理器
    from qwen_tts_webui.config_manager import shared
    shared.queue_manager = QueueManager()
    queue_mgr = get_queue_manager()
    
    print(f"队列最大容量：{queue_mgr.MAX_QUEUE_SIZE}")
    
    # 尝试添加超过容量的任务
    success_count = 0
    for i in range(queue_mgr.MAX_QUEUE_SIZE + 5):
        result = add_voice_generation_task_to_queue(
            model_name="Qwen/Qwen-TTS-Custom",
            text=f"测试文本 {i}",
            instruct="用温柔的语气说。",
            speaker="default",
            language="auto",
            segment_gen=False,
        )
        
        if result[0] is not None:  # 成功提交
            success_count += 1
        else:
            print(f"第 {i+1} 个任务被拒绝（队列已满）")
            break
    
    print(f"成功提交的任务数：{success_count}")
    assert success_count <= queue_mgr.MAX_QUEUE_SIZE, f"成功提交的任务数不应超过 {queue_mgr.MAX_QUEUE_SIZE}"
    print("✓ 队列容量限制正常工作")
    
    queue_mgr.shutdown()


def run_all_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("阶段三：任务提交接口功能测试")
    print("=" * 60)
    
    try:
        test_voice_generation_submission()
        test_voice_design_submission()
        test_voice_clone_submission()
        test_invalid_params_submission()
        test_queue_capacity_limit()
        
        print("\n" + "=" * 60)
        print("✅ 所有阶段三测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败：{e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常：{e}")
        raise


if __name__ == "__main__":
    run_all_tests()
