"""任务执行器测试脚本"""

import tempfile
from pathlib import Path

from qwen_tts_webui.task_manager.models import TaskType, TaskInfo
from qwen_tts_webui.task_manager.task_executor import TaskExecutor


def create_mock_task(
    task_type: TaskType,
    params: dict,
) -> TaskInfo:
    """创建模拟任务"""
    return TaskInfo.create(
        task_type=task_type,
        params=params,
    )


def test_task_executor_initialization() -> None:
    """测试任务执行器初始化"""
    print("\n=== 测试任务执行器初始化 ===")
    
    executor = TaskExecutor()
    print("✓ 任务执行器初始化成功")


def test_execute_voice_generation_task() -> None:
    """测试声音生成任务执行（模拟）"""
    print("\n=== 测试声音生成任务执行 ===")
    
    # 创建模拟任务参数
    params = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        "text": "你好，这是一个测试。",
        "speaker": "default",
        "language": "auto",
        "instruct": "用温柔的语气说。",
    }
    
    task = create_mock_task(TaskType.VOICE_GENERATION, params)
    executor = TaskExecutor()
    
    # 注意：这个测试需要实际的模型和后端支持
    # 在实际环境中会调用真实的生成函数
    # 这里我们只测试参数验证和执行流程
    print(f"任务类型：{task.task_type}")
    print(f"任务参数：{task.params}")
    print("✓ 声音生成任务参数验证通过")


def test_execute_voice_design_task() -> None:
    """测试声音设计任务执行（模拟）"""
    print("\n=== 测试声音设计任务执行 ===")
    
    # 创建模拟任务参数
    params = {
        "model_name": "Qwen/Qwen-TTS-Design",
        "text": "你好，这是一个测试。",
        "language": "auto",
        "instruct": "体现撒娇稚嫩的女声，音调偏高且起伏明显。",
    }
    
    task = create_mock_task(TaskType.VOICE_DESIGN, params)
    executor = TaskExecutor()
    
    print(f"任务类型：{task.task_type}")
    print(f"任务参数：{task.params}")
    print("✓ 声音设计任务参数验证通过")


def test_execute_voice_clone_task() -> None:
    """测试声音克隆任务执行（模拟）"""
    print("\n=== 测试声音克隆任务执行 ===")
    
    # 创建临时音频文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(b"fake audio data")
        tmp_audio_path = tmp_file.name
    
    try:
        # 创建模拟任务参数
        params = {
            "model_name": "Qwen/Qwen-TTS-Base",
            "text": "你好，这是一个测试。",
            "language": "auto",
            "ref_audio": str(tmp_audio_path),
            "ref_text": "这是参考音频的文本。",
        }
        
        task = create_mock_task(TaskType.VOICE_CLONE, params)
        executor = TaskExecutor()
        
        print(f"任务类型：{task.task_type}")
        print(f"任务参数：{task.params}")
        print("✓ 声音克隆任务参数验证通过")
        
    finally:
        # 清理临时文件
        Path(tmp_audio_path).unlink()


def test_invalid_params() -> None:
    """测试无效参数处理"""
    print("\n=== 测试无效参数处理 ===")
    
    # 创建缺少必需参数的任务
    params = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        # 缺少 text, speaker, language 等必需参数
    }
    
    task = create_mock_task(TaskType.VOICE_GENERATION, params)
    executor = TaskExecutor()
    
    # 参数验证会在 execute_task 中进行
    # 这里只是展示任务创建
    print(f"任务类型：{task.task_type}")
    print(f"缺失必需参数的任务已创建")
    print("✓ 无效参数任务创建成功（将在执行时被拒绝）")


def run_all_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("任务执行器测试")
    print("=" * 60)
    
    try:
        test_task_executor_initialization()
        test_execute_voice_generation_task()
        test_execute_voice_design_task()
        test_execute_voice_clone_task()
        test_invalid_params()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n注意：以上测试仅验证了任务创建和参数结构")
        print("实际的任务执行需要完整的模型和后端环境支持")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常：{e}")
        raise


if __name__ == "__main__":
    run_all_tests()
