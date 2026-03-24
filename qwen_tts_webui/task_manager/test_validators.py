"""参数验证函数测试脚本"""

from qwen_tts_webui.task_manager.validators import (
    validate_voice_generation,
    validate_voice_design,
    validate_voice_clone,
    validate_task_params,
)


def test_validate_voice_generation() -> None:
    """测试声音生成参数验证"""
    print("\n=== 测试声音生成参数验证 ===")
    
    # 测试有效参数
    valid_params = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        "text": "你好，这是一个测试。",
        "speaker": "default",
        "language": "auto",
        "instruct": "用温柔的语气说。",
    }
    success, msg = validate_voice_generation(valid_params)
    print(f"有效参数验证：{success}, {msg}")
    assert success, f"验证应该通过：{msg}"
    
    # 测试缺少必需参数
    missing_params = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        "text": "你好，这是一个测试。",
        # 缺少 speaker 和 language
    }
    success, msg = validate_voice_generation(missing_params)
    print(f"缺少必需参数：{success}, {msg}")
    assert not success, "应该检测到缺少必需参数"
    
    # 测试空文本
    empty_text = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        "text": "",
        "speaker": "default",
        "language": "auto",
    }
    success, msg = validate_voice_generation(empty_text)
    print(f"空文本验证：{success}, {msg}")
    assert not success, "应该检测到空文本"
    
    # 测试过长的 instruct
    long_instruct = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        "text": "你好，这是一个测试。",
        "speaker": "default",
        "language": "auto",
        "instruct": "A" * 501,  # 超过 500 字符
    }
    success, msg = validate_voice_generation(long_instruct)
    print(f"过长 instruct 验证：{success}, {msg}")
    assert not success, "应该检测到 instruct 过长"
    
    print("✓ 声音生成参数验证测试通过")


def test_validate_voice_design() -> None:
    """测试声音设计参数验证"""
    print("\n=== 测试声音设计参数验证 ===")
    
    # 测试有效参数
    valid_params = {
        "model_name": "Qwen/Qwen-TTS-Design",
        "text": "你好，这是一个测试。",
        "language": "auto",
        "instruct": "体现撒娇稚嫩的女声，音调偏高且起伏明显。",
    }
    success, msg = validate_voice_design(valid_params)
    print(f"有效参数验证：{success}, {msg}")
    assert success, f"验证应该通过：{msg}"
    
    # 测试缺少 instruct
    missing_instruct = {
        "model_name": "Qwen/Qwen-TTS-Design",
        "text": "你好，这是一个测试。",
        "language": "auto",
    }
    success, msg = validate_voice_design(missing_instruct)
    print(f"缺少 instruct: {success}, {msg}")
    assert not success, "应该检测到缺少 instruct"
    
    # 测试空 instruct
    empty_instruct = {
        "model_name": "Qwen/Qwen-TTS-Design",
        "text": "你好，这是一个测试。",
        "language": "auto",
        "instruct": "",
    }
    success, msg = validate_voice_design(empty_instruct)
    print(f"空 instruct 验证：{success}, {msg}")
    assert not success, "应该检测到空 instruct"
    
    print("✓ 声音设计参数验证测试通过")


def test_validate_voice_clone() -> None:
    """测试声音克隆参数验证"""
    print("\n=== 测试声音克隆参数验证 ===")
    
    import tempfile
    from pathlib import Path
    
    # 创建一个临时音频文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(b"fake audio data")
        tmp_audio_path = tmp_file.name
    
    try:
        # 测试有效参数
        valid_params = {
            "model_name": "Qwen/Qwen-TTS-Base",
            "text": "你好，这是一个测试。",
            "language": "auto",
            "ref_audio": tmp_audio_path,
            "ref_text": "这是参考音频的文本。",
        }
        success, msg = validate_voice_clone(valid_params)
        print(f"有效参数验证：{success}, {msg}")
        assert success, f"验证应该通过：{msg}"
        
        # 测试缺少 ref_audio
        missing_ref_audio = {
            "model_name": "Qwen/Qwen-TTS-Base",
            "text": "你好，这是一个测试。",
            "language": "auto",
        }
        success, msg = validate_voice_clone(missing_ref_audio)
        print(f"缺少 ref_audio: {success}, {msg}")
        assert not success, "应该检测到缺少 ref_audio"
        
        # 测试不存在的音频文件
        non_existent_audio = {
            "model_name": "Qwen/Qwen-TTS-Base",
            "text": "你好，这是一个测试。",
            "language": "auto",
            "ref_audio": "/path/to/nonexistent/file.wav",
        }
        success, msg = validate_voice_clone(non_existent_audio)
        print(f"不存在的音频文件：{success}, {msg}")
        assert not success, "应该检测到音频文件不存在"
        
        # 测试过长的 ref_text
        long_ref_text = {
            "model_name": "Qwen/Qwen-TTS-Base",
            "text": "你好，这是一个测试。",
            "language": "auto",
            "ref_audio": tmp_audio_path,
            "ref_text": "A" * 501,  # 超过 500 字符
        }
        success, msg = validate_voice_clone(long_ref_text)
        print(f"过长 ref_text 验证：{success}, {msg}")
        assert not success, "应该检测到 ref_text 过长"
        
    finally:
        # 清理临时文件
        Path(tmp_audio_path).unlink()
    
    print("✓ 声音克隆参数验证测试通过")


def test_validate_task_params() -> None:
    """测试统一的参数验证入口"""
    print("\n=== 测试统一验证入口 ===")
    
    # 测试声音生成
    gen_params = {
        "model_name": "Qwen/Qwen-TTS-Custom",
        "text": "你好，这是一个测试。",
        "speaker": "default",
        "language": "auto",
    }
    success, msg = validate_task_params("voice_generation", gen_params)
    print(f"声音生成验证：{success}, {msg}")
    assert success, f"声音生成验证应该通过：{msg}"
    
    # 测试声音设计
    design_params = {
        "model_name": "Qwen/Qwen-TTS-Design",
        "text": "你好，这是一个测试。",
        "language": "auto",
        "instruct": "体现撒娇稚嫩的女声。",
    }
    success, msg = validate_task_params("voice_design", design_params)
    print(f"声音设计验证：{success}, {msg}")
    assert success, f"声音设计验证应该通过：{msg}"
    
    # 测试声音克隆（使用临时文件）
    import tempfile
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(b"fake audio data")
        tmp_audio_path = tmp_file.name
    
    try:
        clone_params = {
            "model_name": "Qwen/Qwen-TTS-Base",
            "text": "你好，这是一个测试。",
            "language": "auto",
            "ref_audio": tmp_audio_path,
        }
        success, msg = validate_task_params("voice_clone", clone_params)
        print(f"声音克隆验证：{success}, {msg}")
        assert success, f"声音克隆验证应该通过：{msg}"
    finally:
        Path(tmp_audio_path).unlink()
    
    # 测试未知任务类型
    success, msg = validate_task_params("unknown_type", {})
    print(f"未知任务类型验证：{success}, {msg}")
    assert not success, "应该检测到未知的任务类型"
    
    print("✓ 统一验证入口测试通过")


def run_all_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("参数验证函数测试")
    print("=" * 60)
    
    try:
        test_validate_voice_generation()
        test_validate_voice_design()
        test_validate_voice_clone()
        test_validate_task_params()
        
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
