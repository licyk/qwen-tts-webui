"""命令行参数解析"""

import argparse


def get_args_parser() -> argparse.ArgumentParser:
    """获取命令行参数

    Returns:
        argparse.ArgumentParser: 命令行参数解析器
    """

    parser = argparse.ArgumentParser(description="Qwen TTS WebUI 命令行参数")
    parser.add_argument("--debug", action="store_true", help="启用 Debug 模式")
    parser.add_argument("--reinstall-torch", action="store_true", help="重装环境中的 PyTorch")
    parser.add_argument("--disable-pypi-cn-mirror", action="store_true", help="禁用 PyPI 国内镜像")
    parser.add_argument("--disable-uv", action="store_true", help="禁用 uv 包管理器, 使用 Pip 进行依赖安装")
    parser.add_argument("--skip-check", action="store_true", help="跳过运行环境的依赖检测")
    parser.add_argument("--no-inbrowser", action="store_true", help="禁止自动打开浏览器访问 Qwen TTS WebUI")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Qwen TTS 启动服务器名 (默认为 127.0.0.1)")
    parser.add_argument("--server-port", type=int, default=7860, help="Qwen TTS 启动端口 (默认为 7860)")
    parser.add_argument("--share", action="store_true", help="启用 Gradio 共享")
    parser.add_argument("--version", action="store_true", help="显示版本信息")
    return parser
