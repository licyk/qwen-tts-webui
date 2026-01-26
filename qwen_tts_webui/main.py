"""主启动文件"""

import time
import logging
import sys

from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
)
from qwen_tts_webui.logger import (
    get_logger,
    set_all_loggers_level,
)
from qwen_tts_webui.cmd_args import get_args_parser
from qwen_tts_webui.env_check import install_requirement
from qwen_tts_webui.version import VERSION

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


def main() -> None:
    """主函数"""
    args = get_args_parser().parse_args()
    if args.version:
        logger.info("Qwen TTS WebUI 版本: %s", VERSION)
        sys.exit(0)

    if args.debug:
        logger.info("启用 Debug 模式")
        set_all_loggers_level(level=logging.DEBUG)
        import qwen_tts_webui.config_manager.config
        qwen_tts_webui.config_manager.config.LOGGER_LEVEL = logging.DEBUG

    logger.info("初始化 Qwen TTS WebUI")

    if not args.skip_check:
        install_requirement(
            reinstall_torch=args.reinstall_torch,
            use_cn_mirror=not args.disable_pypi_cn_mirror,
        )

    from qwen_tts_webui.frontend import create_ui
    from qwen_tts_webui.utils import find_port

    demo = create_ui()
    _, local_url, share_url = demo.launch(
        server_name=args.server_name,
        server_port=find_port(args.server_port),
        share=args.share,
        prevent_thread_lock=True,
    )
    logger.info("Qwen TTS WebUI 已启动, 界面访问地址: %s", local_url)
    if share_url:
        logger.info("Qwen TTS WebUI 远程访问地址: %s", share_url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在关闭 Qwen TTS WebUI...")
