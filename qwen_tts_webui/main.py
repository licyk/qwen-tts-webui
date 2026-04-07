"""主启动文件"""

import time
import logging
import sys
from typing import TYPE_CHECKING

from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
    OUTPUT_PATH,
)
from qwen_tts_webui.logger import (
    get_logger,
    set_all_loggers_level,
)
from qwen_tts_webui.cmd_args import get_args_parser
from qwen_tts_webui.env_check import install_requirement
from qwen_tts_webui.version import VERSION

if TYPE_CHECKING:
    from fastapi import FastAPI
    from qwen_tts_webui.api.api import Api

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


def create_api(app: "FastAPI") -> "Api":
    """创建 API 实例

    Args:
        app (FastAPI): FastAPI 应用实例

    Returns:
        Api: API 实例
    """
    from qwen_tts_webui.api.api import Api
    from qwen_tts_webui.task_manager.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only() -> None:
    """仅启动 API 服务器"""
    from fastapi import FastAPI
    from qwen_tts_webui.utils import find_port

    args = get_args_parser().parse_args()

    logger.info("初始化 Qwen TTS API 服务器")

    app = FastAPI(
        title="Qwen TTS API",
        description="Qwen TTS WebUI API 接口",
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    api = create_api(app)

    port = find_port(args.server_port)
    logger.info("Qwen TTS API 服务器已启动")
    logger.info("API 文档地址: http://%s:%d/docs", args.server_name, port)
    logger.info("API 地址: http://%s:%d/qwenapi/v1/", args.server_name, port)

    api.launch(server_name=args.server_name, port=port)


def webui() -> None:
    """启动 WebUI (可选启用 API)"""
    args = get_args_parser().parse_args()

    logger.info("初始化 Qwen TTS WebUI 界面")

    from qwen_tts_webui.frontend import create_ui
    from qwen_tts_webui.utils import find_port

    demo = create_ui()
    app, local_url, share_url = demo.launch(
        server_name=args.server_name,
        server_port=find_port(args.server_port),
        inbrowser=not args.no_inbrowser,
        share=args.share,
        prevent_thread_lock=True,
        allowed_paths=[OUTPUT_PATH.as_posix()],
        app_kwargs={
            "docs_url": "/docs",
            "redoc_url": "/redoc",
        },
        css="""
        .token-counter {
            font-family: monospace;
            font-size: 0.9em;
            color: #666;
            padding: 4px 8px;
            background: #f0f0f0;
            border-radius: 4px;
            display: inline-block;
        }
        """,
    )
    local_url = local_url.rstrip("/")
    logger.info("Qwen TTS WebUI 已启动, 界面访问地址: %s", local_url)
    if share_url:
        logger.info("Qwen TTS WebUI 远程访问地址: %s", share_url)

    # 如果启用了 API, 则创建 API 路由
    if args.api:
        create_api(app)
        logger.info("API 已启用, API 文档地址: %s/docs", local_url)
        logger.info("API 地址: %s/qwenapi/v1/", local_url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在关闭 Qwen TTS WebUI...")


def main() -> None:
    """主函数"""
    args = get_args_parser().parse_args()
    if args.version:
        logger.info("Qwen TTS WebUI 版本: %s", VERSION)
        sys.exit(0)

    logger.info("初始化 Qwen TTS WebUI")

    if not args.skip_check:
        install_requirement(
            reinstall_torch=args.reinstall_torch,
            use_cn_mirror=not args.disable_pypi_cn_mirror,
            use_uv=not args.disable_uv,
        )

    if args.debug:
        logger.info("启用 Debug 模式")
        set_all_loggers_level(level=logging.DEBUG)
        import qwen_tts_webui.config_manager.config

        qwen_tts_webui.config_manager.config.LOGGER_LEVEL = logging.DEBUG

    if args.nowebui:
        api_only()
    else:
        webui()
