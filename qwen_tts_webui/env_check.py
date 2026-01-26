"""Qwen TTS 运行环境检查"""

import sys
import os
from importlib.metadata import version
from pathlib import Path

from qwen_tts_webui.package_analyzer.pkg_check import validate_requirements
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
    PYTORCH_PACKAGES,
    PYTORCH_MIRROR_NJU,
    PYTORCH_MIRROR,
    PYPI_MIRROR_CERNET,
    PYPI_MIRROR,
    ROOT_PATH,
)
from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.cmd_runner import run_cmd

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


def install_requirement(
    reinstall_torch: bool | None = None,
    use_cn_mirror: bool | None = True,
) -> None:
    """安装 Qwen TTS 依赖"""

    def _set_pypi_mirror(mirror: str) -> None:
        custom_env["PIP_INDEX_URL"] = os.getenv("PIP_INDEX_URL", mirror)
        custom_env["UV_DEFAULT_INDEX"] = os.getenv("UV_DEFAULT_INDEX", mirror)

    pytorch_list = ["torch", "torchvision", "torchaudio", "xformers"]
    need_install_pytorch = False

    logger.info("检查 Qwen TTS WebUI 依赖完整性中")

    if reinstall_torch:
        logger.info("卸载原有 PyTorch 中")
        run_cmd(
            [
                Path(sys.executable).as_posix(),
                "-m",
                "pip",
                "uninstall",
                *PYTORCH_PACKAGES,
                "-y",
            ],
            check=False,
        )

    for i in pytorch_list:
        try:
            version(i)
        except Exception:
            need_install_pytorch = True
            break

    custom_env = os.environ.copy()

    if need_install_pytorch:
        logger.info("安装 PyTorch 中")
        if use_cn_mirror:
            _set_pypi_mirror(PYTORCH_MIRROR_NJU)
        else:
            _set_pypi_mirror(PYTORCH_MIRROR)

        try:
            run_cmd(
                [
                    Path(sys.executable).as_posix(),
                    "-m",
                    "pip",
                    "install",
                    *PYTORCH_PACKAGES,
                ],
                custom_env=custom_env,
            )
        except RuntimeError as e:
            logger.error("安装 PyTorch 时发生错误: %s", e)
            raise RuntimeError(f"安装 PyTorch 时发生错误: {e}") from e

    requirements = ROOT_PATH / "requirements.txt"
    if not requirements.is_file():
        logger.error("在 %s 中未找到依赖记录文件, 请检查项目文件是否完整", requirements)
        raise FileNotFoundError(f"在 {requirements} 中未找到依赖记录文件, 请检查项目文件是否完整")

    if not validate_requirements(requirements):
        logger.info("安装 Qwen TTS WebUI 依赖中")
        if use_cn_mirror:
            _set_pypi_mirror(PYPI_MIRROR_CERNET)
        else:
            _set_pypi_mirror(PYPI_MIRROR)

        try:
            run_cmd(
                [
                    Path(sys.executable).as_posix(),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    requirements.as_posix(),
                ],
                custom_env=custom_env,
            )
        except RuntimeError as e:
            logger.error("安装 Qwen TTS WebUI 依赖时发生错误: %s", e)
            raise RuntimeError(f"安装 Qwen TTS WebUI 依赖时发生错误: {e}") from e

    logger.info("检查 Qwen TTS WebUI 依赖完整性完成")
