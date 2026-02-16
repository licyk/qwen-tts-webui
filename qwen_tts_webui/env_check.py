"""Qwen TTS 运行环境检查"""

import sys
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from qwen_tts_webui.package_analyzer.pkg_check import (
    validate_requirements,
    PyWhlVersionComparison,
)
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
    PYTORCH_PACKAGES,
    PYTORCH_MIRROR_NJU,
    PYTORCH_MIRROR,
    PYPI_MIRROR_CERNET,
    PYPI_MIRROR,
    ROOT_PATH,
    UV_MINIMUN_VER,
)
from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.cmd_runner import run_cmd

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


def check_uv(
    custom_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> bool:
    """检查 uv 是否已安装

    Args:
        custom_env (dict[str, str] | None):
            自定义环境变量
        cwd (Path | None):
            执行 Pip 时的起始路径

    Returns:
        bool:
            如果 uv 可用
    """

    def _run_pip(
        *args: Any,
        custom_env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        run_cmd(
            [
                Path(sys.executable).as_posix(),
                "-m",
                "pip",
                "install",
                *args,
            ],
            custom_env=custom_env,
            cwd=cwd,
        )

    try:
        uv_ver = version("uv")
    except Exception:
        logger.info("安装 uv 中")
        try:
            _run_pip(
                "uv",
                "--upgrade",
                custom_env=custom_env,
                cwd=cwd,
            )
            logger.info("uv 安装完成")
            return True
        except RuntimeError:
            logger.warning("uv 安装失败")
            return False

    if PyWhlVersionComparison(uv_ver) < PyWhlVersionComparison(UV_MINIMUN_VER):
        logger.info("更新 uv 中")
        try:
            _run_pip(
                "uv",
                "--upgrade",
                custom_env=custom_env,
                cwd=cwd,
            )
            logger.info("uv 更新完成")
            return True
        except RuntimeError:
            logger.warning("uv 更新失败")
            return False

    return True


def pip_install(
    *args: Any,
    use_uv: bool | None = True,
    custom_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """使用 Pip / uv 安装 Python 软件包

    Args:
        *args (Any):
            要安装的 Python 软件包 (可使用 Pip / uv 命令行参数, 如`--upgrade`, `--force-reinstall`)
        use_uv (bool | None):
            使用 uv 代替 Pip 进行安装, 当 uv 安装 Python 软件包失败时, 将回退到 Pip 进行重试
        custom_env (dict[str, str] | None):
            自定义环境变量
        cwd (Path | str | None):
            执行 Pip / uv 时的起始路径

    Raises:
        RuntimeError:
            当 uv 和 pip 都无法安装软件包时抛出异常
    """
    uv_status = False
    if use_uv:
        uv_status = check_uv(
            custom_env=custom_env,
            cwd=cwd,
        )
    else:
        logger.warning("uv 不可用, 将使用 Pip 进行 Python 软件包安装")

    if uv_status:
        try:
            run_cmd(
                ["uv", "pip", "install", *args],
                custom_env=custom_env,
                cwd=cwd,
            )
        except RuntimeError as e:
            logger.warning("检测到 uv 安装 Python 软件包失败, 尝试回退到 Pip 重试 Python 软件包安装: %s", e)
            try:
                run_cmd(
                    [Path(sys.executable).as_posix(), "-m", "pip", "install", *args],
                    custom_env=custom_env,
                    cwd=cwd,
                )
            except RuntimeError as ee:
                logger.error("安装 Python 软件包时发生错误: %s", ee)
                raise RuntimeError(f"安装 Python 软件包时发生错误: {ee}") from ee
    else:
        try:
            run_cmd(
                [Path(sys.executable).as_posix(), "-m", "pip", "install", *args],
                custom_env=custom_env,
                cwd=cwd,
            )
        except RuntimeError as e:
            logger.error("安装 Python 软件包时发生错误: %s", e)
            raise RuntimeError(f"安装 Python 软件包时发生错误: {e}") from e


def install_requirement(
    reinstall_torch: bool | None = False,
    use_cn_mirror: bool | None = True,
    use_uv: bool | None = True,
) -> None:
    """安装 Qwen TTS 依赖

    Args:
        reinstall_torch (bool | None):
            重新安装 PyTorch
        use_cn_mirror (bool | None):
            使用国内镜像进行 Python 软件包安装
        use_uv (bool | None):
            使用 uv 进行依赖安装

    Raises:
        RuntimeError:
            当安装依赖失败时
    """

    def _set_pypi_mirror(mirror: str) -> None:
        custom_env["PIP_INDEX_URL"] = os.getenv("PIP_INDEX_URL", mirror)
        custom_env["UV_DEFAULT_INDEX"] = os.getenv("UV_DEFAULT_INDEX", mirror)

    pytorch_list = ["torch", "torchvision", "torchaudio"]
    need_install_pytorch = False
    custom_env = os.environ.copy()
    logger.info("检查 Qwen TTS WebUI 依赖完整性中")

    if use_uv:
        if use_cn_mirror:
            _set_pypi_mirror(PYPI_MIRROR_CERNET)
        else:
            _set_pypi_mirror(PYPI_MIRROR)
        check_uv(custom_env=custom_env)

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

    if need_install_pytorch:
        logger.info("安装 PyTorch 中")
        if use_cn_mirror:
            _set_pypi_mirror(PYTORCH_MIRROR_NJU)
        else:
            _set_pypi_mirror(PYTORCH_MIRROR)

        try:
            pip_install(
                *PYTORCH_PACKAGES,
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
            pip_install(
                "-r",
                requirements.as_posix(),
                custom_env=custom_env,
            )
        except RuntimeError as e:
            logger.error("安装 Qwen TTS WebUI 依赖时发生错误: %s", e)
            raise RuntimeError(f"安装 Qwen TTS WebUI 依赖时发生错误: {e}") from e

    logger.info("检查 Qwen TTS WebUI 依赖完整性完成")
