"""参数存储框架"""

import json
import traceback
import os
from typing import Any

from qwen_tts_webui.logger import get_logger
from qwen_tts_webui.config_manager.config import (
    LOGGER_LEVEL,
    LOGGER_COLOR,
)

logger = get_logger(
    level=LOGGER_LEVEL,
    color=LOGGER_COLOR,
)


class OptionInfo:
    """配置项信息类"""

    def __init__(
        self,
        default: Any = None,
        label: str = "",
        component: Any = None,
        component_args: dict[str, Any] | None = None,
    ) -> None:
        """初始化配置项信息

        Args:
            default (Any): 默认值
            label (str): 界面显示的标签
            component (Any): Gradio 组件类
            component_args (dict[str, Any] | None): 组件初始化参数
        """
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args or {}


class Options:
    """配置管理类"""

    def __init__(
        self,
        data_labels: dict[str, OptionInfo],
    ) -> None:
        """初始化配置管理

        Args:
            data_labels (dict[str, OptionInfo]): 配置项定义字典
        """
        self.data_labels = data_labels
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __getattr__(
        self,
        item: str,
    ) -> Any:
        if item == "data":
            return super().__getattribute__(item)
        if item in self.data:
            return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super().__getattribute__(item)

    def __setattr__(
        self,
        key: str,
        value: Any,
    ) -> None:
        if key in ["data", "data_labels"]:
            return super().__setattr__(key, value)
        if key in self.data_labels:
            self.data[key] = value
        else:
            super().__setattr__(key, value)

    def save(
        self,
        filename: str | os.PathLike,
    ) -> None:
        """保存配置到文件

        Args:
            filename (str | os.PathLike): 配置文件路径
        """
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    def load(
        self,
        filename: str | os.PathLike,
    ) -> None:
        """从文件加载配置

        Args:
            filename (str | os.PathLike): 配置文件路径
        """
        if not os.path.exists(filename):
            return
        try:
            with open(filename, "r", encoding="utf8") as file:
                loaded_data = json.load(file)
                for k, v in loaded_data.items():
                    if k in self.data_labels:
                        self.data[k] = v
        except Exception as e:
            traceback.print_exc()
            logger.error("加载配置文件时发生错误: %s", e)

    def reset(
        self,
    ) -> None:
        """重置所有配置为默认值"""
        self.data = {k: v.default for k, v in self.data_labels.items()}
