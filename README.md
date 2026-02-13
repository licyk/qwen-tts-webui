# Qwen TTS WebUI
一个基于 [Gradio](https://github.com/gradio-app/gradio) 的 Web 用户界面，提供通义千问 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 系列模型进行文本到语音的转换。


# 安装
可选择以下其中一个安装方式进行安装。


## 使用整合包 (仅 Windows)
前往此处查找 Qwen TTS WebUI：[AI 绘画 / 训练整合包 · licyk/sd-webui-all-in-one · Discussion #1](https://github.com/licyk/sd-webui-all-in-one/discussions/1)


## 使用安装器 (Windows / Linux / MacOS)
阅读此处的说明进行使用：[Qwen TTS WebUI Installer 使用文档](https://github.com/licyk/sd-webui-all-in-one/blob/main/docs/qwen_tts_webui_installer.md)


## 手动安装 (Windows / Linux / MacOS)
请确保已经在系统中安装了 [Python](https://www.python.org) 和 [Git](https://git-scm.com)。

下载该项目到本地：

```bash
git clone https://github.com/licyk/qwen-tts-webui
```

启动该项目：

```bash
# Windows
cmd /k start.bat # 或者直接双击 start.bat

# Windows 平台的另一种启动方式
./start.ps1 # 或者右键选择使用 PowerShell 运行

# Linux / MacOS
./start.sh
```

或者手动创建并进入虚拟环境并进入，再运行`python launch.py`启动该项目。


# 使用
进入 Qwen TTS WebUI 后可直接根据界面提示进行使用，进行声音生成时将自动下载对应的模型。


## 可使用的模型
|模型|下载地址|
|---|---|
|Qwen/Qwen3-TTS-12Hz-1.7B-Base|[HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)|
|Qwen/Qwen3-TTS-12Hz-0.6B-Base|[HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)|
|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|[HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)|
|Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice|[HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)|
|Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign|[HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)|


## 启动参数
```
usage: launch.py [-h] [--debug] [--reinstall-torch] [--disable-pypi-cn-mirror] [--disable-uv] [--skip-check]
                 [--no-inbrowser] [--server-name SERVER_NAME] [--server-port SERVER_PORT] [--share] [--version]

Qwen TTS WebUI 命令行参数

options:
  -h, --help            show this help message and exit
  --debug               启用 Debug 模式
  --reinstall-torch     重装环境中的 PyTorch
  --disable-pypi-cn-mirror
                        禁用 PyPI 国内镜像
  --disable-uv          禁用 uv 包管理器, 使用 Pip 进行依赖安装
  --skip-check          跳过运行环境的依赖检测
  --no-inbrowser        禁止自动打开浏览器访问 Qwen TTS WebUI
  --server-name SERVER_NAME
                        Qwen TTS 启动服务器名 (默认为 127.0.0.1)
  --server-port SERVER_PORT
                        Qwen TTS 启动端口 (默认为 7860)
  --share               启用 Gradio 共享
  --version             显示版本信息
```


# 使用的项目
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): 文本转语音。


# 许可证
- [GPL-3.0](LICENSE)
