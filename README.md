<div align="center">

# Qwen TTS WebUI

_✨生成音频如此简单_
  <p align="center">
    <a href="https://github.com/licyk/qwen-tts-webui/stargazers" style="margin: 2px;">
      <img src="https://img.shields.io/github/stars/licyk/qwen-tts-webui?style=flat&logo=github&logoColor=silver&color=bluegreen&labelColor=grey" alt="Stars">
    </a>
    <a href="https://github.com/licyk/qwen-tts-webui/issues">
      <img src="https://img.shields.io/github/issues/licyk/qwen-tts-webui?style=flat&logo=github&logoColor=silver&color=bluegreen&labelColor=grey" alt="Issues">
    </a>
    <a href="https://github.com/licyk/qwen-tts-webui/commits/dev">
      <img src="https://flat.badgen.net/github/last-commit/licyk/qwen-tts-webui/dev?icon=github&color=green&label=last%20dev%20commit" alt="Commit">
    </a>
    <a href="https://github.com/licyk/qwen-tts-webui/actions/workflows/sync_repo.yml">
      <img src="https://github.com/licyk/qwen-tts-webui/actions/workflows/sync_repo.yml/badge.svg" alt="Sync">
    </a>
    <a href="https://github.com/licyk/qwen-tts-webui/actions/workflows/py-lint.yml">
      <img src="https://github.com/licyk/qwen-tts-webui/actions/workflows/py-lint.yml/badge.svg" alt="Ruff Lint">
    </a>
  </p>

</div>

# Qwen TTS WebUI
一个基于 [Gradio](https://github.com/gradio-app/gradio) 的 Web 用户界面，提供通义千问 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 系列模型进行文本到语音的转换。


# 安装
可选择以下其中一个安装方式进行安装。


## 使用整合包 (仅 Windows)
前往此处查找 Qwen TTS WebUI：[AI 绘画 / 训练 / 语音生成整合包 · licyk/sd-webui-all-in-one · Discussion #1](https://github.com/licyk/sd-webui-all-in-one/discussions/1)

>[!IMPORTANT]  
>在`AI 绘画 / 训练 / 语音生成整合包`下载页面中包含大量不同类型的整合包，请注意找到 Qwen TTS WebUI 的页面说明再点击下载！
>
>如果找不到下载页面说明，请按下`Ctrl + F`打开浏览器的搜索功能，输入`Qwen TTS WebUI`进行当前网页搜索。


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

其他说明请参看：[使用离线模式运行与加载本地路径的模型 · licyk/qwen-tts-webui · Discussion #1](https://github.com/licyk/qwen-tts-webui/discussions/1)


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


# API 使用
Qwen TTS WebUI 支持启用 API 模式，在启动参数中加入`--api`参数即可启用。


## 启动 API 服务

### 方式 1: 启动 WebUI 并启用 API
```bash
python launch.py --api
```

访问地址:
- WebUI: http://127.0.0.1:7860
- API 文档: http://127.0.0.1:7860/docs
- API 端点: http://127.0.0.1:7860/qwenapi/v1/


### 方式 2: 仅启动 API 服务器 (不启动 WebUI)
```bash
python launch.py --nowebui
```

访问地址:
- API 文档: http://127.0.0.1:7860/docs
- API 端点: http://127.0.0.1:7860/qwenapi/v1/


### 自定义服务器地址和端口
```bash
python launch.py --api --server-name 0.0.0.0 --server-port 8080
```


## API 端点

### 1. 声音生成 (Custom Voice)

**端点**: `POST /qwenapi/v1/custom-voice`

**请求示例**:

```python
import requests
import base64

url = "http://127.0.0.1:7860/qwenapi/v1/custom-voice"
data = {
    "model_name": "Qwen/Qwen2.5-TTS-Custom",
    "text": "你好,这是一个测试。",
    "instruct": "用温柔的语气说。",
    "speaker": None,  # 可选,默认使用第一个发言人
    "language": None,  # 可选,默认自动检测
    "segment_gen": False  # 是否分段生成
}

response = requests.post(url, json=data)
result = response.json()

# 保存生成的音频
for i, audio_base64 in enumerate(result["audio_files_base64"]):
    audio_bytes = base64.b64decode(audio_base64)
    with open(f"output_{i}.wav", "wb") as f:
        f.write(audio_bytes)

print(result["info"])
```

### 2. 声音设计 (Voice Design)

**端点**: `POST /qwenapi/v1/voice-design`

**请求示例**:

```python
import requests
import base64

url = "http://127.0.0.1:7860/qwenapi/v1/voice-design"
data = {
    "model_name": "Qwen/Qwen2.5-TTS-Design",
    "text": "你好,这是一个测试。",
    "instruct": "体现撒娇稚嫩的女声,音调偏高且起伏明显。",
    "language": None,  # 可选,默认自动检测
    "segment_gen": False
}

response = requests.post(url, json=data)
result = response.json()

# 保存生成的音频
for i, audio_base64 in enumerate(result["audio_files_base64"]):
    audio_bytes = base64.b64decode(audio_base64)
    with open(f"output_{i}.wav", "wb") as f:
        f.write(audio_bytes)

print(result["info"])
```

### 3. 声音克隆 (Voice Clone)

**端点**: `POST /qwenapi/v1/voice-clone`

**请求示例**:

```python
import requests
import base64

# 读取参考音频并编码为 Base64
with open("reference_audio.wav", "rb") as f:
    ref_audio_base64 = base64.b64encode(f.read()).decode("utf-8")

url = "http://127.0.0.1:7860/qwenapi/v1/voice-clone"
data = {
    "model_name": "Qwen/Qwen2.5-TTS",
    "text": "你好,这是一个测试。",
    "ref_audio_base64": ref_audio_base64,
    "ref_text": "参考音频中说话的内容",  # 可选
    "language": None,  # 可选,默认自动检测
    "segment_gen": False
}

response = requests.post(url, json=data)
result = response.json()

# 保存生成的音频
for i, audio_base64 in enumerate(result["audio_files_base64"]):
    audio_bytes = base64.b64decode(audio_base64)
    with open(f"output_{i}.wav", "wb") as f:
        f.write(audio_bytes)

print(result["info"])
```

### 4. 获取可用模型列表

**端点**: `GET /qwenapi/v1/models`

**请求示例**:

```python
import requests

url = "http://127.0.0.1:7860/qwenapi/v1/models"
response = requests.get(url)
result = response.json()

for model in result["models"]:
    print(f"模型名称: {model['name']}, 类型: {model['type']}")
```

### 5. 获取支持的发言人列表

**端点**: `GET /qwenapi/v1/speakers`

**请求示例**:

```python
import requests

url = "http://127.0.0.1:7860/qwenapi/v1/speakers"
response = requests.get(url)
result = response.json()

print("支持的发言人:", result["speakers"])
```

### 6. 获取支持的语言列表

**端点**: `GET /qwenapi/v1/languages`

**请求示例**:

```python
import requests

url = "http://127.0.0.1:7860/qwenapi/v1/languages"
response = requests.get(url)
result = response.json()

print("支持的语言:", result["languages"])
```

### 7. 获取配置选项

**端点**: `GET /qwenapi/v1/options`

**请求示例**:

```python
import requests

url = "http://127.0.0.1:7860/qwenapi/v1/options"
response = requests.get(url)
result = response.json()

print("当前配置:", result)
```

### 8. 中断当前任务

**端点**: `POST /qwenapi/v1/interrupt`

**请求示例**:

```python
import requests

url = "http://127.0.0.1:7860/qwenapi/v1/interrupt"
response = requests.post(url)
result = response.json()

print(result["message"])
```

## 高级参数

所有生成 API (custom-voice, voice-design, voice-clone) 都支持以下可选的采样参数:

```python
data = {
    "model_name": "Qwen/Qwen2.5-TTS-Custom",
    "text": "你好,这是一个测试。",
    # ... 其他必需参数 ...
    
    # 可选的采样参数
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 0.95,
    "subtalker_temperature": 1.0,
    "max_new_tokens": 2048
}
```

如果不指定这些参数，将使用配置文件中的默认值。

## 完整示例脚本

```python
#!/usr/bin/env python3
"""Qwen TTS API 使用示例"""

import requests
import base64
from pathlib import Path


def custom_voice_example():
    """声音生成示例"""
    url = "http://127.0.0.1:7860/qwenapi/v1/custom-voice"
    data = {
        "model_name": "Qwen/Qwen2.5-TTS-Custom",
        "text": "你好,这是一个声音生成测试。",
        "instruct": "用温柔的语气说。",
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        for i, audio_base64 in enumerate(result["audio_files_base64"]):
            audio_bytes = base64.b64decode(audio_base64)
            output_path = Path(f"custom_voice_{i}.wav")
            output_path.write_bytes(audio_bytes)
            print(f"已保存: {output_path}")
        print(result["info"])
    else:
        print(f"错误: {response.status_code}, {response.text}")


def voice_clone_example():
    """声音克隆示例"""
    # 读取参考音频
    ref_audio_path = Path("reference.wav")
    if not ref_audio_path.exists():
        print(f"参考音频文件不存在: {ref_audio_path}")
        return
    
    ref_audio_base64 = base64.b64encode(ref_audio_path.read_bytes()).decode("utf-8")
    
    url = "http://127.0.0.1:7860/qwenapi/v1/voice-clone"
    data = {
        "model_name": "Qwen/Qwen2.5-TTS",
        "text": "你好,这是一个声音克隆测试。",
        "ref_audio_base64": ref_audio_base64,
        "ref_text": "参考音频的文本内容",
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        for i, audio_base64 in enumerate(result["audio_files_base64"]):
            audio_bytes = base64.b64decode(audio_base64)
            output_path = Path(f"voice_clone_{i}.wav")
            output_path.write_bytes(audio_bytes)
            print(f"已保存: {output_path}")
        print(result["info"])
    else:
        print(f"错误: {response.status_code}, {response.text}")


if __name__ == "__main__":
    print("=== 声音生成示例 ===")
    custom_voice_example()
    
    print("\n=== 声音克隆示例 ===")
    voice_clone_example()
```

## 错误处理

API 返回的错误信息格式:

```json
{
    "detail": "错误描述信息"
}
```

常见错误码:
- `400`: 请求参数错误
- `500`: 服务器内部错误 (如显存不足、模型加载失败等)

## 注意事项
1. 首次使用某个模型时，需要下载模型文件，可能需要较长时间
2. 音频文件使用 Base64 编码传输，大文件可能导致响应较大
3. 分段生成 (`segment_gen=True`) 会按行分割文本并分别生成音频
4. 参考音频需要编码为 Base64 格式传输
5. API 使用队列锁机制，同一时间只能处理一个请求


# 使用的项目
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): 文本转语音。


# 许可证
- [GPL-3.0](LICENSE)
