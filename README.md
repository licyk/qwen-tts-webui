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
    <a href="https://github.com/licyk/qwen-tts-webui/commits/main">
      <img src="https://flat.badgen.net/github/last-commit/licyk/qwen-tts-webui/main?icon=github&color=green&label=last%20main%20commit" alt="Commit">
    </a>
    <a href="https://github.com/licyk/qwen-tts-webui/actions/workflows/py-lint.yml">
      <img src="https://github.com/licyk/qwen-tts-webui/actions/workflows/py-lint.yml/badge.svg" alt="Ruff Lint">
    </a>
  </p>

</div>

- [Qwen TTS WebUI](#qwen-tts-webui)
- [简介](#简介)
- [安装](#安装)
  - [使用整合包 (仅 Windows)](#使用整合包-仅-windows)
  - [使用安装器 (Windows / Linux / MacOS)](#使用安装器-windows--linux--macos)
  - [手动安装 (Windows / Linux / MacOS)](#手动安装-windows--linux--macos)
  - [云端在线体验](#云端在线体验)
- [使用](#使用)
  - [可使用的模型](#可使用的模型)
  - [启动参数](#启动参数)
  - [生成加速](#生成加速)
- [API 使用](#api-使用)
  - [启动 API 服务](#启动-api-服务)
  - [使用 OpenAI SDK 调用](#使用-openai-sdk-调用)
  - [API 端点](#api-端点)
    - [语音合成 `POST /v1/audio/speech`](#语音合成-post-v1audiospeech)
    - [流式输出](#流式输出)
    - [声音克隆与注册声音](#声音克隆与注册声音)
    - [模型列表 `GET /v1/models`](#模型列表-get-v1models)
    - [声音列表 `GET /v1/audio/voices`](#声音列表-get-v1audiovoices)
    - [中断当前任务 `POST /v1/audio/interrupt`](#中断当前任务-post-v1audiointerrupt)
  - [错误处理](#错误处理)
  - [从旧版 API 迁移](#从旧版-api-迁移)
  - [注意事项](#注意事项)
- [使用的项目](#使用的项目)
- [许可证](#许可证)

***

# 简介
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


## 云端在线体验
[Qwen TTS WebUI Colab Jupyter NoteBook - licyk/sd-webui-all-in-one](https://github.com/licyk/sd-webui-all-in-one?tab=readme-ov-file#qwen-tts-webui-colab-jupyter-notebook)


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
                 [--api] [--nowebui] [--api-key API_KEY]

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
  --api                 启用 API 功能
  --nowebui             仅启动 API 服务器, 不启动 WebUI
  --api-key API_KEY     API 访问密钥, 设置后请求需要携带 `Authorization: Bearer <key>` 请求头 (也可通过 QWEN_TTS_API_KEY 环境变量设置)
```


## 生成加速
Qwen TTS WebUI 内置了 Qwen3-TTS 的推理代码 (不再需要安装 `qwen-tts` 软件包), 并对音频生成过程进行了优化:

- **快速生成**: 上游实现在每生成一帧音频 (80ms) 时都会调用一次 HuggingFace `generate()` 来生成剩余的 15 个码本, 推理被大量的 CPU 调度开销和小 Kernel 拖慢, GPU 大部分时间处于空闲状态. 快速生成使用静态 KV Cache, GPU 端采样, 融合的 Decoder Layer 实现替代该过程, 计算结果与上游一致 (float32 下逐帧完全一致).
- **CUDA Graph**: 在 NVIDIA 显卡上将每一帧的全部计算捕获为一个 CUDA Graph, 每帧只需一次启动, 进一步降低 CPU 开销. 首次生成某个长度区间的音频时需要捕获 CUDA Graph, 会稍慢一些.
- **批量生成**: 启用`分段生成`时, 多段文本会按`分段生成的批量大小`一起生成, 显存不足时会自动减小批量大小.
- **自动选择推理设备**: `推理设备`为`auto`时, 如果单张显卡能放下模型, 将使用空闲显存最多的显卡, 避免模型被拆分到多张显卡上导致速度下降.
- **低显存运行**: 显存不足以完整加载模型时 (`推理设备`为`auto`或者指定的显卡), 每一帧都会使用的模块与语音解码器会保留在显卡上, 部分 Talker 层依次放到其他显卡或者卸载到 CPU (生成速度会降低); 连这些模块都放不下时使用 CPU 推理 (CPU 不支持原生 bfloat16 计算时自动使用 float32, 速度更快). 生成时显存不足会自动减小批量大小, 并在必要时关闭 CUDA Graph 后重试.
- **中断生成**: 点击`终止`后会在当前帧生成完成后立即停止.

以上选项可在`设置`页面的`生成加速设置`中调整, 如果遇到兼容性问题, 可以关闭`快速生成`使用上游原始的生成方式.


# API 使用
Qwen TTS WebUI 提供与 [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 API, 可直接使用 OpenAI 官方 SDK 或任何支持 OpenAI TTS 接口的客户端调用.


## 启动 API 服务
```bash
# 启动 WebUI 并启用 API (WebUI 与 API 共用同一个端口)
python launch.py --api

# 仅启动 API 服务器 (不启动 WebUI)
python launch.py --nowebui

# 自定义服务器地址, 端口, 并启用密钥鉴权
python launch.py --nowebui --server-name 0.0.0.0 --server-port 8080 --api-key sk-your-key
```

访问地址:
- API 端点 (Base URL): `http://127.0.0.1:7860/v1`
- API 文档: `http://127.0.0.1:7860/docs`


## 使用 OpenAI SDK 调用
```python
from openai import OpenAI

# 未设置 --api-key 时 api_key 可以填写任意值
client = OpenAI(base_url="http://127.0.0.1:7860/v1", api_key="sk-your-key")

# 生成音频并保存
response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # 也可以使用 OpenAI 模型别名, 例如 tts-1 / tts-1-hd / gpt-4o-mini-tts
    voice="vivian",  # 发言人, 也可以使用 OpenAI 发言人别名, 例如 alloy / nova / onyx
    input="你好, 欢迎使用 Qwen TTS WebUI.",
    instructions="用温柔的语气说",  # 可选, 描述声音的特征
    response_format="mp3",
)
response.write_to_file("speech.mp3")

# 流式输出: 边生成边接收 PCM 音频 (24kHz, 16bit, 单声道)
with client.audio.speech.with_streaming_response.create(
    model="tts-1-hd",
    voice="serena",
    input="流式输出可以显著降低首段音频的等待时间.",
    response_format="pcm",
) as response:
    with open("speech.pcm", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

# Qwen TTS 的扩展参数通过 extra_body 传递
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="ryan",
    input="Hello from Qwen TTS.",
    extra_body={"language": "english", "seed": 42, "temperature": 0.8},
)
```


## API 端点

### 语音合成 `POST /v1/audio/speech`
与 OpenAI 相同的参数:

|参数|类型|说明|
|---|---|---|
|`model`|string|**必填**. Qwen TTS 模型名称 (见`GET /v1/models`), 或 OpenAI 模型别名: `tts-1` (0.6B CustomVoice), `tts-1-hd` / `gpt-4o-mini-tts` (1.7B CustomVoice)|
|`input`|string|**必填**. 待合成文本|
|`voice`|string|声音生成模型: 发言人名称 (`vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`) 或 OpenAI 发言人别名 (`alloy`, `nova`, `onyx` 等), 默认使用第一个发言人; 声音克隆模型: 通过`POST /v1/audio/voices`注册的声音名称; 声音设计模型: 忽略|
|`instructions`|string|声音生成模型: 声音特征描述 (可选); 声音设计模型: 声音描述 (**必填**)|
|`response_format`|string|`mp3` (默认), `opus`, `flac`, `wav`, `pcm`. 暂不支持 `aac`|
|`speed`|number|语速, 范围 0.25 ~ 4.0, 默认 1.0|
|`stream_format`|string|`audio` (默认) 或 `sse`, 见[流式输出](#流式输出)|

Qwen TTS 扩展参数 (使用 OpenAI SDK 时通过 `extra_body` 传递):

|参数|类型|说明|
|---|---|---|
|`language`|string|语言, 例如 `chinese`, `english`, `japanese`, 默认自动检测|
|`ref_audio`|string|声音克隆参考音频 (Base64 或 data URI), 设置后忽略 `voice`|
|`ref_text`|string|参考音频对应的文本, 为空时仅使用说话人特征进行克隆|
|`segment`|bool|按行切分文本并批量生成, 结果按顺序拼接为一段音频|
|`seed`|int|随机种子, 相同参数和种子生成的音频一致|
|`do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`|-|采样参数, 未设置时使用 WebUI 设置中的值|
|`subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature`|-|子说话者采样参数|
|`max_new_tokens`|int|最大生成 Token 数 (每个 Token 对应 80ms 音频)|

非流式响应的响应头中包含生成统计信息: `X-Audio-Duration` (音频时长), `X-Generation-Time` (生成耗时), `X-Real-Time-Factor` (实时率).

```bash
curl http://127.0.0.1:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1-hd", "voice": "vivian", "input": "你好, 世界.", "response_format": "wav"}' \
  --output speech.wav
```

声音设计:

```python
response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    voice="default",
    input="哥哥, 你回来啦.",
    instructions="体现撒娇稚嫩的萝莉女声, 音调偏高且起伏明显.",
)
```


### 流式输出
- `stream_format="audio"`: 当 `response_format` 为 `pcm` 或 `wav` 时, 音频会在生成过程中分块返回 (首段音频通常在 0.5 秒内返回), `wav` 格式的文件头中的长度字段为 `0xFFFFFFFF`; 其他格式在生成完成后一次性返回.
- `stream_format="sse"`: 以 Server-Sent Events 的方式返回, 每个音频片段为一个 `speech.audio.delta` 事件 (`audio` 字段为 Base64 编码的音频), 结束时返回包含用量信息的 `speech.audio.done` 事件. 推荐配合 `pcm` 格式使用.

```text
data: {"type": "speech.audio.delta", "audio": "<base64>"}

data: {"type": "speech.audio.done", "usage": {"input_tokens": 22, "output_tokens": 96, "total_tokens": 118}}
```


### 声音克隆与注册声音
声音克隆模型 (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`) 需要参考音频. 可以先注册一个声音, 之后通过 `voice` 参数使用 (参考音频的特征提取结果会被缓存):

```python
import base64
import httpx

audio = base64.b64encode(open("reference.wav", "rb").read()).decode()
httpx.post(
    "http://127.0.0.1:7860/v1/audio/voices",
    json={"name": "my_voice", "audio": audio, "ref_text": "参考音频中说话的内容"},
)

response = client.audio.speech.create(model="Qwen/Qwen3-TTS-12Hz-1.7B-Base", voice="my_voice", input="这是克隆的声音.")
```

也可以在请求中直接传入参考音频: `extra_body={"ref_audio": audio, "ref_text": "..."}`.

- `POST /v1/audio/voices`: 注册声音, 参数为 `name` (字母, 数字, `_`, `.`, `-`), `audio` (Base64 或 data URI, 支持 wav / flac / ogg / mp3), `ref_text` (可选, 为空时仅使用说话人特征进行克隆). 注册的声音保存在 `voices` 文件夹中.
- `DELETE /v1/audio/voices/{name}`: 删除注册的声音.


### 模型列表 `GET /v1/models`
返回可用的模型 (包括设置中添加的额外模型) 以及 OpenAI 模型别名, `type` 字段表示模型类型 (`custom_voice` / `voice_design` / `voice_clone`).


### 声音列表 `GET /v1/audio/voices`
返回内置发言人 (`speaker`), OpenAI 发言人别名 (`alias`) 以及注册的声音 (`clone`).


### 中断当前任务 `POST /v1/audio/interrupt`
中断正在进行的生成任务, 被中断的请求返回 `503` 错误 (`code` 为 `generation_interrupted`).


## 错误处理
错误响应与 OpenAI 的格式一致, OpenAI SDK 会将其转换为对应的异常 (例如 `openai.NotFoundError`):

```json
{
    "error": {
        "message": "The model `xxx` does not exist",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found"
    }
}
```

|状态码|说明|
|---|---|
|`400`|请求参数错误 (例如发言人或语言不存在, 声音设计模型缺少 `instructions`)|
|`401`|API 密钥错误 (启用 `--api-key` 时)|
|`404`|模型或声音不存在|
|`500`|服务器内部错误|
|`503`|显存不足 (`out_of_memory`) 或任务被中断 (`generation_interrupted`)|


## 从旧版 API 迁移
旧版的 `/qwenapi/v1/*` 接口已被 OpenAI 兼容接口替代:

|旧版接口|新版接口|
|---|---|
|`POST /qwenapi/v1/custom-voice`|`POST /v1/audio/speech` (使用声音生成模型, `text` -> `input`, `speaker` -> `voice`, `instruct` -> `instructions`)|
|`POST /qwenapi/v1/voice-design`|`POST /v1/audio/speech` (使用声音设计模型, `instruct` -> `instructions`)|
|`POST /qwenapi/v1/voice-clone`|`POST /v1/audio/speech` (使用声音克隆模型, `ref_audio_base64` -> `ref_audio`)|
|`GET /qwenapi/v1/models`|`GET /v1/models`|
|`GET /qwenapi/v1/speakers`|`GET /v1/audio/voices`|
|`POST /qwenapi/v1/interrupt`|`POST /v1/audio/interrupt`|

新版接口直接返回音频数据, 而不是包含 Base64 编码音频的 JSON. `segment_gen` 参数改为 `segment`, 并且多段音频会拼接为一段返回.


## 注意事项
1. 首次使用某个模型时, 需要下载模型文件, 可能需要较长时间.
2. API 与 WebUI 共用一个任务队列, 同一时间只会执行一个生成任务, 其他请求会排队等待.
3. API 生成的音频同样会保存到 `outputs` 文件夹中.
4. 只能使用 `GET /v1/models` 中列出的模型, 如需使用其他模型, 请在 WebUI 的`设置`页面中添加到额外模型列表.


# 使用的项目
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): 文本转语音。推理代码 (Apache-2.0) 已内置于 `qwen_tts_webui/qwen_tts`, 修改说明见 [NOTICE.md](qwen_tts_webui/qwen_tts/NOTICE.md)。


# 许可证
- [GPL-3.0](LICENSE)
