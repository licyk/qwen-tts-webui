"""Gradio 前端界面"""

import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from qwen_tts_webui.backend.qwen_backend import QwenTTSBackend
from qwen_tts_webui.task_manager.call_queue import wrap_gradio_call, wrap_queued_call
from qwen_tts_webui.config_manager.config import (
    CONFIG_PATH,
    OUTPUT_PATH,
    QWEN_TTS_BASE_MODEL_LIST,
    QWEN_TTS_CUSTOM_VOICE_MODEL_LIST,
    QWEN_TTS_VOICE_DESIGN_MODEL_LIST,
    ATTN_IMPL_LIST,
)
from qwen_tts_webui.backend.memory_manager import (
    MODEL_PRECISION_LIST,
    get_available_devices,
    OutOfMemoryError,
)
from qwen_tts_webui.config_manager.shared import opts, state

backend = QwenTTSBackend()


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面

    Returns:
        gr.Blocks: Gradio 界面实例
    """
    with gr.Blocks(title="Qwen TTS WebUI") as demo:
        gr.Markdown("# Qwen TTS WebUI")

        with gr.Tabs():
            with gr.Tab("声音生成", id="voice_generation"):
                with gr.Row():
                    with gr.Column():
                        gen_model = gr.Dropdown(label="模型选择", choices=QWEN_TTS_CUSTOM_VOICE_MODEL_LIST, value=QWEN_TTS_CUSTOM_VOICE_MODEL_LIST[0], interactive=True)
                        gen_text = gr.Textbox(label="合成文本", placeholder="请输入要合成的文本...", lines=5)
                        gen_instruct = gr.Textbox(label="声音特征描述", placeholder="例如: 体现撒娇稚嫩的女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。", lines=3)
                        with gr.Row():
                            gen_speaker = gr.Dropdown(label="发言人", choices=["default"], value="default", interactive=True)
                            gen_language = gr.Dropdown(label="语言", choices=["auto"], value="auto", interactive=True)
                    with gr.Column():
                        with gr.Row():
                            gen_button = gr.Button("开始生成", variant="primary")
                            stop_gen_button = gr.Button("终止", variant="stop", visible=False)
                        gen_output = gr.Audio(label="生成的音频", type="filepath")
                        gr.Markdown("## 使用说明\n" \
                        "1. 在`合成文本`中输入要合成的文本, `声音特征描述`描述合成出来的声音是什么特征的, 即可点击`开始生成`.\n" \
                        "2. `发言人`和`语言`选项默认分别只有`default`和`auto`, 当点击`开始生成`并且生成结束后, 这两个选项将会刷新, 刷新后即可选择其他选项.\n" \
                        "3. 生成的音频可在`音频浏览`中查看, 或者在 Qwen TTS WebUI 的 `outputs` 文件夹中查看.")

            with gr.Tab("声音设计", id="voice_design"):
                with gr.Row():
                    with gr.Column():
                        design_model = gr.Dropdown(label="模型选择", choices=QWEN_TTS_VOICE_DESIGN_MODEL_LIST, value=QWEN_TTS_VOICE_DESIGN_MODEL_LIST[0], interactive=True)
                        design_text = gr.Textbox(label="合成文本", placeholder="请输入要合成的文本...", lines=5)
                        design_instruct = gr.Textbox(
                            label="声音特征描述", placeholder="例如：体现撒娇稚嫩的女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。", lines=3
                        )
                        design_language = gr.Dropdown(label="语言", choices=["auto"], value="auto", interactive=True)
                    with gr.Column():
                        with gr.Row():
                            design_button = gr.Button("开始生成", variant="primary")
                            stop_design_button = gr.Button("终止", variant="stop", visible=False)
                        design_output = gr.Audio(label="生成的音频", type="filepath")
                        gr.Markdown("## 使用说明\n" \
                        "1. 在`合成文本`中输入要合成的文本, `声音特征描述`描述合成出来的声音是什么特征的, 即可点击`开始生成`.\n" \
                        "2. `语言`选项默认只有`auto`, 当点击`开始生成`并且生成结束后, 这个选项将会刷新, 刷新后即可选择其他选项.\n" \
                        "3. 生成的音频可在`音频浏览`中查看, 或者在 Qwen TTS WebUI 的 `outputs` 文件夹中查看.")

            with gr.Tab("声音克隆", id="voice_clone"):
                with gr.Row():
                    with gr.Column():
                        clone_model = gr.Dropdown(label="模型选择", choices=QWEN_TTS_BASE_MODEL_LIST, value=QWEN_TTS_BASE_MODEL_LIST[0], interactive=True)
                        clone_text = gr.Textbox(label="合成文本", placeholder="请输入要合成的文本...", lines=5)
                        clone_language = gr.Dropdown(label="语言", choices=["auto"], value="auto", interactive=True)
                        clone_audio = gr.Audio(label="参考音频文件", type="filepath")
                        clone_ref_text = gr.Textbox(label="参考音频文本描述", placeholder="请输入参考音频对应的文本内容 (描述参考音频文件中说话的内容, 可不填)", lines=2)
                    with gr.Column():
                        with gr.Row():
                            clone_button = gr.Button("开始生成", variant="primary")
                            stop_clone_button = gr.Button("终止", variant="stop", visible=False)
                        clone_output = gr.Audio(label="生成的音频", type="filepath")
                        gr.Markdown("## 使用说明\n" \
                        "1. 在`合成文本`中输入要合成的文本, 再将一段音频导入`参考音频文件`中, 即可点击`开始生成`.\n" \
                        "2. `参考音频文本描述 `填写导入的音频中, 音频内说话的内容, 这个选项可不填.\n" \
                        "3. `语言`选项默认只有`auto`, 当点击`开始生成`并且生成结束后, 这个选项将会刷新, 刷新后即可选择其他选项.\n" \
                        "4. 生成的音频可在`音频浏览`中查看, 或者在 Qwen TTS WebUI 的 `outputs` 文件夹中查看.")

            with gr.Tab("音频浏览", id="audio_browser"):
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("刷新文件列表", variant="secondary")
                        explorer = gr.FileExplorer(root_dir=OUTPUT_PATH, glob="**/*.wav", file_count="single", label="选择音频文件")
                    with gr.Column():
                        with gr.Row():
                            audio_player = gr.Audio(label="播放器", type="filepath")

                refresh_btn.click(  # pylint: disable=no-member
                    fn=gr.update,
                    outputs=explorer,
                )
                explorer.change(  # pylint: disable=no-member
                    fn=lambda x: x,
                    inputs=explorer,
                    outputs=audio_player,
                )

            with gr.Tab("设置", id="settings"):
                with gr.Row():
                    save_settings_btn = gr.Button("保存设置", variant="primary")
                    reset_settings_btn = gr.Button("重置设置", variant="secondary")

                api_type = gr.Dropdown(label="下载模型的 API 类型", choices=["huggingface", "modelscope"], value=opts.api_type)

                available_devices = ["auto"] + [str(d) for d in get_available_devices()]
                device_map = gr.Dropdown(label="推理设备", choices=available_devices, value=str(opts.device_map))
                dtype = gr.Dropdown(label="推理精度", choices=[str(p) for p in MODEL_PRECISION_LIST], value=str(opts.dtype))
                attn_implementation = gr.Dropdown(label="加速方案", choices=[str(None)] + ATTN_IMPL_LIST, value=str(opts.attn_implementation))

                do_sample = gr.Checkbox(label="是否使用采样", value=opts.do_sample)
                top_k = gr.Slider(label="Top-k 采样参数", minimum=1, maximum=100, step=1, value=opts.top_k)
                top_p = gr.Slider(label="Top-p 采样参数", minimum=0.0, maximum=1.0, step=0.01, value=opts.top_p)
                temperature = gr.Slider(label="采样温度", minimum=0.0, maximum=2.0, step=0.01, value=opts.temperature)
                repetition_penalty = gr.Slider(label="重复惩罚系数", minimum=1.0, maximum=2.0, step=0.01, value=opts.repetition_penalty)
                subtalker_dosample = gr.Checkbox(label="子说话者采样开关", value=opts.subtalker_dosample)
                subtalker_top_k = gr.Slider(label="子说话者 Top-k 采样", minimum=1, maximum=100, step=1, value=opts.subtalker_top_k)
                subtalker_top_p = gr.Slider(label="子说话者 Top-p 采样", minimum=0.0, maximum=1.0, step=0.01, value=opts.subtalker_top_p)
                subtalker_temperature = gr.Slider(label="子说话者采样温度", minimum=0.0, maximum=2.0, step=0.01, value=opts.subtalker_temperature)
                max_new_tokens = gr.Slider(label="最大生成 Token 数", minimum=1, maximum=4096, step=1, value=opts.max_new_tokens)

                def save_settings(
                    api_type_val: str,
                    device_map_val: str,
                    dtype_val: str,
                    attn_impl_val: str,
                    do_sample_val: bool,
                    top_k_val: int,
                    top_p_val: float,
                    temperature_val: float,
                    repetition_penalty_val: float,
                    subtalker_dosample_val: bool,
                    subtalker_top_k_val: int,
                    subtalker_top_p_val: float,
                    subtalker_temperature_val: float,
                    max_new_tokens_val: int,
                ) -> None:
                    """保存设置

                    Args:
                        api_type_val (str): API 类型
                        device_map_val (str): 推理设备
                        dtype_val (str): 推理精度
                        attn_impl_val (str): 加速方案
                        do_sample_val (bool): 是否使用采样
                        top_k_val (int): Top-k 采样参数
                        top_p_val (float): Top-p 采样参数
                        temperature_val (float): 采样温度
                        repetition_penalty_val (float): 重复惩罚系数
                        subtalker_dosample_val (bool): 子说话者采样开关
                        subtalker_top_k_val (int): 子说话者 Top-k 采样
                        subtalker_top_p_val (float): 子说话者 Top-p 采样
                        subtalker_temperature_val (float): 子说话者采样温度
                        max_new_tokens_val (int): 最大生成 Token 数
                    """
                    opts.api_type = api_type_val
                    opts.device_map = device_map_val
                    opts.dtype = dtype_val
                    opts.attn_implementation = None if attn_impl_val == str(None) else attn_impl_val
                    opts.do_sample = do_sample_val
                    opts.top_k = top_k_val
                    opts.top_p = top_p_val
                    opts.temperature = temperature_val
                    opts.repetition_penalty = repetition_penalty_val
                    opts.subtalker_dosample = subtalker_dosample_val
                    opts.subtalker_top_k = subtalker_top_k_val
                    opts.subtalker_top_p = subtalker_top_p_val
                    opts.subtalker_temperature = subtalker_temperature_val
                    opts.max_new_tokens = max_new_tokens_val
                    opts.save(CONFIG_PATH)
                    gr.Info("设置已保存")

                save_settings_btn.click(  # pylint: disable=no-member
                    fn=save_settings,
                    inputs=[
                        api_type,
                        device_map,
                        dtype,
                        attn_implementation,
                        do_sample,
                        top_k,
                        top_p,
                        temperature,
                        repetition_penalty,
                        subtalker_dosample,
                        subtalker_top_k,
                        subtalker_top_p,
                        subtalker_temperature,
                        max_new_tokens,
                    ],
                )

                def reset_settings() -> list[Any]:
                    """重置设置

                    Returns:
                        list[Any]: 包含所有重置后设置项值的列表, 用于更新 Gradio 组件
                    """
                    opts.reset()
                    opts.save(CONFIG_PATH)
                    gr.Info("设置已重置为默认值")
                    return [
                        opts.api_type,
                        str(opts.device_map),
                        str(opts.dtype),
                        str(opts.attn_implementation),
                        opts.do_sample,
                        opts.top_k,
                        opts.top_p,
                        opts.temperature,
                        opts.repetition_penalty,
                        opts.subtalker_dosample,
                        opts.subtalker_top_k,
                        opts.subtalker_top_p,
                        opts.subtalker_temperature,
                        opts.max_new_tokens,
                    ]

                reset_settings_btn.click(  # pylint: disable=no-member
                    fn=reset_settings,
                    inputs=[],
                    outputs=[
                        api_type,
                        device_map,
                        dtype,
                        attn_implementation,
                        do_sample,
                        top_k,
                        top_p,
                        temperature,
                        repetition_penalty,
                        subtalker_dosample,
                        subtalker_top_k,
                        subtalker_top_p,
                        subtalker_temperature,
                        max_new_tokens,
                    ],
                )

        def update_metadata(
            current_speaker: str,
            current_language: str,
        ) -> tuple[str, str | None, Any, Any]:
            """更新模型元数据

            Args:
                current_speaker (str):
                    当前选择的发言人
                current_language (str):
                    当前选择的语言

            Returns:
                (tuple[str, str | None, Any, Any]): 实际发言人, 实际语言, 发言人组件更新, 语言组件更新
            """
            speakers = backend.get_supported_speakers() or []
            languages = backend.get_supported_languages() or []

            speaker_choices = ["default"] + speakers
            if "auto" in languages:
                language_choices = languages
            else:
                language_choices = ["auto"] + languages

            actual_speaker = current_speaker
            if current_speaker == "default" and speakers:
                actual_speaker = speakers[0]

            actual_language = None if current_language == "auto" else current_language

            return (
                actual_speaker,
                actual_language,
                gr.update(choices=speaker_choices, value=current_speaker),
                gr.update(choices=language_choices, value=current_language),
            )

        def update_metadata_simple(
            current_language: str,
        ) -> tuple[str | None, Any]:
            """更新模型元数据 (仅语言)

            Args:
                current_language (str): 当前选择的语言

            Returns:
                tuple[str | None, Any]: 实际语言, 语言组件更新
            """
            languages = backend.get_supported_languages() or []
            if "auto" in languages:
                language_choices = languages
            else:
                language_choices = ["auto"] + languages
            actual_language = None if current_language == "auto" else current_language
            return actual_language, gr.update(choices=language_choices, value=current_language)

        def generate_voice_fn(
            model_name: str,
            text: str,
            instruct: str,
            speaker: str,
            language: str,
        ) -> tuple[str | None, Any, Any]:
            """声音生成处理函数

            Args:
                model_name (str): 所选模型名称
                text (str): 待合成文本
                instruct (str): 声音特征描述指令
                speaker (str): 发言人标识
                language (str): 语言标识

            Returns:
                (tuple[str | None, Any, Any]): 生成的音频路径, 发言人组件更新, 语言组件更新
            """
            try:
                backend.load_model(
                    model_name=model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )
                actual_speaker, actual_language, speaker_update, language_update = update_metadata(speaker, language)

                output_path = backend.generate_custom_voice(
                    text=text,
                    speaker=actual_speaker,
                    language=actual_language,
                    instruct=instruct,
                    do_sample=opts.do_sample,
                    top_k=opts.top_k,
                    top_p=opts.top_p,
                    temperature=opts.temperature,
                    repetition_penalty=opts.repetition_penalty,
                    subtalker_dosample=opts.subtalker_dosample,
                    subtalker_top_k=opts.subtalker_top_k,
                    subtalker_top_p=opts.subtalker_top_p,
                    subtalker_temperature=opts.subtalker_temperature,
                    max_new_tokens=opts.max_new_tokens,
                )
                if state.interrupted:
                    return None, speaker_update, language_update
                return str(output_path), speaker_update, language_update
            except OutOfMemoryError as e:
                traceback.print_exc()
                gr.Warning(f"生成音频所需的显存不足: {str(e)}")
                return None, gr.update(), gr.update()
            except Exception as e:  # pylint: disable=duplicate-except
                traceback.print_exc()
                gr.Warning(f"生成失败: {str(e)}")
                return None, gr.update(), gr.update()

        def generate_design_fn(
            model_name: str,
            text: str,
            instruct: str,
            language: str,
        ) -> tuple[str | None, Any]:
            """声音设计处理函数

            Args:
                model_name (str): 所选模型名称
                text (str): 待合成文本
                instruct (str): 声音描述指令
                language (str): 语言标识

            Returns:
                (tuple[str | None, Any]): 生成的音频路径, 语言组件更新
            """
            try:
                backend.load_model(
                    model_name=model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )
                actual_language, language_update = update_metadata_simple(language)

                output_path = backend.generate_voice_design(
                    text=text,
                    instruct=instruct,
                    language=actual_language,
                    do_sample=opts.do_sample,
                    top_k=opts.top_k,
                    top_p=opts.top_p,
                    temperature=opts.temperature,
                    repetition_penalty=opts.repetition_penalty,
                    subtalker_dosample=opts.subtalker_dosample,
                    subtalker_top_k=opts.subtalker_top_k,
                    subtalker_top_p=opts.subtalker_top_p,
                    subtalker_temperature=opts.subtalker_temperature,
                    max_new_tokens=opts.max_new_tokens,
                )
                if state.interrupted:
                    return None, language_update
                return str(output_path), language_update
            except OutOfMemoryError as e:
                traceback.print_exc()
                gr.Warning(f"生成音频所需的显存不足: {str(e)}")
                return None, gr.update(), gr.update()
            except Exception as e:  # pylint: disable=duplicate-except
                traceback.print_exc()
                gr.Warning(f"生成失败: {str(e)}")
                return None, gr.update(), gr.update()

        def generate_clone_fn(
            model_name: str,
            text: str,
            language: str,
            ref_audio: str,
            ref_text: str,
        ) -> tuple[str | None, Any]:
            """声音克隆处理函数

            Args:
                model_name (str): 所选模型名称
                text (str): 待合成文本
                language (str): 语言标识
                ref_audio (str): 参考音频路径
                ref_text (str): 参考音频的文本描述内容

            Returns:
                (tuple[str | None, Any]): 生成的音频路径, 语言组件更新
            """
            try:
                if not ref_audio:
                    gr.Warning("请先上传参考音频文件")
                    return None, gr.update()

                backend.load_model(
                    model_name=model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )
                actual_language, language_update = update_metadata_simple(language)

                output_path = backend.generate_voice_clone(
                    text=text,
                    language=actual_language,
                    ref_audio=Path(ref_audio),
                    ref_text=ref_text,
                    x_vector_only_mode=(ref_text is None or ref_text.strip() == ""),
                    do_sample=opts.do_sample,
                    top_k=opts.top_k,
                    top_p=opts.top_p,
                    temperature=opts.temperature,
                    repetition_penalty=opts.repetition_penalty,
                    subtalker_dosample=opts.subtalker_dosample,
                    subtalker_top_k=opts.subtalker_top_k,
                    subtalker_top_p=opts.subtalker_top_p,
                    subtalker_temperature=opts.subtalker_temperature,
                    max_new_tokens=opts.max_new_tokens,
                )
                if state.interrupted:
                    return None, language_update
                return str(output_path), language_update
            except OutOfMemoryError as e:
                traceback.print_exc()
                gr.Warning(f"生成音频所需的显存不足: {str(e)}")
                return None, gr.update(), gr.update()
            except Exception as e:  # pylint: disable=duplicate-except
                traceback.print_exc()
                gr.Warning(f"生成失败: {str(e)}")
                return None, gr.update(), gr.update()

        def interrupt_fn() -> None:
            """中断任务"""
            state.interrupt()

        gen_event = gen_button.click(  # pylint: disable=no-member
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[gen_button, stop_gen_button],
            queue=False,
        ).then(
            fn=wrap_queued_call(wrap_gradio_call(generate_voice_fn)),
            inputs=[gen_model, gen_text, gen_instruct, gen_speaker, gen_language],
            outputs=[gen_output, gen_speaker, gen_language],
        )
        gen_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[gen_button, stop_gen_button],
            queue=False,
        )
        stop_gen_button.click(  # pylint: disable=no-member
            fn=lambda: (interrupt_fn(), gr.update(visible=True), gr.update(visible=False))[1:],
            cancels=[gen_event],
            outputs=[gen_button, stop_gen_button],
            queue=False,
        )

        design_event = design_button.click(  # pylint: disable=no-member
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[design_button, stop_design_button],
            queue=False,
        ).then(
            fn=wrap_queued_call(wrap_gradio_call(generate_design_fn)),
            inputs=[design_model, design_text, design_instruct, design_language],
            outputs=[design_output, design_language],
        )
        design_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[design_button, stop_design_button],
            queue=False,
        )
        stop_design_button.click(  # pylint: disable=no-member
            fn=lambda: (interrupt_fn(), gr.update(visible=True), gr.update(visible=False))[1:],
            cancels=[design_event],
            outputs=[design_button, stop_design_button],
            queue=False,
        )

        clone_event = clone_button.click(  # pylint: disable=no-member
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[clone_button, stop_clone_button],
            queue=False,
        ).then(
            fn=wrap_queued_call(wrap_gradio_call(generate_clone_fn)),
            inputs=[clone_model, clone_text, clone_language, clone_audio, clone_ref_text],
            outputs=[clone_output, clone_language],
        )
        clone_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[clone_button, stop_clone_button],
            queue=False,
        )
        stop_clone_button.click(  # pylint: disable=no-member
            fn=lambda: (interrupt_fn(), gr.update(visible=True), gr.update(visible=False))[1:],
            cancels=[clone_event],
            outputs=[clone_button, stop_clone_button],
            queue=False,
        )

    return demo
