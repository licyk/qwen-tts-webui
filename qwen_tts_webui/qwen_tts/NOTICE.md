# Vendored Qwen3-TTS inference code

This directory contains the inference code of the [`qwen-tts`](https://pypi.org/project/qwen-tts/) package
version 0.1.1 ([QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)), Copyright 2026 The Alibaba Qwen team,
licensed under the Apache License 2.0 (see `LICENSE` in this directory).

It is integrated into Qwen TTS WebUI so that the decoding loop can be optimized and the package no longer has to be
installed separately.

## Modifications

Modified upstream code is marked with `[qwen-tts-webui]` comments.

- Removed the Gradio demo (`cli/`, `__main__.py`), Qwen TTS WebUI provides its own UI.
- `fast_generation.py` (new): fast decoding loop replacing the per-frame HuggingFace `generate()` calls
  (static KV cache, on-GPU sampling, fused decoder layers and CUDA Graph capture, streaming callback).
- `core/models/modeling_qwen3_tts.py`: `Qwen3TTSForConditionalGeneration.generate()` dispatches to the fast decoder
  when it is enabled; the original HuggingFace path is kept as a fallback. `device_map="auto"` fixes: decoder layers and
  the code predictor are no longer split across devices (`_no_split_modules`), and the speech tokenizer is loaded on a
  single device next to the talker when the model is spread over several devices.
- `inference/qwen3_tts_model.py`: `fast_generation` / `use_cuda_graph` / `fuse_layers` loading options,
  `set_fast_generation()` and `release_fast_generation()`; Auto* registrations use `exist_ok=True`.
- `core/__init__.py`, `inference/qwen3_tts_tokenizer.py`: the 25Hz tokenizer model (which needs the optional
  `sox`, `onnxruntime`, `einops` and `flash-attn` packages) is imported lazily, only when such a tokenizer is loaded.
