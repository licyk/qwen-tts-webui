"""Qwen3 TTS 快速自回归解码

上游实现在每生成一帧音频 (12Hz, 即每 80ms 音频) 时, 都会调用一次 HuggingFace `generate()` 让 Code Predictor 生成剩余 15 个码本,
每次调用都需要重新构建 Logits Processor / Stopping Criteria / DynamicCache / Attention Mask, 并且伴随大量的小 Kernel 启动和 CPU-GPU 同步.
这使得推理完全受 CPU 调度开销限制, GPU 大部分时间处于空闲状态.

该模块使用以下方式替代上游的解码循环:
1. 使用预分配的静态 KV Cache, 直接调用上游的 Decoder Layer, 保持与上游一致的数值计算.
2. 将 Code Predictor 的 15 步解码展开为固定形状的计算, 并在 GPU 上完成采样, 不产生任何 CPU-GPU 同步.
3. 在 CUDA 设备上将 "Talker 一步 + Code Predictor 15 步" 捕获为一个 CUDA Graph, 每帧仅需一次 Graph 启动.
4. 停止条件检查延迟一步进行, CPU 不会阻塞 GPU 的任务队列.
5. 支持在生成过程中通过回调函数中断生成.

采样逻辑与 HuggingFace `generate()` 保持一致:
Repetition Penalty -> Min New Tokens -> Suppress Tokens -> Temperature -> Top-k -> Top-p -> 采样.
"""

import contextlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

_KV_BUCKET = 256
"""Talker 注意力窗口的分桶大小, 每个分桶对应一个 CUDA Graph"""

_INITIAL_FRAMES = 512
"""初次分配 KV Cache 时预留的帧数"""

_MAX_GRAPHS_PER_WORKSPACE = 32
"""每个工作区缓存的 CUDA Graph 数量上限"""

_MAX_WORKSPACES = 2
"""缓存的工作区 (不同批次大小) 数量上限"""


class GenerationStoppedError(RuntimeError):
    """`should_stop` 回调要求中断生成"""


@dataclass(frozen=True)
class SamplingConfig:
    """采样参数"""

    do_sample: bool
    """是否使用采样, 为 False 时使用贪婪解码"""

    top_k: int
    """Top-k 采样参数, 为 0 时禁用"""

    top_p: float
    """Top-p 采样参数, 大于等于 1.0 时禁用"""

    temperature: float
    """采样温度"""

    repetition_penalty: float = 1.0
    """重复惩罚系数, 为 1.0 时禁用"""

    @classmethod
    def create(
        cls,
        do_sample: bool | None,
        top_k: int | None,
        top_p: float | None,
        temperature: float | None,
        repetition_penalty: float | None = None,
        vocab_size: int | None = None,
    ) -> "SamplingConfig":
        """规范化采样参数, 使参数相同的配置可复用同一个 CUDA Graph

        Args:
            do_sample (bool | None):
                是否使用采样
            top_k (int | None):
                Top-k 采样参数
            top_p (float | None):
                Top-p 采样参数
            temperature (float | None):
                采样温度, 小于等于 0 时使用贪婪解码
            repetition_penalty (float | None):
                重复惩罚系数
            vocab_size (int | None):
                词表大小, 用于限制 Top-k 的上限

        Returns:
            SamplingConfig: 规范化后的采样参数
        """
        do_sample = bool(do_sample) and temperature is not None and temperature > 0
        top_k = int(top_k or 0)
        if vocab_size is not None:
            top_k = min(top_k, vocab_size)
        top_p = 1.0 if top_p is None else float(top_p)
        penalty = 1.0 if repetition_penalty is None else float(repetition_penalty)
        if not do_sample:
            return cls(do_sample=False, top_k=0, top_p=1.0, temperature=1.0, repetition_penalty=penalty)
        return cls(do_sample=True, top_k=max(top_k, 0), top_p=top_p, temperature=float(temperature), repetition_penalty=penalty)


def _sample_tokens(
    scores: torch.Tensor,
    cfg: SamplingConfig,
) -> torch.Tensor:
    """根据处理后的 Logits 采样 Token, 不产生 CPU-GPU 同步

    Args:
        scores (torch.Tensor):
            形状为 `[B, V]` 的 float32 Logits
        cfg (SamplingConfig):
            采样参数

    Returns:
        torch.Tensor: 形状为 `[B]` 的 Token ID
    """
    if not cfg.do_sample:
        return scores.argmax(dim=-1)
    if cfg.temperature != 1.0:
        scores = scores / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(scores, cfg.top_k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth, -math.inf)
    if cfg.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_to_remove = cumulative_probs <= (1 - cfg.top_p)
        sorted_to_remove[..., -1:] = False
        to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
        scores = scores.masked_fill(to_remove, -math.inf)
    probs = torch.softmax(scores, dim=-1)
    # 指数分布竞争采样, 与 torch.multinomial 分布等价, 但可以被 CUDA Graph 捕获且不需要同步
    noise = torch.empty_like(probs).exponential_().clamp_min_(torch.finfo(probs.dtype).tiny)
    return (probs / noise).argmax(dim=-1)


def _rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """与上游 Qwen3TTSRMSNorm 等价 (float32 计算后转换回原精度再乘权重), 但只需要 2 个 Kernel"""
    return weight * F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=eps)


def _rotate_half(
    x: torch.Tensor,
) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _fuse_linears(
    linears: list[nn.Linear],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """将多个 Linear 的权重拼接为一个矩阵, 并让原 Linear 的权重变为该矩阵的视图 (不额外占用显存)

    Args:
        linears (list[nn.Linear]):
            输入维度相同的 Linear 层

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: 融合后的权重与偏置
    """
    weight = torch.cat([layer.weight.detach() for layer in linears], dim=0)
    bias = None
    if any(layer.bias is not None for layer in linears):
        bias = torch.cat([layer.bias.detach() if layer.bias is not None else weight.new_zeros(layer.out_features) for layer in linears], dim=0)
    offset = 0
    for layer in linears:
        n = layer.out_features
        layer.weight = nn.Parameter(weight[offset : offset + n], requires_grad=False)
        if layer.bias is not None:
            layer.bias = nn.Parameter(bias[offset : offset + n], requires_grad=False)
        offset += n
    return weight, bias


_SDPA_SUPPORTS_GQA = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5)


class _FusedDecoderLayer:
    """融合权重的 Decoder Layer 推理实现

    与上游 `Qwen3TTSTalkerDecoderLayer` / `Qwen3TTSDecoderLayer` 数学等价, 区别在于:
    1. Q / K / V 投影与 Gate / Up 投影各合并为一次矩阵乘法.
    2. RMSNorm 使用融合 Kernel, Q / K Norm 合并为一次计算.
    3. 旋转位置编码在所有层之间共享, 只计算一次.
    """

    def __init__(
        self,
        layer: nn.Module,
    ) -> None:
        """初始化融合层

        Args:
            layer (nn.Module):
                上游的 Decoder Layer
        """
        attn = layer.self_attn
        mlp = layer.mlp
        self.layer_idx = attn.layer_idx
        self.num_heads = attn.config.num_attention_heads
        self.num_kv_heads = attn.config.num_key_value_heads
        self.head_dim = attn.head_dim
        self.kv_groups = attn.num_key_value_groups
        self.scaling = attn.scaling

        fused = getattr(layer, "_qwen_tts_webui_fused", None)
        if fused is None:
            fused = (*_fuse_linears([attn.q_proj, attn.k_proj, attn.v_proj]), *_fuse_linears([mlp.gate_proj, mlp.up_proj]))
            layer._qwen_tts_webui_fused = fused  # pylint: disable=protected-access
        self.qkv_weight, self.qkv_bias, self.gate_up_weight, self.gate_up_bias = fused
        self.o_proj = attn.o_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

        self.input_norm_weight = layer.input_layernorm.weight
        self.input_norm_eps = layer.input_layernorm.variance_epsilon
        self.post_norm_weight = layer.post_attention_layernorm.weight
        self.post_norm_eps = layer.post_attention_layernorm.variance_epsilon
        self.split_qk_norm = attn.q_norm.variance_epsilon != attn.k_norm.variance_epsilon
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.qk_norm_weight = torch.cat(
            [attn.q_norm.weight.detach().expand(self.num_heads, -1), attn.k_norm.weight.detach().expand(self.num_kv_heads, -1)],
            dim=0,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: "_StaticCache",
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        """前向计算

        Args:
            hidden_states (torch.Tensor):
                形状为 `[B, T, hidden]` 的输入
            cos (torch.Tensor):
                形状为 `[B or 1, T, 1, head_dim]` 的旋转位置编码
            sin (torch.Tensor):
                形状为 `[B or 1, T, 1, head_dim]` 的旋转位置编码
            cache (_StaticCache):
                静态 KV Cache
            attention_mask (torch.Tensor | None):
                加性注意力掩码, 为 None 时注意力覆盖缓存中全部可见位置
            is_causal (bool):
                未提供掩码时是否使用因果掩码

        Returns:
            torch.Tensor: 输出
        """
        batch_size, seq_len, _ = hidden_states.shape
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        # 模型被拆分到多个设备上时, 将激活值移动到当前层所在的设备 (位于同一设备时 `to()` 不会产生任何操作)
        device = self.qkv_weight.device
        hidden_states, cos, sin = hidden_states.to(device), cos.to(device), sin.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None

        h = _rms_norm(hidden_states, self.input_norm_weight, self.input_norm_eps)
        qkv = F.linear(h, self.qkv_weight, self.qkv_bias).view(batch_size, seq_len, nh + 2 * nkv, hd)
        if self.split_qk_norm:
            qk = torch.cat([self.q_norm(qkv[:, :, :nh]), self.k_norm(qkv[:, :, nh : nh + nkv])], dim=2)
        else:
            qk = _rms_norm(qkv[:, :, : nh + nkv], self.qk_norm_weight, self.q_norm.variance_epsilon)
        qk = (qk * cos) + (_rotate_half(qk) * sin)
        query = qk[:, :, :nh].transpose(1, 2)
        key = qk[:, :, nh:].transpose(1, 2)
        value = qkv[:, :, nh + nkv :].transpose(1, 2)
        key, value = cache.update(key, value, self.layer_idx)

        if attention_mask is None and _SDPA_SUPPORTS_GQA:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=self.scaling, enable_gqa=self.kv_groups > 1)
        else:
            attn = F.scaled_dot_product_attention(
                query,
                _repeat_kv(key, self.kv_groups),
                _repeat_kv(value, self.kv_groups),
                attn_mask=attention_mask,
                is_causal=is_causal and attention_mask is None,
                scale=self.scaling,
            )
        hidden_states = hidden_states + self.o_proj(attn.transpose(1, 2).reshape(batch_size, seq_len, nh * hd))

        h = _rms_norm(hidden_states, self.post_norm_weight, self.post_norm_eps)
        gate, up = F.linear(h, self.gate_up_weight, self.gate_up_bias).chunk(2, dim=-1)
        return hidden_states + self.down_proj(self.act_fn(gate) * up)


class _StaticCache:
    """静态 KV Cache, 仅实现上游 Attention 模块用到的 `update()` 接口"""

    def __init__(
        self,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> None:
        """初始化静态 KV Cache

        Args:
            keys (list[torch.Tensor]):
                每一层的 Key 缓存, 形状为 `[B, H, L, D]`
            values (list[torch.Tensor]):
                每一层的 Value 缓存, 形状为 `[B, H, L, D]`
        """
        self.keys = keys
        self.values = values
        self.write_pos: torch.Tensor | None = None
        """本次前向写入的位置"""
        self.kv_len = 0
        """本次前向注意力可见的缓存长度"""

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入当前步的 KV 并返回注意力可见的缓存

        Args:
            key_states (torch.Tensor):
                当前步的 Key
            value_states (torch.Tensor):
                当前步的 Value
            layer_idx (int):
                层序号
            cache_kwargs (dict[str, Any] | None):
                未使用, 仅为兼容 HuggingFace Cache 接口

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 注意力可见的 Key / Value
        """
        k = self.keys[layer_idx]
        v = self.values[layer_idx]
        write_pos = self.write_pos.to(k.device)
        k.index_copy_(2, write_pos, key_states.to(k.device))
        v.index_copy_(2, write_pos, value_states.to(v.device))
        return k[:, :, : self.kv_len], v[:, :, : self.kv_len]


@dataclass
class _Workspace:
    """某个批次大小下解码使用的全部静态缓冲区

    CUDA Graph 记录的是张量地址, 因此所有在 Graph 中读写的张量都必须保存在工作区中并在多次请求之间复用.
    """

    batch_size: int
    kv_capacity: int
    text_capacity: int
    talker_cache: _StaticCache
    cp_cache: _StaticCache
    trailing: torch.Tensor
    codes: torch.Tensor
    hiddens: torch.Tensor
    kv_arange: torch.Tensor
    pad_len: torch.Tensor
    step: torch.Tensor
    cur_pos: torch.Tensor
    last_token: torch.Tensor
    past_hidden: torch.Tensor
    seen: torch.Tensor
    unfinished: torch.Tensor
    graphs: "OrderedDict[tuple[Any, ...], torch.cuda.CUDAGraph]" = field(default_factory=OrderedDict)

    def state_tensors(self) -> list[torch.Tensor]:
        """会被单步解码修改的状态张量 (不包含按位置覆盖写入的缓冲区)

        Returns:
            list[torch.Tensor]: 状态张量列表
        """
        return [self.step, self.cur_pos, self.last_token, self.past_hidden, self.seen, self.unfinished]


@dataclass
class GenerationStats:
    """最近一次生成的统计信息"""

    batch_size: int = 0
    prompt_len: int = 0
    steps: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    graph_captures: int = 0
    capture_seconds: float = 0.0
    used_cuda_graph: bool = False
    interrupted: bool = False


class Qwen3TTSFastGenerator:
    """Qwen3 TTS Talker + Code Predictor 快速解码器"""

    def __init__(
        self,
        model: Any,
        use_cuda_graph: bool = True,
        fuse_layers: bool = True,
    ) -> None:
        """初始化快速解码器

        Args:
            model (Qwen3TTSForConditionalGeneration):
                已加载的 Qwen3 TTS 模型
            use_cuda_graph (bool):
                在 CUDA 设备上是否使用 CUDA Graph
            fuse_layers (bool):
                是否使用融合权重的 Decoder Layer 实现 (仅在模型位于单个设备时生效),
                为 False 时直接调用上游的 Decoder Layer, 计算结果与上游完全一致
        """
        self.model = model
        self.talker = model.talker
        self.code_predictor = model.talker.code_predictor
        talker_config = model.config.talker_config
        self.num_code_groups = talker_config.num_code_groups
        self.vocab_size = talker_config.vocab_size
        self.cp_vocab_size = self.code_predictor.config.vocab_size

        self.main_device = self.talker.model.codec_embedding.weight.device
        self.dtype = self.talker.model.codec_embedding.weight.dtype
        devices = {p.device for p in self.talker.parameters()}
        self.single_device = len(devices) == 1
        self.use_cuda_graph = bool(use_cuda_graph) and self.single_device and self.main_device.type == "cuda"

        self.workspaces: OrderedDict[int, _Workspace] = OrderedDict()
        self.graph_pool = None
        self.capture_stream = None
        """CUDA Graph 捕获使用的 Stream. `torch.cuda.graph` 默认使用一个全局共享的 Stream (创建于首次捕获时的设备上),
        模型切换到其他显卡后会在错误的设备上捕获, 因此需要为每个模型所在的设备单独创建"""
        self.last_stats = GenerationStats()
        self.suppress_masks: dict[tuple[int, ...], torch.Tensor] = {}
        """CUDA Graph 会记录张量地址, 因此 Suppress Mask 需要长期保存"""

        attn_configs = {}
        for layer in list(self.talker.model.layers) + list(self.code_predictor.model.layers):
            attn_configs[id(layer.self_attn.config)] = layer.self_attn.config
        self.attn_configs = list(attn_configs.values())

        # Code Predictor 每一步的写入位置 / 位置编码 / 注意力掩码均为常量
        dev = self.code_predictor.lm_head[0].weight.device
        self.cp_device = dev
        self.cp_write_pos = [torch.arange(0, 2, device=dev)] + [torch.tensor([j + 1], device=dev) for j in range(1, self.num_code_groups - 1)]
        self.cp_position_ids = [torch.arange(0, 2, device=dev).unsqueeze(0)] + [torch.tensor([[j + 1]], device=dev) for j in range(1, self.num_code_groups - 1)]
        neg = torch.finfo(self.dtype).min
        self.cp_prefill_mask = torch.tensor([[0.0, neg], [0.0, 0.0]], dtype=self.dtype, device=dev).view(1, 1, 2, 2)

        self.mask_zero = torch.zeros((), dtype=self.dtype, device=self.main_device)
        self.mask_neg = torch.full((), neg, dtype=self.dtype, device=self.main_device)

        self.talker_layers: list[_FusedDecoderLayer] | None = None
        self.cp_layers: list[_FusedDecoderLayer] | None = None
        self.cp_rotary: list[tuple[torch.Tensor, torch.Tensor]] = []
        # 模型被拆分到多个设备上时同样可以使用融合层 (逐层移动激活值), 但权重被卸载到 CPU / meta 设备时只能使用上游的实现
        weights_on_accelerator = all(d.type not in ("cpu", "meta") for d in devices) or (self.single_device and self.main_device.type == "cpu")
        if fuse_layers and weights_on_accelerator:
            with torch.no_grad():
                self.talker_layers = [_FusedDecoderLayer(layer) for layer in self.talker.model.layers]
                cp_model = self.code_predictor.model
                self.cp_layers = [_FusedDecoderLayer(layer) for layer in cp_model.layers[: cp_model.config.num_hidden_layers]]
                probe = torch.zeros(1, dtype=self.dtype, device=dev)
                for position_ids in self.cp_position_ids:
                    cos, sin = cp_model.rotary_emb(probe, position_ids)
                    self.cp_rotary.append((cos.unsqueeze(2), sin.unsqueeze(2)))

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------
    def release(self) -> None:
        """释放全部工作区与 CUDA Graph"""
        self.workspaces.clear()
        self.suppress_masks.clear()
        self.graph_pool = None
        self.capture_stream = None

    def _get_suppress_mask(
        self,
        suppress_tokens: tuple[int, ...],
    ) -> torch.Tensor:
        mask = self.suppress_masks.get(suppress_tokens)
        if mask is None:
            mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.main_device)
            if suppress_tokens:
                mask[torch.tensor(suppress_tokens, device=self.main_device)] = True
            self.suppress_masks[suppress_tokens] = mask
        return mask

    def _layer_device(
        self,
        layer: torch.nn.Module,
    ) -> torch.device:
        return layer.self_attn.k_proj.weight.device

    def _new_cache(
        self,
        layers: torch.nn.ModuleList,
        config: Any,
        batch_size: int,
        length: int,
    ) -> _StaticCache:
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        keys, values = [], []
        for layer in layers:
            dev = self._layer_device(layer)
            keys.append(torch.zeros(batch_size, num_kv_heads, length, head_dim, dtype=self.dtype, device=dev))
            values.append(torch.zeros(batch_size, num_kv_heads, length, head_dim, dtype=self.dtype, device=dev))
        return _StaticCache(keys, values)

    def _allocate_workspace(
        self,
        batch_size: int,
        kv_capacity: int,
        text_capacity: int,
    ) -> _Workspace:
        dev = self.main_device
        hidden_size = self.model.config.talker_config.hidden_size
        cp_layers = self.code_predictor.model.layers[: self.code_predictor.model.config.num_hidden_layers]
        return _Workspace(
            batch_size=batch_size,
            kv_capacity=kv_capacity,
            text_capacity=text_capacity,
            talker_cache=self._new_cache(self.talker.model.layers, self.talker.config, batch_size, kv_capacity),
            cp_cache=self._new_cache(cp_layers, self.code_predictor.config, batch_size, self.num_code_groups),
            trailing=torch.zeros(batch_size, text_capacity, hidden_size, dtype=self.dtype, device=dev),
            codes=torch.zeros(batch_size, kv_capacity, self.num_code_groups, dtype=torch.long, device=dev),
            hiddens=torch.zeros(batch_size, kv_capacity, hidden_size, dtype=self.dtype, device=dev),
            kv_arange=torch.arange(kv_capacity, device=dev),
            pad_len=torch.zeros(batch_size, dtype=torch.long, device=dev),
            step=torch.zeros(1, dtype=torch.long, device=dev),
            cur_pos=torch.zeros(1, dtype=torch.long, device=dev),
            last_token=torch.zeros(batch_size, dtype=torch.long, device=dev),
            past_hidden=torch.zeros(batch_size, 1, hidden_size, dtype=self.dtype, device=dev),
            seen=torch.zeros(batch_size, self.vocab_size, dtype=torch.bool, device=dev),
            unfinished=torch.ones(batch_size, dtype=torch.bool, device=dev),
        )

    def _get_workspace(
        self,
        batch_size: int,
        kv_needed: int,
        text_needed: int,
    ) -> _Workspace:
        """获取满足容量需求的工作区, 容量不足时重新分配 (同时丢弃该工作区的 CUDA Graph)

        Args:
            batch_size (int):
                批次大小
            kv_needed (int):
                需要的 KV Cache 长度
            text_needed (int):
                需要的文本缓冲区长度

        Returns:
            _Workspace: 工作区
        """
        ws = self.workspaces.get(batch_size)
        if ws is not None and ws.kv_capacity >= kv_needed and ws.text_capacity >= text_needed:
            self.workspaces.move_to_end(batch_size)
            return ws

        kv_capacity = _round_up(kv_needed, _KV_BUCKET)
        text_capacity = _round_up(text_needed, 64)
        if ws is not None:
            kv_capacity = max(kv_capacity, ws.kv_capacity)
            text_capacity = max(text_capacity, ws.text_capacity)
            del self.workspaces[batch_size]
            ws = None
        while len(self.workspaces) >= _MAX_WORKSPACES:
            self.workspaces.popitem(last=False)
        ws = self._allocate_workspace(batch_size, kv_capacity, text_capacity)
        self.workspaces[batch_size] = ws
        return ws

    def _grow_kv(
        self,
        ws: _Workspace,
        kv_needed: int,
    ) -> _Workspace:
        """在解码过程中扩容 KV Cache, 保留已有内容

        Args:
            ws (_Workspace):
                当前工作区
            kv_needed (int):
                需要的 KV Cache 长度

        Returns:
            _Workspace: 扩容后的工作区
        """
        new_capacity = _round_up(max(kv_needed, ws.kv_capacity * 2), _KV_BUCKET)
        new_ws = self._allocate_workspace(ws.batch_size, new_capacity, ws.text_capacity)
        old_len = ws.kv_capacity
        for src, dst in zip(ws.talker_cache.keys + ws.talker_cache.values, new_ws.talker_cache.keys + new_ws.talker_cache.values, strict=True):
            dst[:, :, :old_len].copy_(src)
        new_ws.codes[:, :old_len].copy_(ws.codes)
        new_ws.hiddens[:, :old_len].copy_(ws.hiddens)
        new_ws.trailing.copy_(ws.trailing)
        for src, dst in zip(ws.state_tensors() + [ws.pad_len], new_ws.state_tensors() + [new_ws.pad_len], strict=True):
            dst.copy_(src)
        self.workspaces[ws.batch_size] = new_ws
        return new_ws

    # ------------------------------------------------------------------
    # 模型前向
    # ------------------------------------------------------------------
    def _talker_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: _StaticCache,
    ) -> torch.Tensor:
        model = self.talker.model
        hidden = inputs_embeds
        if self.talker_layers is not None:
            # 纯文本输入时三个位置维度完全相同, 多模态旋转位置编码退化为普通旋转位置编码
            cos, sin = model.rotary_emb(hidden, position_ids)
            cos, sin = cos[0].unsqueeze(2), sin[0].unsqueeze(2)
            for layer in self.talker_layers:
                hidden = layer(hidden, cos, sin, cache, attention_mask, is_causal=False)
            return _rms_norm(hidden.to(model.norm.weight.device), model.norm.weight, model.norm.variance_epsilon).to(self.main_device)

        position_embeddings = model.rotary_emb(hidden, position_ids)
        for layer in model.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids[0],
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )[0]
        return model.norm(hidden).to(self.main_device)

    def _cp_forward(
        self,
        inputs_embeds: torch.Tensor,
        step: int,
        attention_mask: torch.Tensor | None,
        cache: _StaticCache,
    ) -> torch.Tensor:
        cp = self.code_predictor
        hidden = cp.small_to_mtp_projection(inputs_embeds.to(self.cp_device))
        if self.cp_layers is not None:
            cos, sin = self.cp_rotary[step]
            for layer in self.cp_layers:
                hidden = layer(hidden, cos, sin, cache, None, is_causal=step == 0)
            return _rms_norm(hidden.to(cp.model.norm.weight.device), cp.model.norm.weight, cp.model.norm.variance_epsilon)

        position_ids = self.cp_position_ids[step]
        position_embeddings = cp.model.rotary_emb(hidden, position_ids)
        for layer in cp.model.layers[: cp.model.config.num_hidden_layers]:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )[0]
        return cp.model.norm(hidden)

    def _predict_codes(
        self,
        ws: _Workspace,
        cp_inputs: torch.Tensor,
        cfg: SamplingConfig,
    ) -> torch.Tensor:
        """使用 Code Predictor 预测剩余码本 (等价于上游的 `code_predictor.generate()`)

        Args:
            ws (_Workspace):
                工作区
            cp_inputs (torch.Tensor):
                形状为 `[B, 2, D]` 的输入 (上一步 Talker 隐藏状态 + 第一码本的 Embedding)
            cfg (SamplingConfig):
                Code Predictor 的采样参数

        Returns:
            torch.Tensor: 形状为 `[B, num_code_groups - 1]` 的码本 ID
        """
        cp = self.code_predictor
        cache = ws.cp_cache
        codes = []
        hidden_in = cp_inputs
        mask = self.cp_prefill_mask
        for j in range(self.num_code_groups - 1):
            if j > 0:
                hidden_in = cp.model.codec_embedding[j - 1](codes[-1].unsqueeze(1).to(self.cp_device))
                mask = None
            cache.write_pos = self.cp_write_pos[j]
            cache.kv_len = j + 2
            hidden = self._cp_forward(hidden_in, j, mask, cache)
            logits = cp.lm_head[j](hidden[:, -1, :]).float()
            codes.append(_sample_tokens(logits, cfg).to(self.main_device))
        return torch.stack(codes, dim=1)

    def _process_talker_logits(
        self,
        logits: torch.Tensor,
        ws: _Workspace,
        cfg: SamplingConfig,
        block_eos: torch.Tensor | bool,
        eos_token_id: int,
        suppress_mask: torch.Tensor,
    ) -> torch.Tensor:
        """按 HuggingFace Logits Processor 的顺序处理 Talker 的 Logits 并采样"""
        scores = logits
        if cfg.repetition_penalty != 1.0:
            penalized = torch.where(scores < 0, scores * cfg.repetition_penalty, scores / cfg.repetition_penalty)
            scores = torch.where(ws.seen, penalized, scores)
        if isinstance(block_eos, torch.Tensor):
            eos_col = scores[:, eos_token_id : eos_token_id + 1]
            scores = torch.cat(
                [scores[:, :eos_token_id], torch.where(block_eos, -math.inf, eos_col), scores[:, eos_token_id + 1 :]],
                dim=1,
            )
        elif block_eos:
            scores = scores.clone()
            scores[:, eos_token_id] = -math.inf
        scores = scores.masked_fill(suppress_mask, -math.inf)
        return _sample_tokens(scores, cfg)

    def _decode_step(
        self,
        ws: _Workspace,
        kv_len: int,
        talker_cfg: SamplingConfig,
        sub_cfg: SamplingConfig,
        min_new_tokens: int,
        eos_token_id: int,
        suppress_mask: torch.Tensor,
    ) -> None:
        """单帧解码: Code Predictor 预测当前帧的剩余码本, 再由 Talker 预测下一帧的第一码本

        该函数的全部计算都只依赖工作区中的张量, 可以被 CUDA Graph 捕获后重复执行.
        """
        talker = self.talker
        last_id_hidden = talker.model.codec_embedding(ws.last_token.unsqueeze(1))
        ws.hiddens.index_copy_(1, ws.step, ws.past_hidden)

        # 1. Code Predictor: 预测当前帧的第 2 ~ 16 个码本
        sub_codes = self._predict_codes(ws, torch.cat((ws.past_hidden, last_id_hidden), dim=1), sub_cfg)
        frame = torch.cat((ws.last_token.unsqueeze(1), sub_codes), dim=1)
        ws.codes.index_copy_(1, ws.step, frame.unsqueeze(1))

        # 2. 构建 Talker 的输入: 全部码本 Embedding 之和 + 对应位置的文本 Embedding
        codec_hiddens = torch.cat(
            [last_id_hidden]
            + [self.code_predictor.model.codec_embedding[i](sub_codes[:, i : i + 1].to(self.cp_device)).to(self.main_device) for i in range(self.num_code_groups - 1)],
            dim=1,
        )
        inputs_embeds = codec_hiddens.sum(1, keepdim=True)
        text_index = ws.step.clamp(max=ws.text_capacity - 1)
        inputs_embeds = inputs_embeds + ws.trailing.index_select(1, text_index)

        # 3. Talker 前向
        cache = ws.talker_cache
        cache.write_pos = ws.cur_pos
        cache.kv_len = kv_len
        kv_index = ws.kv_arange[:kv_len].unsqueeze(0)
        valid = (kv_index >= ws.pad_len.unsqueeze(1)) & (kv_index <= ws.cur_pos)
        attention_mask = torch.where(valid, self.mask_zero, self.mask_neg)[:, None, None, :]
        position_ids = (ws.cur_pos - ws.pad_len).view(1, -1, 1).expand(3, -1, 1)
        hidden = self._talker_forward(inputs_embeds, position_ids, attention_mask, cache)
        ws.past_hidden.copy_(hidden[:, -1:, :])
        logits = talker.codec_head(hidden[:, -1, :]).float().to(self.main_device)

        # 4. 采样下一帧的第一码本并更新状态
        block_eos = (ws.step + 1) < min_new_tokens
        next_token = self._process_talker_logits(logits, ws, talker_cfg, block_eos, eos_token_id, suppress_mask)
        next_token = torch.where(ws.unfinished, next_token, eos_token_id)
        ws.seen.scatter_(1, next_token.unsqueeze(1), True)
        ws.unfinished.logical_and_(next_token != eos_token_id)
        ws.last_token.copy_(next_token)
        ws.step.add_(1)
        ws.cur_pos.add_(1)

    def _get_runner(
        self,
        ws: _Workspace,
        kv_len: int,
        step_args: tuple[Any, ...],
        graph_key: tuple[Any, ...],
        stats: GenerationStats,
    ) -> Callable[[], None]:
        """获取单帧解码的执行函数, 在 CUDA 设备上返回 CUDA Graph 的重放函数"""
        if not self.use_cuda_graph:
            return lambda: self._decode_step(ws, kv_len, *step_args)

        key = (kv_len, *graph_key)
        graph = ws.graphs.get(key)
        if graph is not None:
            ws.graphs.move_to_end(key)
            return graph.replay

        start = _now()
        if self.graph_pool is None:
            self.graph_pool = torch.cuda.graph_pool_handle()
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=self.main_device)
        # 预热一次 (初始化 cuBLAS 等惰性资源), 之后恢复状态与随机数状态 (保证固定随机种子时结果可复现), 再进行捕获 (捕获过程不会实际执行计算)
        snapshot = [t.clone() for t in ws.state_tensors()]
        rng_state = torch.cuda.get_rng_state(self.main_device)
        side_stream = torch.cuda.Stream(device=self.main_device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            self._decode_step(ws, kv_len, *step_args)
        torch.cuda.current_stream().wait_stream(side_stream)
        for dst, src in zip(ws.state_tensors(), snapshot, strict=True):
            dst.copy_(src)
        torch.cuda.set_rng_state(rng_state, self.main_device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool, stream=self.capture_stream, capture_error_mode="thread_local"):
            self._decode_step(ws, kv_len, *step_args)
        while len(ws.graphs) >= _MAX_GRAPHS_PER_WORKSPACE:
            ws.graphs.popitem(last=False)
        ws.graphs[key] = graph
        stats.graph_captures += 1
        stats.capture_seconds += _now() - start
        return graph.replay

    # ------------------------------------------------------------------
    # 生成入口
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        max_new_tokens: int,
        min_new_tokens: int,
        eos_token_id: int,
        suppress_tokens: list[int],
        talker_sampling: SamplingConfig,
        subtalker_sampling: SamplingConfig,
        should_stop: Callable[[], bool] | None = None,
        on_frames: Callable[[torch.Tensor, bool], None] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """生成音频码本

        Args:
            inputs_embeds (torch.Tensor):
                左侧填充后的 Talker 输入, 形状为 `[B, L, D]`
            attention_mask (torch.Tensor):
                Talker 输入的注意力掩码, 形状为 `[B, L]`
            trailing_text_hidden (torch.Tensor):
                解码阶段逐步加入的文本 Embedding, 形状为 `[B, T, D]`
            tts_pad_embed (torch.Tensor):
                文本结束后使用的填充 Embedding, 形状为 `[1, 1, D]`
            max_new_tokens (int):
                最大生成帧数
            min_new_tokens (int):
                最小生成帧数
            eos_token_id (int):
                结束符 ID
            suppress_tokens (list[int]):
                禁止生成的 Token ID
            talker_sampling (SamplingConfig):
                Talker 采样参数
            subtalker_sampling (SamplingConfig):
                Code Predictor 采样参数
            should_stop (Callable[[], bool] | None):
                返回 True 时中断生成并抛出 `GenerationStoppedError`
            on_frames (Callable[[torch.Tensor, bool], None] | None):
                流式输出回调, 每解码一帧调用一次, 参数为目前已生成的码本 (`[B, T, num_code_groups]`, 包含结束符之后的帧)
                以及是否为最后一次调用. 回调函数只在需要读取数据时才会产生 CPU-GPU 同步

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: 每条样本的码本 (`[T, num_code_groups]`) 与 Talker 隐藏状态 (`[T, D]`)

        Raises:
            GenerationStoppedError: `should_stop` 返回 True 时
        """
        stats = GenerationStats(used_cuda_graph=self.use_cuda_graph)
        self.last_stats = stats
        restore_attn = self._use_static_cache_attention()
        try:
            device_ctx = torch.cuda.device(self.main_device) if self.main_device.type == "cuda" else contextlib.nullcontext()
            with device_ctx:
                return self._generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    eos_token_id=eos_token_id,
                    suppress_tokens=suppress_tokens,
                    talker_sampling=talker_sampling,
                    subtalker_sampling=subtalker_sampling,
                    should_stop=should_stop,
                    on_frames=on_frames,
                    stats=stats,
                )
        finally:
            restore_attn()

    def _use_static_cache_attention(self) -> Callable[[], None]:
        """静态 KV Cache 需要使用 4D 注意力掩码, Flash Attention 2 不支持, 因此在解码时临时切换为 SDPA

        Returns:
            Callable[[], None]: 恢复原始设置的函数
        """
        configs = self.attn_configs
        original = [c._attn_implementation for c in configs]  # pylint: disable=protected-access
        for c, impl in zip(configs, original, strict=True):
            if impl not in ("eager", "sdpa"):
                c._attn_implementation = "sdpa"  # pylint: disable=protected-access

        def restore() -> None:
            for c, impl in zip(configs, original, strict=True):
                c._attn_implementation = impl  # pylint: disable=protected-access

        return restore

    def _generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        max_new_tokens: int,
        min_new_tokens: int,
        eos_token_id: int,
        suppress_tokens: list[int],
        talker_sampling: SamplingConfig,
        subtalker_sampling: SamplingConfig,
        should_stop: Callable[[], bool] | None,
        on_frames: Callable[[torch.Tensor, bool], None] | None,
        stats: GenerationStats,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        dev = self.main_device
        batch_size, prompt_len, _ = inputs_embeds.shape
        max_steps = max(int(max_new_tokens) - 1, 0)
        text_len = trailing_text_hidden.shape[1]
        stats.batch_size = batch_size
        stats.prompt_len = prompt_len

        ws = self._get_workspace(batch_size, prompt_len + min(max_steps, _INITIAL_FRAMES) + 2, text_len + 1)
        suppress_mask = self._get_suppress_mask(tuple(suppress_tokens))

        # ---------------- Prefill ----------------
        start = _now()
        attention_mask = attention_mask.to(dev)
        pad_len = prompt_len - attention_mask.sum(dim=1)
        ws.pad_len.copy_(pad_len)
        ws.trailing[:, :text_len].copy_(trailing_text_hidden)
        ws.trailing[:, text_len:].copy_(tts_pad_embed.reshape(1, 1, -1).expand(batch_size, ws.text_capacity - text_len, -1))

        positions = attention_mask.long().cumsum(-1) - 1
        positions = positions.masked_fill(attention_mask == 0, 1)
        position_ids = positions.unsqueeze(0).expand(3, -1, -1)
        q_index = torch.arange(prompt_len, device=dev)
        causal = q_index.view(1, -1, 1) >= q_index.view(1, 1, -1)
        valid = causal & attention_mask.bool().unsqueeze(1)
        prefill_mask = torch.where(valid, self.mask_zero, self.mask_neg).unsqueeze(1)

        cache = ws.talker_cache
        cache.write_pos = q_index
        cache.kv_len = prompt_len
        hidden = self._talker_forward(inputs_embeds.to(dev), position_ids, prefill_mask, cache)
        logits = self.talker.codec_head(hidden[:, -1, :]).float().to(dev)

        ws.seen.zero_()
        first_token = self._process_talker_logits(logits, ws, talker_sampling, min_new_tokens > 0, eos_token_id, suppress_mask)
        ws.past_hidden.copy_(hidden[:, -1:, :])
        ws.last_token.copy_(first_token)
        ws.seen.scatter_(1, first_token.unsqueeze(1), True)
        ws.unfinished.copy_(first_token != eos_token_id)
        ws.step.zero_()
        ws.cur_pos.fill_(prompt_len)
        stats.prefill_seconds = _now() - start

        # ---------------- Decode ----------------
        start = _now()
        step_args = (talker_sampling, subtalker_sampling, min_new_tokens, eos_token_id, suppress_mask)
        graph_key = (talker_sampling, subtalker_sampling, min_new_tokens, eos_token_id, tuple(suppress_tokens))
        is_cuda = dev.type == "cuda"
        pending: tuple[Any, torch.Tensor] | None = None
        flag_buffers = [torch.ones(1, dtype=torch.bool, pin_memory=is_cuda) for _ in range(2)]
        steps = 0
        for s in range(max_steps):
            if prompt_len + s + 1 > ws.kv_capacity:
                ws = self._grow_kv(ws, prompt_len + s + 1)
            kv_len = min(_round_up(prompt_len + s + 1, _KV_BUCKET), ws.kv_capacity)
            self._get_runner(ws, kv_len, step_args, graph_key, stats)()
            steps += 1
            if on_frames is not None:
                on_frames(ws.codes[:, :steps], False)

            # 延迟一步检查是否全部结束: 先读取上一步的结果, 当前步的结果在下一轮读取
            flag = flag_buffers[s % 2]
            flag.copy_(ws.unfinished.any().view(1), non_blocking=is_cuda)
            event = None
            if is_cuda:
                event = torch.cuda.Event()
                event.record()
            if pending is not None:
                prev_event, prev_flag = pending
                if prev_event is not None:
                    prev_event.synchronize()
                if not bool(prev_flag.item()):
                    break
            pending = (event, flag)
            if should_stop is not None and should_stop():
                stats.interrupted = True
                stats.steps = steps
                stats.decode_seconds = _now() - start
                raise GenerationStoppedError("generation stopped by should_stop()")
        if on_frames is not None:
            on_frames(ws.codes[:, :steps], True)
        if is_cuda:
            torch.cuda.current_stream().synchronize()
        stats.steps = steps
        stats.decode_seconds = _now() - start

        # ---------------- 收集结果 ----------------
        codes = ws.codes[:, :steps].clone()
        hiddens = ws.hiddens[:, :steps].clone()
        is_eos = codes[:, :, 0] == eos_token_id
        lengths = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1), steps).tolist()
        codes_list = [codes[i, :length] for i, length in enumerate(lengths)]
        hiddens_list = [hiddens[i, :length] for i, length in enumerate(lengths)]
        return codes_list, hiddens_list


def _round_up(
    value: int,
    multiple: int,
) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _now() -> float:
    return time.perf_counter()
