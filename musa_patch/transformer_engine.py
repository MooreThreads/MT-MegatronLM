# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import io
import os
import pickle
import warnings
from typing import Callable

import torch
import transformer_engine as te
from packaging.version import Version as PkgVersion
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_te_version, is_te_min_version


class TEDotProductAttention(te.musa.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        k_channels: int = None,
        v_channels: int = None,
        cp_comm_type: str = "p2p",
    ):
        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = 'sbhd'

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv('NVTE_APPLY_QK_LAYER_SCALING', '0'))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs = {}
        if is_te_min_version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{get_te_version()} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if is_te_min_version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if is_te_min_version("0.12.0", check_equality=False):
            self.te_forward_mask_type = True

        # This check is important as CP config can be disabled while having a valid CP group
        # Example - Disabling CP for encoder while a valid CP group exists for decoder
        if self.config.context_parallel_size > 1:
            assert is_te_min_version(
                "1.0.0"
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)
            extra_kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(
                check_initialized=False
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
            if is_te_min_version("1.10.0"):
                if cp_comm_type is None:
                    extra_kwargs["cp_comm_type"] = "p2p"
                elif cp_comm_type == "a2a+p2p":
                    assert is_te_min_version("1.12.0"), (
                        f"Transformer-Engine v{get_te_version()} must be >= 1.12.0 to support"
                        "hierarchical cp commucation."
                    )
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(
                        check_initialized=False
                    )
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        if self.config.deterministic_mode:
            if int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")) != 0:
                raise RuntimeError(
                    "deterministic_mode is on and we are using DotProductAttention from "
                    "Transformer Engine, but NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0. "
                    f"Currently set to: {os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'not set')}."
                )

        if config.window_size is not None:
            # Check version
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "sliding window attention."
            )
            extra_kwargs['window_size'] = config.window_size

        # if is_te_min_version("1.10.0"):
            # TE 1.10.0 introduces the ability to set the different k and v channels
        kv_channels = (
            (k_channels, v_channels)
            if k_channels is not None and v_channels is not None
            else self.config.kv_channels
        )
        extra_kwargs['softmax_scale'] = softmax_scale
        # else:
        #     kv_channels = self.config.kv_channels

        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )
        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            layer_number=layer_number,
            **extra_kwargs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set
        # after init
        if self.config.apply_rope_fusion and is_te_min_version("0.13.0", check_equality=False):
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        # WAR for peak memory usage.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2388
        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[0] == 1 and value.stride() != key.stride():
                value = value.as_strided(value.shape, key.stride())

        attention_bias_kwargs = {}
        if attention_bias is not None:
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "`attention_bias`."
            )
            attention_bias_kwargs = dict(
                core_attention_bias_type='post_scale_bias', core_attention_bias=attention_bias
            )

        if self.te_forward_mask_type:
            if qkv_format == 'thd' and is_te_min_version("1.7.0"):
                # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
                # so the only acceptable mask types are `padding_causal` and `padding`. These do not
                # necessarily indicate there are padded tokens in the sequence.
                if attn_mask_type == AttnMaskType.causal:
                    attn_mask_type = AttnMaskType.padding_causal
                elif attn_mask_type == AttnMaskType.no_mask:
                    attn_mask_type = AttnMaskType.padding
            core_attn_out = super().forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type.name,
                **attention_bias_kwargs,
                **packed_seq_kwargs,
            )
        else:
            core_attn_out = super().forward(
                query, key, value, attention_mask, **attention_bias_kwargs, **packed_seq_kwargs
            )

        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            return core_attn_out.transpose(0, 1)
        else:
            return core_attn_out

import sys
for k in sys.modules:
    if k.startswith('megatron'):
        print(f'k is {k}')
        for target in ['TEDotProductAttention']:
            if getattr(sys.modules[k], target, None):
                print(f'target is {target}')
                setattr(sys.modules[k], target, TEDotProductAttention)
