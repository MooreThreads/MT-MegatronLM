# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock

import logging

import torch
import torch_musa
from torch import Tensor, nn

from megatron.core import parallel_state

logger = logging.getLogger(__name__)

try:
    from apex.transformer.functional import (
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )

    HAVE_APPLY_ROPE_FUSION = True
except ImportError:
    HAVE_APPLY_ROPE_FUSION = False

def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)
    
def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_thd(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb_bshd(x.unsqueeze(1), freqs[: x.size(0)])
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)
    
def apply_rotary_pos_emb(
    t: Tensor, freqs: Tensor, config: TransformerConfig, cu_seqlens: Optional[Tensor] = None, mscale: float = 1.0,
):

    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    # assert cu_seqlens is None, "Only support cu_seqlens is None for now!"
    if config.apply_rope_fusion and not HAVE_APPLY_ROPE_FUSION:
        # setting apply_rope_fusion in config to False so that subsequent queries to this config also return False
        config.apply_rope_fusion = False
        if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
            logger.warning(
                "Setting apply_rope_fusion to false because its implementation"
                " is not included in Apex. Try upgrading to the latest version"
            )
            apply_rotary_pos_emb.printed_fused_warning = True
    if config.apply_rope_fusion:
        if cu_seqlens is None:
            return torch.rope(t, freqs.squeeze(1).squeeze(1), rotary_interleaved=False, batch_first=False)
            # return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        else:
            return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if cu_seqlens is None:
            return apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)
        else:
            return apply_rotary_pos_emb_thd(
                t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved
            )
            
# import megatron.core.models.common.embeddings.rotary_pos_embedding
# megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb = apply_rotary_pos_emb
# import megatron.core.transformer.attention
# megatron.core.transformer.attention.apply_rotary_pos_emb = apply_rotary_pos_emb

import sys
for k in sys.modules:
    if k.startswith('megatron.core'):
        for target in ['apply_rotary_pos_emb']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, apply_rotary_pos_emb)
