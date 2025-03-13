# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numbers

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from megatron.core.transformer import TransformerConfig



class FusedLayerNorm(torch.nn.Module):

    """Layer Norm, fused into a single CUDA kernel.

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      persist_layer_norm (bool): Use persistent fused layer norm kernel.
      This kernel supports only a set of hidden sizes. Please
      check persist_ln_hidden_sizes if your hidden size is supported.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      config (TransformerConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        super().__init__()
        print("use FusedLayerNorm")

        self.config = config

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma

        if self.config.normalization == "LayerNorm":
            self.norm_impl = torch.layer_norm
        elif self.config.normalization == "RMSNorm":
            def naive_rms_norm(hidden_states, hidden_size, weight, eps):        
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + eps)
                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                hidden_states = weight * hidden_states
                return hidden_states 
            self.norm_impl = naive_rms_norm
        else:
            raise ValueError(f'({self.config.normalization}) is not supported in FusedLayerNorm')

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        # self.hidden_size = torch.Size(hidden_size)
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*hidden_size))
        self.bias = Parameter(torch.Tensor(*hidden_size)) if self.config.normalization == "LayerNorm" else None
        self.reset_parameters()
        self.sequence_parallel = self.config.sequence_parallel


        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        if self.config.normalization == "LayerNorm":
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):

        if self.zero_centered_gamma:
            init.zeros_(self.weight)
            if self.config.normalization == "LayerNorm":
                init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            if self.config.normalization == "LayerNorm":
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        if self.config.normalization == "LayerNorm":
            output = self.norm_impl(input, self.hidden_size, weight, self.bias, self.eps)
        else:
            output = self.norm_impl(input, self.hidden_size, weight, self.eps)

        return output

import megatron.core.fusions.fused_layer_norm
megatron.core.fusions.fused_layer_norm.FusedLayerNorm = FusedLayerNorm
