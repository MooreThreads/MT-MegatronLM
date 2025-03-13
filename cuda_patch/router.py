# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.moe.router import TopKRouter

from .moe_utils import (
    sequence_load_balancing_loss_func,
    topk_softmax_with_capacity,
)


def init(self, config):
    """Initialize the zero token dropping router.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    super(TopKRouter, self).__init__(config=config)
    self.topk = self.config.moe_router_topk
    self.routing_type = self.config.moe_router_load_balancing_type
    self.input_jitter = None
    self.e_score_correction_bias = None
    self.moe_noaux_gamma = self.config.moe_noaux_gamma
    if self.moe_noaux_gamma:
        self.tokens_per_expert = torch.nn.Parameter(
             torch.empty((self.config.num_moe_experts)), requires_grad=False
        )
        self.e_score_correction_bias = torch.nn.Parameter(
             torch.empty((self.config.num_moe_experts)), requires_grad=False
        )
        if config.perform_initialization:
            # initialize bias to zero.
            with torch.no_grad():
                self.e_score_correction_bias.zero_()
                self.tokens_per_expert.zero_()

def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
    """Apply loss-based load balancing to the logits tensor."""

    probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
        logits,
        self.topk,
        capacity_factor=self.config.moe_expert_capacity_factor,
        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        drop_policy=self.config.moe_token_drop_policy,
        use_pre_softmax=self.config.moe_router_pre_softmax,
        moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
        moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
        use_sigmoid=self.config.moe_router_use_sigmoid,
        norm_topk_prob=self.config.moe_router_norm_topk_prob,
        deterministic_mode=self.config.deterministic_mode,
        device_level_capacity=self.config.moe_device_level_capacity,
        num_node_group=self.config.moe_router_num_node_group,
        e_score_correction_bias=self.e_score_correction_bias,
    )

    if self.training:
        if self.config.moe_router_use_sigmoid == "sigmoid":
            scores = torch.sigmoid(logits)
        else: 
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        aux_loss_func = partial(
            sequence_load_balancing_loss_func,
            probs=scores,
            routing_map=routing_map,
            tokens_per_expert=tokens_per_expert,
            batch_size=bsz,
            seq_length=seq_length,
            topk=self.topk,
            moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
            moe_device_level_aux_loss_coeff=self.config.moe_device_level_aux_loss_coeff,
            moe_comm_aux_loss_coeff=self.config.moe_comm_aux_loss_coeff,
            moe_complementary_seq_aux_loss=self.config.moe_complementary_seq_aux_loss,
        )
        probs = self.apply_load_balancing_loss(
            activation=probs, load_balancing_loss_func=aux_loss_func
        )

        if self.moe_noaux_gamma and torch.is_grad_enabled():
            with torch.no_grad():
                mean_token = logits.shape[0] * self.topk / logits.shape[1]
                self.tokens_per_expert += (tokens_per_expert /mean_token).to(self.tokens_per_expert.device)
    return probs, routing_map


import megatron.core.transformer.moe.router
megatron.core.transformer.moe.router.TopKRouter.seq_aux_loss_load_balancing = seq_aux_loss_load_balancing
megatron.core.transformer.moe.router.TopKRouter.__init__ = init

