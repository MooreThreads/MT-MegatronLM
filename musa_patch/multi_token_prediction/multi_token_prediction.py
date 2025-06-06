# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer, TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import is_te_min_version, make_viewless_tensor

try:
    from megatron.core.extensions.transformer_engine import (
        TEDelayedScaling,
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None

    try:
        import apex  # pylint: disable=unused-import

        LayerNormImpl = FusedLayerNorm

    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        LayerNormImpl = WrappedTorchNorm
from .transformer_block import (
    get_num_layers_to_build, 
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, _get_block_submodules

class MultiTokenPredictionBlock(MegatronModule):
    """Multi Token Prediction class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(config=config)

        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        # Dictionary to store CUDA graphs. Number of items in the dictionary = len(self.layers).
        # Item `i` in the dictionary is a list of `N` CUDA graphs for layer 'i' where N is the
        # number of microbatches. Multiple CUDA graphs per layer is required to support
        # pipelining which requires running FWD graph of multiple microbatches before BWD graph.
        # To enable CUDA graph, this dictionary should be populated in the model training script
        # with the graphs returned by make_graphed_callables API before the first trainng step.
        self.cuda_graphs = {}
        self.current_microbatch = -1

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        if get_cpu_offload_context is not None:
            (self.offload_context, self.group_prefetch_offload_commit_async) = (
                get_cpu_offload_context(
                    self.config.cpu_offloading,
                    self.config.cpu_offloading_num_layers,
                    self.config.num_layers,
                    self.config.cpu_offloading_activations,
                    self.config.cpu_offloading_weights,
                )
            )
            self.config._cpu_offloading_context = (
                self.offload_context if self.config.cpu_offloading else None
            )
        else:
            assert (
                self.config.cpu_offloading is False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)
        self.tp_only_amax_red = config.tp_only_amax_red
        self.multi_token_prediction_depth = config.multi_token_prediction_depth
        
    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number)
        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        self.logit_layernorms = torch.nn.ModuleList(
            [
                build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
                for i in enumerate(self.submodules.layer_specs)
            ]
        )
        
        self.embedding_layernorms = torch.nn.ModuleList(
            [
                build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
                for i in enumerate(self.submodules.layer_specs)
            ]
        )
        
        self.linear_projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(2*self.config.hidden_size, self.config.hidden_size) for i in enumerate(self.submodules.layer_specs)
            ]
        )
        # @TODO: add back standalone_embedding_stage (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        # if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
        #     self.final_layernorm = build_module(
        #         self.submodules.layer_norm,
        #         config=self.config,
        #         hidden_size=self.config.hidden_size,
        #         eps=self.config.layernorm_epsilon,
        #     )
        # else:
        #     self.final_layernorm = None  # Either this or nn.Identity

    def _get_layer(self, layer_number: int):
        return self.logit_layernorms[layer_number], self.embedding_layernorms[layer_number], self.linear_projections[layer_number], self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        embedding_input: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, embedding_input, attention_mask, context, context_mask, rotary_pos_emb
            ):
                for index in range(start, end):
                    logit_layernorm, embedding_layernorm, linear_projection, layer = self._get_layer(index)
                    hidden_states = torch.nn.functional.pad(hidden_states[:-1, ...], (0, 0, 0, 0, 1, 0), value=0)
                    hidden_states = logit_layernorm(hidden_states)
                    embedding_input = embedding_layernorm(embedding_input)
                    proj_input = torch.concat([hidden_states, embedding_input], -1)
                    hidden_states = linear_projection(proj_input)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    embedding_input,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    embedding_input,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.multi_token_prediction_depth:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )
                self.mtp_logit_list.append(hidden_states)
                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.multi_token_prediction_depth):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, embedding_input, attention_mask, context, context_mask, rotary_pos_emb
                    )
                self.mtp_logit_list.append(hidden_states)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def get_cuda_graph_optional_args(
        self,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        inference_params: InferenceParams,
        packed_seq_params: PackedSeqParams,
    ):
        """Get optional tensor arguments for CUDA graph."""

        optional_inputs = {}
        optional_inputs['is_first_microbatch'] = self.current_microbatch == 0
        try:
            import transformer_engine.pytorch as te  # pylint: disable=unused-import

            if is_te_min_version("1.10.0", check_equality=False):
                assert not any(
                    [attention_mask, context, context_mask, rotary_pos_emb]
                ), "Keyword Arguments not supported with CUDA graph."
            else:
                optional_inputs['attention_mask'] = attention_mask
                optional_inputs['context'] = context
                optional_inputs['context_mask'] = context_mask
                optional_inputs['rotary_pos_emb'] = rotary_pos_emb
        except ImportError:
            raise RuntimeError("CUDAGraph requires TransformerEngine, but not installed")
        return optional_inputs

    def forward(
        self,
        hidden_states: Tensor,
        embedding_input: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params (InferenceParams, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        self.mtp_logit_list = []
        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Update the inference parameters with the current batch size in case it is variable
        if inference_params and not self.training:
            inference_params.current_batch_size = hidden_states.size(1)

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()
            
        
        with rng_context, fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training: # 0:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    embedding_input=embedding_input,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                for l_no, (logit_layernorm, embedding_layernorm, linear_projection, layer) in \
                        enumerate(zip(self.logit_layernorms, self.embedding_layernorms,self.linear_projections, self.layers)):
                    with self.offload_context:
                        layer.use_cudagraph = True
                        if (len(self.cuda_graphs) == 0) or (not self.training): # 1: 
                            hidden_states = torch.nn.functional.pad(hidden_states[:-1, ...], (0, 0, 0, 0, 1, 0), value=0)
                            hidden_states = logit_layernorm(hidden_states)
                            embedding_input = embedding_layernorm(hidden_states)
                            proj_input = torch.concat([hidden_states, embedding_input], -1)
                            hidden_states = linear_projection(proj_input)
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin,
                                attention_bias=attention_bias,
                                inference_params=inference_params,
                                packed_seq_params=packed_seq_params,
                                sequence_len_offset=sequence_len_offset,
                            )
                            self.mtp_logit_list = [].append(hidden_states)
                        else:
                            # CUDA graph replay for layer `l_no` and microbatch
                            # `self.current_microbatch`. TransformerEngine versions>=1.10
                            # allow keyword arguments with CUDA graph. However, CUDA graph
                            # acccepts only Tensor inputs and Tensor outputs. Hence,
                            # `inference_params` and `packed_seq_params` are excluded from
                            # input list while output is limited to `hidden_states`.
                            cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                            assert not any(
                                [inference_params, packed_seq_params]
                            ), "CUDA graph accepts only Tensor inputs."
                            optional_inputs = self.get_cuda_graph_optional_args(
                                attention_mask,
                                context,
                                context_mask,
                                rotary_pos_emb,
                                attention_bias,
                                inference_params,
                                packed_seq_params,
                            )
                            hidden_states = self.cuda_graphs[l_no][cg_index](
                                hidden_states, **optional_inputs
                            )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # # Final layer norm.
        # if self.final_layernorm is not None:
        #     hidden_states = self.final_layernorm(hidden_states)
        #     # TENorm produces a "viewed" tensor. This will result in schedule.py's
        #     # deallocate_output_tensor() throwing an error, so a viewless tensor is
        #     # created to prevent this.
        #     hidden_states = make_viewless_tensor(
        #         inp=hidden_states, requires_grad=True, keep_graph=True
        #     )

        return self.mtp_logit_list

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if isinstance(self.config.moe_layer_freq, int):
            if self.config.moe_layer_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.moe_layer_freq, list):
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = TransformerLayer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict
