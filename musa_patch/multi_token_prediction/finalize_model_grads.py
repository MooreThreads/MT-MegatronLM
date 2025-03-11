from typing import List, Optional, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model, get_model_config

from megatron.core.distributed.finalize_model_grads import (
    _unshard_if_dtensor, 
    _reshard_if_dtensor, 
    _allreduce_conditional_embedding_grads,
    _allreduce_layernorm_grads,
    _allreduce_embedding_grads
)
def _allreduce_mtp_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync.
    """

    if (
        parallel_state.is_rank_in_mtp_embedding_group(ignore_virtual=True)
        and torch.distributed.get_world_size(parallel_state.get_mtp_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.embedding.word_embeddings.weight
            grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
            orig_grad = getattr(weight, grad_attr)
            grad = _unshard_if_dtensor(orig_grad)
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_mtp_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    _allreduce_mtp_word_embedding_grads(model, config)
    # _allreduce_mtp_position_embedding_grads(model, config)
    
def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT).
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    ### Actual MUSA patch modification begins ###
    # All-reduce embedding grads (for multi token prediction).
    if config.timers is not None:
        config.timers('mtp-embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    
    if config.use_multi_token_prediction:
        _allreduce_mtp_embedding_grads(model, config)
    
    if config.timers is not None:
        config.timers('mtp-embedding-grads-all-reduce').stop()
    ###  Actual MUSA patch modification ends  ###

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        if not isinstance(last_rank, list):
            assert not isinstance(last_rank, list)
            last_rank = [last_rank]
            assert not isinstance(pp_group, list)
            pp_group = [pp_group]

        # need to do a broadcast for every pp group, even though num_tokens should be the same.
        num_tokens_list = []
        for lr, group in zip(last_rank, pp_group):
            torch.distributed.broadcast(num_tokens, src=lr, group=group)
            num_tokens_list.append(torch.clone(num_tokens))
        assert all(x.item() == num_tokens_list[0] for x in num_tokens_list)

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)


import sys
for k in sys.modules:
    if k.startswith('megatron'):
        for target in ['finalize_model_grads']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, finalize_model_grads)