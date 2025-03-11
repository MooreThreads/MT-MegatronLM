from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state

def get_num_layers_to_build(config: TransformerConfig) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    if config.first_pipeline_num_layers is not None or config.last_pipeline_num_layers is not None:
        assert (
            parallel_state.get_virtual_pipeline_model_parallel_world_size() is None
        ), "Uneven number of layer not compatible with interleaved pipeline schedule"

        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = parallel_state.get_pipeline_model_parallel_world_size()

        if config.first_pipeline_num_layers is not None:
            layers_to_distribute -= config.first_pipeline_num_layers
            pipeline_stages_left -= 1
            if parallel_state.is_pipeline_first_stage():
                return config.first_pipeline_num_layers

        if config.last_pipeline_num_layers is not None:
            layers_to_distribute -= config.last_pipeline_num_layers
            pipeline_stages_left -= 1
            if parallel_state.is_pipeline_last_stage():
                return config.last_pipeline_num_layers

        assert (
            layers_to_distribute % pipeline_stages_left == 0
        ), "With uneven pipelineing the left over layers must be divisible by left over stages"
        num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
    else:
        pipeline_ranks = config.pipeline_model_parallel_size
        num_layers_per_pipeline_rank = config.num_layers // pipeline_ranks

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_rank

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.

        num_layers_to_build = num_layers_per_pipeline_rank
    ### Actual MUSA patch modification begins ###
    if config.use_multi_token_prediction and parallel_state.is_pipeline_last_stage():
        num_layers_to_build += config.multi_token_prediction_depth
    ### Actual MUSA patch modification ends ###
    return num_layers_to_build

import sys
for k in sys.modules:
    if k.startswith('megatron'):
        for target in ['get_num_layers_to_build']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, get_num_layers_to_build)