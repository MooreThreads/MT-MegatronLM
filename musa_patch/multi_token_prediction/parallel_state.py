import os
import warnings
from datetime import timedelta
from functools import partial
from itertools import cycle
from typing import Callable, List, Optional

import torch

from megatron.core.parallel_state import (
    default_embedding_ranks,
    default_position_embedding_ranks,
    RankGenerator,
    get_nccl_options,
    get_data_parallel_group,
    create_hierarchical_parallel_groups,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    _set_global_memory_buffer
)

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra-, pipeline, and expert) that the current rank belongs to.
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None

### Expert-related parallel states
# Naming convention:
# _EXPERT prefix in group name means it's used for expert layer in MoE models.
# _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
# _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# Expert tensor, model, pipeline combined parallel group
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None
# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
# Parallel state values changed on the fly
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK = None
### End of expert related parallel states

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

_PIPELINE_MODEL_PARALLEL_DECODER_START = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_RANK = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each model parallel group to ease calculation of
# the first local rank in the model parallel group
_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
# Hierarchical context parallel groups
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = []

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# Partial Data parallel group information with context parallel combined.
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None

# combined parallel group of TP and CP
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# MOE logging
_MOE_LAYER_WISE_LOGGING_TRACKER = {}

# MTP Embedding group.
_MTP_EMBEDDING_GROUP = None
# MTP Position embedding group.
_MTP_POSITION_EMBEDDING_GROUP = None

# A list of ranks that have a copy of the mtp embedding.
_MTP_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the mtp position embedding.
_MTP_POSITION_EMBEDDING_GLOBAL_RANKS = None

def get_mtp_embedding_ranks(pp_ranks):
    """Return the default ranks that constitute the stages on which the word embeddings live.
    For most models, these are the first and last pipeline stages.

    We also support the deprecated split rank argument for backwards compatibility."""
    return [pp_ranks[0], pp_ranks[-1]]
    

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: int = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
) -> None:
    # pylint: disable=line-too-long
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            DEPRECATED. For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        num_distributed_optimizer_instances (int, default = 1):
            The number of distributed optimizer replicas across the data-
            parallel domain.

        expert_tensor_parallel_size (int, default = tp_size):
            The number of GPUs to split individual tensors of expert.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

        encoder_tensor_model_parallel_size (int, default = 0):
            The number of GPUs to split individual tensors across in the encoder. If 0,
            then we use the default, decoder's tensor model parallel size.

        encoder_pipeline_model_parallel_size (int, default = 0):
            The number of tensor parallel GPU groups to allocate to the encoder. As an example,
            if pipeline_model_parallel_size is 4 and encoder_pipeline_model_parallel_size is 2,
            then the encoder will use the first two pipeline stages for its layers, and the total
            amount of pipelineing is 6.

        get_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group and returns
            those ranks that should have embeddings.

        get_position_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group, and returns
            those ranks that should have position embeddings.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if encoder_tensor_model_parallel_size == 0 and encoder_pipeline_model_parallel_size > 0:
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    if get_embedding_ranks is None:
        get_embedding_ranks = partial(
            default_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = partial(
            default_position_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if encoder_pipeline_model_parallel_size > 0:
        global _PIPELINE_MODEL_PARALLEL_DECODER_START
        _PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if encoder_tensor_model_parallel_size > 0:
        assert (
            encoder_tensor_model_parallel_size <= tensor_model_parallel_size
        ), "We do not support encoders with more TP than the decoder."

    encoder_model_size = (
        encoder_tensor_model_parallel_size
        * encoder_pipeline_model_parallel_size
        * context_parallel_size
    )
    decoder_model_size = (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {total_model_size}")

    data_parallel_size: int = world_size // total_model_size

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    assert (
        encoder_world_size + decoder_world_size == world_size
    ), f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}"

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    # TODO: support expert specific ordering
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp."""
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == 'pp':
            # Map 1 encoder tp rank to several decoder tp ranks, because
            # these won't be the same size.
            for x, y in zip(cycle(e_ranks), d_ranks):
                yield x + y
        elif group_type == 'tp-pp':
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'

    for ranks in generator_wrapper('dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_GLOO = group_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    assert (
        data_parallel_size % num_distributed_optimizer_instances == 0
    ), 'Data parallel size should be divisible by partial DistOpt shard factor'
    intra_partial_data_parallel_size = data_parallel_size // num_distributed_optimizer_instances

    for ranks_with_cp in generator_wrapper('dp-cp'):
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        group_with_cp_gloo = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, backend="gloo"
        )

        if rank in ranks_with_cp:
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        if num_distributed_optimizer_instances > 1:
            # Create groups for Partial DistOpt, one for intra-partial DP domain
            # Another for inter-partial DP domain
            for i in range(num_distributed_optimizer_instances):
                intra_partial_data_parallel_ranks_with_cp = ranks_with_cp[
                    (i * intra_partial_data_parallel_size) : (
                        (i + 1) * intra_partial_data_parallel_size
                    )
                ]

                intra_partial_data_parallel_group_with_cp = torch.distributed.new_group(
                    intra_partial_data_parallel_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs),
                )
                intra_partial_data_parallel_group_with_cp_gloo = torch.distributed.new_group(
                    intra_partial_data_parallel_ranks_with_cp, timeout=timeout, backend="gloo"
                )

                if rank in intra_partial_data_parallel_ranks_with_cp:
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = (
                        intra_partial_data_parallel_group_with_cp
                    )
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = (
                        intra_partial_data_parallel_group_with_cp_gloo
                    )

            for i in range(intra_partial_data_parallel_size):
                inter_partial_data_parallel_ranks_with_cp = ranks_with_cp[
                    i::intra_partial_data_parallel_size
                ]

                inter_partial_data_parallel_group_with_cp = torch.distributed.new_group(
                    inter_partial_data_parallel_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs),
                )

                if rank in inter_partial_data_parallel_ranks_with_cp:
                    _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = (
                        inter_partial_data_parallel_group_with_cp
                    )
        else:
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP_WITH_CP
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = _DATA_PARALLEL_GROUP_WITH_CP_GLOO

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for ranks in generator_wrapper('cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
        if hierarchical_context_parallel_sizes:
            global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
            _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS += create_hierarchical_parallel_groups(
                rank,
                ranks,
                context_parallel_size,
                hierarchical_context_parallel_sizes,
                get_nccl_options('cp', nccl_comm_cfgs),
            )

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GLOBAL_RANKS
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for ranks in generator_wrapper('tp-pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
            _MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for ranks in generator_wrapper('tp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    ### Actual MUSA patch modification begins ###
    global _MTP_EMBEDDING_GROUP
    global _MTP_EMBEDDING_GLOBAL_RANKS
    assert _MTP_EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _MTP_POSITION_EMBEDDING_GROUP
    global _MTP_POSITION_EMBEDDING_GLOBAL_RANKS
    assert _MTP_POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    ### Actual MUSA patch modification ends ###
    for ranks in generator_wrapper('pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP is None:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                _PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP = [_PIPELINE_MODEL_PARALLEL_GROUP, group]
                _PIPELINE_GLOBAL_RANKS = [_PIPELINE_GLOBAL_RANKS, ranks]

        embedding_ranks = get_embedding_ranks(ranks)
        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        mtp_embedding_ranks = get_mtp_embedding_ranks(ranks)
        mtp_embedding_groups = torch.distributed.new_group(
            mtp_embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )

        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks
            ### Actual MUSA patch modification begins ###
            _MTP_EMBEDDING_GROUP = group
            _MTP_EMBEDDING_GLOBAL_RANKS = embedding_ranks
            _MTP_POSITION_EMBEDDING_GROUP = mtp_embedding_groups
            _MTP_POSITION_EMBEDDING_GLOBAL_RANKS = mtp_embedding_ranks
            ### Actual MUSA patch modification ends ###

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    for ranks in generator_wrapper('tp-dp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in generator_wrapper('tp-dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP = group

    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_CONTEXT_PARALLEL_GROUP is None
    ), 'Tensor + context parallel group is already initialized'
    for ranks in generator_wrapper('tp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    for ranks in generator_wrapper('ep', is_expert=True):
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group

    # Build the expert tensor parallel group
    global _EXPERT_TENSOR_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_PARALLEL_GROUP is None
    ), 'Expert tensor model parallel group is already initialized'
    for ranks in generator_wrapper('tp', is_expert=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _EXPERT_TENSOR_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is None
    ), 'Expert tensor + model parallel group is already initialized'
    for ranks in generator_wrapper('tp-ep', is_expert=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group

    # Build the expert+tensor+pipeline parallel groups
    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is None
    ), 'The expert_tensor_model_pipeline parallel group is already initialized'
    for ranks in generator_wrapper('tp-ep-pp', is_expert=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = group

    # Build the expert data parallel group
    global _EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, 'Expert data group is already initialized'
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert _EXPERT_DATA_PARALLEL_GROUP_GLOO is None, 'Expert data group-gloo is already initialized'

    for ranks in generator_wrapper('dp', is_expert=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            _EXPERT_DATA_PARALLEL_GROUP = group
            _EXPERT_DATA_PARALLEL_GROUP_GLOO = group_gloo
    ### End of expert related parallel groups initialization

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()
    
    for var in list(group_list.keys())[8:]:
        setattr(sys.modules["megatron.core.parallel_state"], var, eval(var))


def get_mtp_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _MTP_EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _MTP_EMBEDDING_GROUP

def get_mtp_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _MTP_POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _MTP_POSITION_EMBEDDING_GROUP

def is_rank_in_mtp_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _MTP_EMBEDDING_GLOBAL_RANKS
    if _MTP_EMBEDDING_GLOBAL_RANKS is None:
        return False
    if ignore_virtual:
        return rank in _MTP_EMBEDDING_GLOBAL_RANKS
    if rank in _MTP_EMBEDDING_GLOBAL_RANKS:
        if rank == _MTP_EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _MTP_EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False

import inspect, sys
group_list = {
    name: value for name, value in globals().items()
    if name.startswith("_") and not callable(value)
}
                
for k in sys.modules:
    if k.startswith('megatron'):
        for target in ['initialize_model_parallel']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, initialize_model_parallel)
                
        