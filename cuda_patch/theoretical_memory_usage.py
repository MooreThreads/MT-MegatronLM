# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""


import math

NUM_BYTES_IN_MEGABYTE = 1024 * 1024

def compute_weight_and_optimizer_memory(args, verbose=False):
    # Attention projection size.
    attn_dim = args.kv_channels
    kv_projection_size = attn_dim * args.num_attention_heads
    query_projection_size = attn_dim * args.num_attention_heads
    ## MLA
    if args.kv_lora_rank:
        kv_projection_size = attn_dim * args.num_attention_heads + args.kv_lora_rank
    if args.q_lora_rank:
        query_projection_size = attn_dim * args.num_attention_heads + args.q_lora_rank

    output_projection_size = attn_dim * args.num_attention_heads
    ## Group Query Attention.
    if args.group_query_attention:
        kv_projection_size = args.num_query_groups / args.num_attention_heads * kv_projection_size
    else:
        attn_size = 1
    attn_size = query_projection_size + 2 * kv_projection_size + output_projection_size
    attn_multiplier = attn_size / 2 / args.hidden_size

    # swiglu
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    # MoE or Dense
    num_experts = 0 if args.num_experts is None else args.num_experts
    shared_expert_ffn_hidden_size = 0 if args.moe_shared_expert_intermediate_size else args.moe_shared_expert_intermediate_size
    num_shared_expert = args.moe_shared_expert_intermediate_size // args.moe_ffn_hidden_size
    num_moe_layer = 0 if args.moe_layer_freq is None else len(args.moe_layer_freq)
    num_dense_layer = args.num_layers - num_moe_layer
    mlp_multiplier_dense = num_dense_layer * args.ffn_hidden_size
    mlp_multiplier_moe =  num_moe_layer * num_experts * args.moe_ffn_hidden_size 
    mlp_multiplier_shared_expert = num_moe_layer * num_shared_expert * shared_expert_ffn_hidden_size
    mlp_multiplier = (mlp_multiplier_dense + mlp_multiplier_moe + mlp_multiplier_shared_expert) /args.num_layers/args.hidden_size

    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            attn_multiplier
            # MLP.
            # + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
            + mlp_multiplier * gated_linear_multiplier
            # Router
            +  num_experts / args.hidden_size * 2
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (args.num_layers * args.hidden_size))
        )
    )

    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    # TODO; add MTP block and projection
    num_parameters_in_mtp = 0

    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers + num_parameters_in_mtp
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / args.tensor_model_parallel_size


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )

    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not args.sequence_parallel or args.recompute_granularity != 'selective':
        print(
            f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB"
        )
        return

    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )

import megatron.training.theoretical_memory_usage
megatron.training.theoretical_memory_usage.compute_weight_and_optimizer_memory = compute_weight_and_optimizer_memory
megatron.training.theoretical_memory_usage.report_theoretical_memory = report_theoretical_memory
megatron.training.theoretical_memory_usage.compute_activation_memory = compute_activation_memory
