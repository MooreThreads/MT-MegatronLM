
import math
from typing import Optional

import torch

from megatron.core import parallel_state
import megatron.core.transformer.moe.moe_utils
get_capacity = megatron.core.transformer.moe.moe_utils.get_capacity
device_limited_topk = megatron.core.transformer.moe.moe_utils.device_limited_topk


def node_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    moe_router_topk_limited_devices: int,
    num_node_group: int=None,
):
    """Perform top-k routing on a subset of expert parallel ranks.

    Selects N ranks for each token, then conducts top-k selection among experts on these node.
    See DeepSeek-V3 technical report for details.

    Args:
        scores (torch.Tensor): Softmax scores from the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """

    # Organize the experts into groups
    if num_node_group is None:
        ep_size = (
            parallel_state.get_expert_model_parallel_world_size()
        )  # num_node_group equals to expert parallel size/8
        assert ep_size % 8 == 0, f"ep_size should be multiple of 8, but get {ep_size}"
        num_node_group = ep_size // 8
    node_k = topk // moe_router_topk_limited_devices #each token select node according to the sum of the highest K/M affinity scores
    group_scores = (
                scores.view(num_tokens, num_node_group, -1).topk(node_k, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
    group_idx = torch.topk(
                group_scores, k=moe_router_topk_limited_devices, dim=-1, sorted=False
            )[
                1
            ]  # [n, moe_router_topk_limited_devices]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_node_group, num_experts // num_node_group)
        .reshape(num_tokens, -1)
    )  # [n, e]
    masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    _, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return top_indices


def sequence_load_balancing_loss_func(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    batch_size: int,
    seq_length: int,
    topk: int,
    moe_aux_loss_coeff: float,
    moe_device_level_aux_loss_coeff: float=None,
    moe_comm_aux_loss_coeff: float=None,
    moe_router_topk_limited_devices: float=None,
    moe_complementary_seq_aux_loss: bool=False,
    sequence_partition_group=None,
):
    """
    Calculate the auxiliary loss in sequence-level by computing the loss for each individual sample.
    Refer to the DeepSeek-V2 huggingface repo
    (https://huggingface.co/deepseek-ai/DeepSeek-V2) for details.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    assert num_sub_sequence == 1, "Do not support sequence aux loss in sequence partition case"

    num_experts = probs.shape[1]

    probs_for_aux_loss = probs.view(seq_length, batch_size, -1)
    cost_coeff = routing_map.view(seq_length, batch_size, -1).sum(dim=0).float()
    cost_coeff.div_(seq_length * topk / num_experts)
    if moe_complementary_seq_aux_loss:
        assert (
            (moe_device_level_aux_loss_coeff is None) and 
            (moe_comm_aux_loss_coeff is None)
            ), "moe_complementary_seq_aux_loss only used in deepseekV3, which means no other aux loss used"
        probs_for_aux_loss = probs.view(seq_length, batch_size, -1)
        sum_value = probs_for_aux_loss.sum(dim=-1, keepdim=True)
        probs_for_aux_loss = probs_for_aux_loss / (sum_value + 1e-20)
    seq_aux_loss = (cost_coeff * probs_for_aux_loss.mean(dim=0)).sum(dim=1).mean()
    seq_aux_loss *= moe_aux_loss_coeff

    if moe_device_level_aux_loss_coeff is not None:
        num_group = (
        parallel_state.get_expert_model_parallel_world_size()
        )  # num_group equals to expert parallel size
        device_aux_loss = (cost_coeff.view(batch_size, num_group, -1).mean(dim=2) * 
                           probs_for_aux_loss.mean(dim=0).view(batch_size, num_group, -1).sum(dim=2)).sum(dim=1).mean()
        device_aux_loss *= moe_device_level_aux_loss_coeff
        seq_aux_loss += device_aux_loss
    if moe_comm_aux_loss_coeff is not None:
        num_group = (
        parallel_state.get_expert_model_parallel_world_size()
        )  # num_group equals to expert parallel size
        cost_coeff = routing_map.view(seq_length, batch_size, num_group, -1).any(dim=3).sum(dim=0).float()
        cost_coeff.div_(seq_length *  moe_router_topk_limited_devices/ num_group)
        comm_aux_loss = (cost_coeff * 
                           probs_for_aux_loss.mean(dim=0).view(batch_size, num_group, -1).sum(dim=2)).sum(dim=1).mean()
        comm_aux_loss *= moe_comm_aux_loss_coeff
        seq_aux_loss += comm_aux_loss
        
    return seq_aux_loss

def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: Optional[float] = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    moe_router_topk_limited_devices: int = None,
    moe_router_topk_scaling_factor: float = None,
    device_level_capacity: Optional[bool] = False,
    use_sigmoid: bool = False,
    norm_topk_prob: bool = False,
    num_node_group: int = None,
    e_score_correction_bias: torch.Tensor = None,
    deterministic_mode: bool = False,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number
                               of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position".
                           If "prob", the tokens with the lowest probabilities will be dropped.
                           If "position", tokens at the end of each batch will be dropped.
        use_pre_softmax (bool): Whether to apply softmax before top-k selection.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.
        moe_router_topk_scaling_factor (float): Scaling factor for routing score in top-k
            selection, only works when use_pre_softmax enabled.
        deterministic_mode (bool): Deprecated.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
            - tokens_per_expert (torch.Tensor): A tensor of shape [num_experts] containing
              the number of local tokens assigned to each expert before dropping and padding.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    if use_pre_softmax:
        # Pre softmax
        if use_sigmoid:
            scores = torch.sigmoid(logits).type_as(logits)
        else:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)

        if e_score_correction_bias is not None:
            scores_gate = scores + e_score_correction_bias.unsqueeze(0)  #correction only used in router not in multiplied ffn output
        else:
            scores_gate = scores

        if moe_router_topk_limited_devices:
            if num_node_group:
                top_indices = node_limited_topk(
                    scores_gate, topk, num_tokens, num_experts, moe_router_topk_limited_devices, num_node_group
                )
                probs = scores.gather(1, top_indices)
            else:
                probs, top_indices = device_limited_topk(
                    scores, topk, num_tokens, num_experts, moe_router_topk_limited_devices
                )
        else:
            probs, top_indices = torch.topk(scores, k=topk, dim=1)

        # Normalize the probs.
        if norm_topk_prob:
            assert use_sigmoid, f"norm_topk_prob only work with use_sigmoid=True, but get {use_sigmoid}"
            denominator = probs.sum(dim=-1, keepdim=True) + 1e-20
            probs = probs / denominator
        if moe_router_topk_scaling_factor:
            probs = probs * moe_router_topk_scaling_factor
    else:
        # Post softmax
        if topk == 1:
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
        assert (
            moe_router_topk_scaling_factor is None
        ), "moe_router_topk_scaling_factor is not supported with post-softmax"
        if moe_router_topk_limited_devices:
            if num_node_group:
                scores, top_indices = node_limited_topk(
                    logits, topk, num_tokens, num_experts, moe_router_topk_limited_devices, num_node_group
                )
            else:
                scores, top_indices = device_limited_topk(
                    logits, topk, num_tokens, num_experts, moe_router_topk_limited_devices
                )
        else:
            scores, top_indices = torch.topk(logits, k=topk, dim=1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    # TODO Try using element-wise operations instead of scatter?
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    tokens_per_expert = topk_map.sum(dim=0)

    if capacity_factor is None:
        # TopK without capacity
        return topk_masked_gates, topk_map, tokens_per_expert
    elif device_level_capacity:
        assert drop_policy=='probs', f"only support 'probs' for device_level capacity, but get {drop_policy}"
        num_group = (
        parallel_state.get_expert_model_parallel_world_size()
        )  # num_group equals to expert parallel size
        device_expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        )*num_experts//num_group
        # Maskout exceeded tokens
        if drop_policy == "probs":
            topk_masked_group_gates = topk_masked_gates.view(num_tokens, num_group, -1)
            topk_masked_group_gates = topk_masked_group_gates.permute(0,2,1).reshape(-1, num_group)
            _, capacity_indices = torch.topk(
                topk_masked_group_gates, k=device_expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros([num_tokens*num_experts//num_group, num_group], device=logits.device).scatter(0, capacity_indices, 1).bool()
            capacity_mask = capacity_mask.view(num_tokens, num_experts//num_group, num_group).permute(0,2,1).reshape(num_tokens, -1)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        )

        # Maskout exceeded tokens
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert


megatron.core.transformer.moe.moe_utils.sequence_load_balancing_loss_func = sequence_load_balancing_loss_func
megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity = topk_softmax_with_capacity