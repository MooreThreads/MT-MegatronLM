'''
Test the NCCL communication performance
'''
import os
import time

from numpy import mean, percentile
import pandas as pd
import argparse

import torch
import torch.nn.parallel
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

try:
    import torch_musa
    _USE_MUSA = True
    _sync = torch_musa._MUSAC._musa_synchronize
    _DEVICE_PREFIX = 'musa'
    _set_device = torch_musa.set_device
    _BACKEND = 'mccl'
    # _backend = 'gloo'
    # _device_prefix = 'cpu'
except Exception as e:
    _USE_MUSA = False
    _sync = torch.cuda.synchronize
    _DEVICE_PREFIX = 'cuda'
    _set_device = torch.cuda.set_device
    _BACKEND = 'nccl'

    print(f"torch_musa not imported. _USE_MUSA: {_USE_MUSA}")
    print(e)

_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None
_EXPERT_MODEL_PARALLEL_GROUP = None
_EXPERT_MODEL_PARALLEL_GROUP_RANKS = None

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

print(f"WORLD_RANK: {WORLD_RANK}")

prefix_unit = {
        'B':1,
        'KB':1024,
        'MB':1024*1024,
        'GB':1024*1024*1024,
        }

def setup():
    '''initialize the process group'''
    dist.init_process_group(_BACKEND, rank=WORLD_RANK, world_size=WORLD_SIZE)

def cleanup():
    '''destroy all groups'''
    dist.destroy_process_group()

def get_tensor_model_parallel_group():
    global _TENSOR_MODEL_PARALLEL_GROUP
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_world_size():
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_rank():
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_expert_model_parallel_group():
    global _EXPERT_MODEL_PARALLEL_GROUP
    return _EXPERT_MODEL_PARALLEL_GROUP

def get_expert_model_parallel_world_size():
    return torch.distributed.get_world_size(group=get_expert_model_parallel_group())

def get_expert_model_parallel_rank():
    return torch.distributed.get_rank(group=get_expert_model_parallel_group())


from typing import List, Optional
def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example, 
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then 
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the 
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the 
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).
        
        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        if 'ep' in order:
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)
        return ranks

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device="cuda",
            )
        # print(f"{torch.distributed.get_rank()} trigger all to all {world_size} output {output.size()} input {input.size()} | {output_split_sizes} {input_split_sizes} \n")
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        # print(f"{torch.distributed.get_rank()} trigger all to all done")
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        print(f"{torch.distributed.get_rank()} trigger all to all bwd")
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )

def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

def _reduce_scatter_along_last_dim(input_):
    """Reduce-scatter tensors on the last dimension."""
    num_dims = input_.dim()
    permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))

    input_ = input_.permute(permute_order).contiguous()
    output = _reduce_scatter_along_first_dim(input_)

    permute_order = tuple(range(1, num_dims)) + (0,)
    output = output.permute(permute_order).contiguous()
    return output

def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device="musa")
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )
    return output

class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"{torch.distributed.get_rank()} trigger _reduce_scatter_along_last_dim {grad_output.size()}")
        return _reduce_scatter_along_last_dim(grad_output)
    
class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)

def all_gather_last_dim_from_tensor_parallel_region(input_):
    return _AllGatherFromTensorParallelRegion.apply(input_)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_):
    return _ReduceScatterToTensorParallelRegion.apply(input_)


def setup_tp_group():
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert WORLD_SIZE == 4, "only trigger tp2dp4 case"
    rank_generator = RankGenerator(
        tp=2,
        ep=2,
        dp=2,
        pp=1,
        cp=1,
        order="tp-cp-ep-dp-pp"
    )
    from datetime import timedelta
    timeout = timedelta(minutes=10)
    for ranks in rank_generator.get_ranks('tp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=None
        )
        if LOCAL_RANK in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks
            print(torch.distributed.get_rank(), " ", _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS)

    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_GROUP_RANKS
    for ranks in rank_generator.get_ranks('ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, pg_options=None
        )
        if LOCAL_RANK in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group
            _EXPERT_MODEL_PARALLEL_GROUP_RANKS = ranks
            print(torch.distributed.get_rank(), " ", _EXPERT_MODEL_PARALLEL_GROUP_RANKS)

def mock_all2all_with_sp():
    setup()
    setup_tp_group()
    
    _set_device(LOCAL_RANK)
    def all_to_all_sp2hp(input_):
        world_size = get_tensor_model_parallel_world_size()
        input_ = input_.reshape(-1, input_.shape[-1])
        split_tensors = torch.split(
            input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
        )
        tp_group = get_tensor_model_parallel_group()
        concat_tensor = torch.cat(split_tensors, dim=0)
        output = all_to_all(tp_group, concat_tensor)
        return output


    def all_to_all_hp2sp(input_):
        world_size = get_tensor_model_parallel_world_size()
        input_ = input_.reshape(-1, input_.shape[-1])
        tp_group = get_tensor_model_parallel_group()
        input_exchanged = all_to_all(tp_group, input_)
        input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
        split_tensors = torch.split(
            input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
        )
        output = torch.cat(split_tensors, dim=-1)
        return output

    
    def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None):
        return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_)
    batch_size, seq_len, hidden_size = 2, 512, 768
    tp_size = 2
    
    repeat_num = 100
    depth = 100
    if LOCAL_RANK == 0:
        input_split_sizes_ = [128, 0]
        output_split_sizes_ = [128, 128]
    else:
        input_split_sizes_ = [0, 128]
        output_split_sizes_ = [0, 0]
    hidden_states = torch.randn([0, 5]).to("cuda")   
    # for i in range(repeat_num):
    #     output = all_to_all(get_expert_model_parallel_group(), hidden_states, output_split_sizes_,  input_split_sizes_)
    #     print(f">>>> {LOCAL_RANK} {i} {output.size()} |")
    for i in range(repeat_num):
        hidden_states = torch.randn([0, 5]).to("cuda")
        print(f">>>> {LOCAL_RANK} {i} |","hidden_states ", hidden_states.size())
        output = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
        print(f">>>> {LOCAL_RANK} {i} |", "output ", output.size())
    # for _ in range(repeat_num):
    #     hidden_states = torch.randn([seq_len // tp_size * batch_size, hidden_size]).musa()
    #     weight = torch.nn.Linear(hidden_size, hidden_size, device="musa")
    #     hidden_states = weight(hidden_states)
    #     for i in range(depth):
    #         # all 2 all
    #         print(f"{i} 1 {hidden_states.size()}")
    #         hidden_states = all_to_all_sp2hp(hidden_states)
    #         print(f"{i} 2 {hidden_states.size()}")
    #         hidden_states = all_to_all(get_expert_model_parallel_group(), hidden_states)
    #         # all gather base
    #         hidden_states = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
    #         print(f"{i} 3 {hidden_states.size()}")
    #         # reduce scatter base
    #         hidden_states = reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states)
    #         # all 2 all
    #         hidden_states = all_to_all(get_expert_model_parallel_group(), hidden_states)
    #         print(f"{i} 4 {hidden_states.size()}")
    #         hidden_states = all_to_all_hp2sp(hidden_states)
    #         print(f"{i} 5 {hidden_states.size()}")
    #     loss = hidden_states.sum()
    #     loss.backward()

    print("done")

if __name__ == "__main__":
    mock_all2all_with_sp()