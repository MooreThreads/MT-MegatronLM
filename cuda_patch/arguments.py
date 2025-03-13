import dataclasses
import torch
import torch.nn.functional as F

import megatron.training.arguments
from megatron.training.activations import squared_relu
from .transformer_config import TransformerConfig, MLATransformerConfig
moe_freq_type = megatron.training.arguments.moe_freq_type

def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")
    # General arguments
    group.add_argument('--expert-model-parallel-size', type=int, default=1,
                       help='Degree of expert model parallelism.')
    group.add_argument('--expert-tensor-parallel-size', type=int, default=None,
                       help='Degree of expert model parallelism. Default is None, which will be set to the value of --tensor-model-paralle-size.')
    group.add_argument('--num-experts', type=int, default=None,
                       help='Number of Experts in MoE (None means no MoE)')
    group.add_argument('--moe-layer-freq', type=moe_freq_type, default=1,
                       help='Frequency between MoE layers and Dense layers. Accepts either: '
                            '- An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers '
                            '- A string containing a Python list expression that defines a custom pattern, e.g.: '
                            '"([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0] '
                            'where 1 indicates an expert layer and 0 indicates a dense layer. '
                            'Examples: "([0]+[1]*23)": 1 dense layer followed by 23 experts layers, '
                            '"([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.')
    group.add_argument('--moe-ffn-hidden-size', type=int, default=None,
                       help='The hidden size of each expert\'s feed-forward network (ffn). '
                       'If not specified, defaults to the ffn_hidden_size.')
    group.add_argument('--moe-shared-expert-intermediate-size', type=int, default=None,
                       help='Shared expert total ffn hidden size. '
                       'It should be equal to "num_shared_experts * ffn_size_of_each_shared_expert" if there are multiple shared experts. '
                       'None means no shared expert.')
    group.add_argument('--moe-shared-expert-overlap', action='store_true',
                       help='Enable overlapping between shared expert computations and dispatcher communications. '
                       'Without this, the shared epxerts execute after the routed experts. '
                       'Only effective when moe-shared-expert-intermediate-size is set.')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine.')
    # Router arguments
    group.add_argument('--moe-router-load-balancing-type', type=str,
                       choices=['aux_loss', 'seq_aux_loss', 'sinkhorn', 'none'],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the load balancing loss used in DeepSeekV2, which computes the loss for each individual sample; "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".')
    group.add_argument('--moe-router-topk', type=int, default=2,
                       help='Number of experts to route to for each token. The default is 2.')
    group.add_argument('--moe-router-pre-softmax', action='store_true',
                       help='Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. By default, softmax is done after top-k.')
    group.add_argument('--moe-router-topk-limited-devices', type=int, default=None, 
                       help='Number of expert parallel ranks to consider for each token during routing. Perform top-k routing on a subset of expert parallel ranks by first selecting N ranks for each token, then conducting top-k selection among experts on these devices. Default is None, which means no limited devices.')
    group.add_argument('--moe-router-topk-scaling-factor', type=float, default=None,
                       help='Scaling factor for routing score in top-k selection, only works when --moe-router-pre-softmax enabled. Defaults to None, which means no scaling.')
    group.add_argument('--moe-use-legacy-grouped-gemm', action='store_true',
                       help='Use legacy GroupedMLP rather than TEGroupedMLP. Note: The legacy one will be deprecated soon.')
    group.add_argument('--moe-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    group.add_argument('--moe-device-level-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the device-level aux loss')
    group.add_argument('--moe-comm-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the communication aux loss')
    group.add_argument('--moe-z-loss-coeff', type=float, default=None,
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')
    group.add_argument('--moe-input-jitter-eps', type=float, default=None,
                       help='Add noise to the input tensor by applying jitter with a specified epsilon value.')
    group.add_argument('--moe-token-dispatcher-type', type=str,
                       choices=['allgather', 'alltoall', 'alltoall_seq'],
                       default='allgather',
                       help="The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather', 'alltoall' and 'alltoall_seq'. We recommend using 'alltoall' when applying expert parallelism. For more information, please refer to the documentation in core/moe/README.")
    group.add_argument('--moe-per-layer-logging', action='store_true',
                       help='Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.')
    # Token dropping arguments
    group.add_argument('--moe-expert-capacity-factor', type=float, default=None,
                       help='The capacity factor for each expert, None means no token will be dropped.')
    group.add_argument('--moe-device-level-capacity', action='store_true',
                       help='Whether to consider the expert capacity of a group together')
    group.add_argument('--moe-pad-expert-input-to-capacity', action='store_true',
                       help='Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set.')
    group.add_argument('--moe-token-drop-policy', type=str, default='probs', choices=['probs', 'position'],
                       help='The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.')
    group.add_argument('--moe-layer-recompute', action='store_true',
                       help='Enable checkpointing for moe_layer, should be used when memory is not sufficient.')
    group.add_argument('--moe-extended-tp', action='store_true',
                       help='Deprecated. Use --expert-tensor-parallel-size instead.')
    group.add_argument('--moe-use-upcycling', action='store_true',
                       help='Load a checkpoint of a dense model, convert it into an MoE model, and save the converted model to the path specified by --save. '
                       'Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model.')

    return parser


def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['first_pipeline_num_layers']= args.decoder_first_pipeline_num_layers
    kw_args['last_pipeline_num_layers']= args.decoder_last_pipeline_num_layers
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        kw_args['activation_func'] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir

    if len(args.cp_comm_type) == 1:
        kw_args['cp_comm_type'] = args.cp_comm_type[0]

    # Return config.
    return config_class(**kw_args)


megatron.training.arguments._add_moe_args = _add_moe_args
megatron.training.arguments.core_transformer_config_from_args = core_transformer_config_from_args