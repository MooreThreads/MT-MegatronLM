from datetime import datetime

import gc
import os
import sys
import time
import torch
from megatron.core import mpu
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.moe.router import TopKRouter
from megatron.training.global_vars import (
    get_args,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger
)
    # get_num_microbatches
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training.utils import (
    report_memory, 
    print_rank_last
)
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas
)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training import one_logger_utils
from megatron.training.initialize import write_args_to_tensorboard
from megatron.core.distributed import finalize_model_grads
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.training.training import (
    print_datetime, 
    save_checkpoint_and_time,
    train_step,
    evaluate_and_print_results,
    _TRAIN_START_TIME
)
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches
)
from megatron.training.utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    print_rank_0,
    print_rank_last,
    report_memory
)
from megatron.training import ft_integration
from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger
)
from .profiling import (
    maybe_enable_profiling,
    maybe_enable_memory_snapshot
)
from megatron.training.training import (
    enable_forward_pre_hook,
    disable_forward_pre_hook,
    post_training_step_callbacks,
    checkpoint_and_decide_exit
)


def throughput_calculator(args, elapsed_time_per_iter, consumed_tokens_per_iter): 
    # training_time = elapsed_time
    system_throughput = float(consumed_tokens_per_iter) / elapsed_time_per_iter
    world_size = args.world_size
    chip_throughput = system_throughput / world_size
    # For 70B
    # all_param_num = getattr(args, "all_param_num", None)
    # assert all_param_num is not None, "please set all_param_num"
    # MFU = chip_throughput * 6 * all_param_num * (1 + args.seq_length / (6 * args.hidden_size) ) / 98e12
    # # tflops_throughput = chip_throughput / float(config.flops_16bit) * 1e12
    # # logger.info("Throughput(token per chip per second): " + str(chip_throughput))
    # # logger.info("MFU: " + str(MFU))
    # # logger.info("Throughput(token per TFLOPS): " + str(tflops_throughput))
    h = args.hidden_size
    s = args.seq_length
    N = 12 * args.num_layers * h **2
    D = 1

    attn_matmul = 2 * N * D
    attn_sdp = N * D * (s / h)
    mlp_matmul = 4 * N * D
    # moe
    if args.num_experts is None:
        factor = 1
    else:
        factor = args.moe_router_topk
    activated_dense_flops = attn_matmul + attn_sdp + mlp_matmul * factor
    if args.num_experts is not None:
        act_params = N + args.num_layers *(args.num_experts - 1) * 8 * h**2
        if torch.distributed.get_rank() == 0:
            print(f"N: {N} Act param: {act_params} Act flops: {activated_dense_flops}")
    tflops =  chip_throughput *  activated_dense_flops
    mfu = tflops / 98e12

    return chip_throughput, mfu

def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    if not args.multi_latent_attention:
        return (
            12
            * batch_size
            * args.seq_length
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (
                        1
                        + (args.num_query_groups / args.num_attention_heads)
                        + (args.seq_length / args.hidden_size)
                    ) * query_projection_to_hidden_size_ratio
                )
                # MLP.
                + (
                    (args.moe_ffn_hidden_size / args.hidden_size)
                    * num_experts_routed_to
                    * gated_linear_multiplier
                )
                # Shared Experts.
                + ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
                # Logit.
                + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
            )
        )
    else:
        if args.q_lora_rank is None:
            mla_flops_q = args.hidden_size * args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
        else:
            mla_flops_q = args.hidden_size * args.q_lora_rank +\
                  args.q_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
        return (
            6
            * batch_size
            * args.seq_length
            * args.num_layers
            * (
                # MLA Attention.
                (
                    (
                        mla_flops_q
                        + args.hidden_size * (args.kv_lora_rank + args.qk_pos_emb_head_dim)
                        + args.num_attention_heads * args.kv_lora_rank * (args.qk_head_dim + args.v_head_dim)
                        + args.num_attention_heads * args.seq_length * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                        + args.num_attention_heads * args.seq_length * args.v_head_dim
                        + args.num_attention_heads * args.v_head_dim * args.hidden_size
                    ) 
                )
                # Router
                + args.hidden_size * args.num_experts
                # MLP.
                + (
                    2 * args.hidden_size *  args.moe_ffn_hidden_size * num_experts_routed_to
                    * gated_linear_multiplier
                )
                # Shared Experts.
                + (2 * args.hidden_size * shared_expert_ffn_hidden_size * gated_linear_multiplier)
                # Logit.
                + (args.padded_vocab_size * args.hidden_size / args.num_layers)
            )
        )


def update_e_score_correction_bias(model, micro_batch_num, ):
    """ 
    Decrease the bias term by gamma if its corresponding expert is overloaded, 
    and increase it by gamma if its corresponding expert is underloaded.
    """
    for layer in model.modules():
        if isinstance(layer, TopKRouter):
            if layer.moe_noaux_gamma:
                # allreduce tokens_per_expert
                layer.tokens_per_expert /= micro_batch_num
                torch.distributed.all_reduce(layer.tokens_per_expert, op=torch.distributed.ReduceOp.SUM)
                layer.tokens_per_expert /= torch.distributed.get_world_size()
                with torch.no_grad():
                    update_value = torch.where(
                        layer.tokens_per_expert > 1,
                        -layer.moe_noaux_gamma, 
                        layer.moe_noaux_gamma
                    )
                    layer.e_score_correction_bias += update_value
                    layer.tokens_per_expert.zero_()  

def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        writer.add_scalar('learning-rate', learning_rate, iteration)
        if args.decoupled_lr is not None:
            writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
        writer.add_scalar('learning-rate vs samples', learning_rate,
                          args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.skipped_train_samples > 0:
            writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
            if wandb_writer:
                wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)
        writer.add_scalar('batch-size', batch_size, iteration)
        writer.add_scalar('batch-size vs samples', batch_size,
                          args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        # mfu = throughput / 465
        # tokens_per_gpu_per_second = float(batch_size * args.seq_length) / elapsed_time_per_iteration / args.world_size
        # log_string += ' tokens_per_gpu_per_second: {:.2f} /s |'.format(tokens_per_gpu_per_second)
        # log_string += ' mfu: {:.4f} |'.format(mfu)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        if args.skipped_train_samples > 0:
            log_string += ' skipped samples: {:12d} |'.format(
                args.skipped_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag

def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context, non_loss_data_func):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Track E2E metrics at the start of training
    one_logger_utils.on_train_start(iteration=iteration, consumed_train_samples=args.consumed_train_samples,
                                    train_samples=args.train_samples, seq_length=args.seq_length,
                                    train_iters=args.train_iters, save=args.save, async_save=args.async_save,
                                    log_throughput=args.log_throughput,
                                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far)

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    # exit = False
    pre_hook_enabled = False 
    should_exit = False
    exit_code = 0
    
    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    # total_flops = 0.0
    num_floating_point_operations_since_last_log_event = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics.
        """
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops': num_floating_point_operations_since_last_log_event,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length
        }
    # Cache into one-logger for callback
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    prof = None
    if args.profile and torch.distributed.get_rank() in args.profile_ranks and args.use_pytorch_profiler:
        prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start-1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end-args.profile_step_start,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
        record_shapes=True,
        with_stack=True)
        prof.start()

    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if args.use_distributed_optimizer and args.overlap_param_gather:
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if args.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), \
            "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
        
    with maybe_enable_profiling(
        args, global_step=iteration
    ) as torch_profiler:
        while iteration < args.train_iters:
            if args.profile and torch.distributed.get_rank() in args.profile_ranks:
                if args.use_pytorch_profiler:
                    prof.step()
                elif iteration == args.profile_step_start:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

            maybe_finalize_async_save(blocking=False)

            # Update number of microbatches first without consistency check to decide if a
            # checkpoint should be saved. If the number of microbatches is different
            # from the previous iteration, save a checkpoint. Then run consistency check
            # to make sure training configuration is still valid.
            update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
            if get_num_microbatches() != num_microbatches and iteration != 0:
                assert get_num_microbatches() > num_microbatches, \
                    "number of microbatches should be increasing due to batch size rampup ... %d -> %d." % (num_microbatches, get_num_microbatches())
                if args.save is not None:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                            opt_param_scheduler,
                                            num_floating_point_operations_so_far,
                                            checkpointing_context, train_data_iterator=train_data_iterator)
            num_microbatches = get_num_microbatches()
            update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)

            args.curr_iteration = iteration
            loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = \
                train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config)
            if config.moe_noaux_gamma:
                update_e_score_correction_bias(model[0], num_microbatches)

            if should_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                        opt_param_scheduler,
                                        num_floating_point_operations_so_far,
                                        checkpointing_context, train_data_iterator=train_data_iterator)
            if should_exit:
                break

            # Enable forward pre-hooks after first set of forward and backward passes.
            # When running in fp16, skip all NaN iterations until steady-state loss scaling value
            # is reached.
            if iteration == start_iteration:
                if skipped_iter:
                    # Only enable forward pre-hook after a training step has successfully run. Relevant
                    # for fp16 codepath where first XX iterations are skipped until steady-state loss
                    # scale value is reached.
                    start_iteration = iteration + 1
                else:
                    # Enable forward pre-hook after training step has successfully run. All subsequent
                    # forward passes will use the forward pre-hook / `param_sync_func` in
                    # `forward_backward_func`.
                    if args.use_distributed_optimizer and args.overlap_param_gather:
                        enable_forward_pre_hook(model)
                        config.param_sync_func = param_sync_func
                        pre_hook_enabled = True
                    
            if torch_profiler:
                torch_profiler.step()
            iteration += 1
            batch_size = mpu.get_data_parallel_world_size() * \
                        args.micro_batch_size * \
                        get_num_microbatches()
            args.consumed_train_samples += batch_size
            num_skipped_samples_in_batch = (get_current_global_batch_size() -
                                            get_current_running_global_batch_size())
            if args.decrease_batch_size_if_needed:
                assert num_skipped_samples_in_batch >= 0
            else:
                assert num_skipped_samples_in_batch == 0
            args.skipped_train_samples += num_skipped_samples_in_batch
            num_floating_point_operations_in_batch = num_floating_point_operations(args, batch_size)
            num_floating_point_operations_so_far += num_floating_point_operations_in_batch
            num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

            # # Send heartbeat to FT package and update timeouts.
            # if args.enable_ft_package:
            #     ft_client = ft_integration.get_rank_monitor_client(
            #         ft_integration.StateMachineActions.TRAIN_HEARTBEAT)
            #     if ft_client is not None:
            #         ft_client.send_heartbeat()
            #         # TODO we are always calculating timeouts in the current implementation
            #         # if we want to rely on manually setup then we need to add additional argument
            #         # to training and pass it here
            #         if ft_integration.can_update_timeouts():
            #             ft_integration.get_rank_monitor_client(
            #                 ft_integration.StateMachineActions.UPDATE_TIMEOUT).calculate_and_set_timeouts()
            #             print_rank_0(f'Updated FT timeouts. New values: \
            #                 {ft_integration.get_rank_monitor_client().timeouts}')

            # # Bring CPU and GPU back in sync if on right iteration.
            # if (
            #     args.train_sync_interval
            #     and iteration % args.train_sync_interval == 0
            # ):
            #     torch.cuda.synchronize()

            # Logging.
            if not optimizer.is_stub_optimizer:
                loss_scale = optimizer.get_loss_scale().item()
            else:
                loss_scale = 1.0
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)

            learning_rate = None
            decoupled_learning_rate = None
            for param_group in optimizer.param_groups:
                if param_group['is_decoupled_lr']:
                    decoupled_learning_rate = param_group['lr']
                else:
                    learning_rate = param_group['lr']
            report_memory_flag = training_log(loss_dict, total_loss_dict,
                                            learning_rate,
                                            decoupled_learning_rate,
                                            iteration, loss_scale,
                                            report_memory_flag, skipped_iter,
                                            grad_norm, params_norm, num_zeros_in_grad)

            # # StragglerDetector
            # if iteration % args.log_interval == 0 and args.log_straggler:
            #     stimer.report(total_flops, args.log_interval)
            #     total_flops = 0.0

            # if args.check_weight_hash_across_dp_replicas_interval is not None and \
            #         iteration % args.check_weight_hash_across_dp_replicas_interval == 0:
            #     if args.use_distributed_optimizer and args.overlap_param_gather:
            #         optimizer.disable_pre_hook()
            #     assert check_param_hashes_across_dp_replicas(model, cross_check=True), \
            #         "Parameter hashes not matching across DP replicas"
            #     torch.distributed.barrier()
            #     print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
            #     if args.use_distributed_optimizer and args.overlap_param_gather:
            #         optimizer.enable_pre_hook()

            # # Autoresume
            # if args.adlr_autoresume and \
            # (iteration % args.adlr_autoresume_interval == 0):
            #     check_adlr_autoresume_termination(iteration, model, optimizer,
            #                                     opt_param_scheduler)

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and \
                args.do_valid:
                timers('interval-time').stop()
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    disable_forward_pre_hook(model)
                    pre_hook_enabled = False
                if args.manual_gc and args.manual_gc_eval:
                    # Collect all objects.
                    gc.collect()
                prefix = f'iteration {iteration}'
                timers('eval-time', log_level=0).start(barrier=True)
                evaluate_and_print_results(prefix, forward_step_func,
                                        valid_data_iterator, model,
                                        iteration, process_non_loss_data_func,
                                        config, verbose=False, write_to_tensorboard=True,
                                        non_loss_data_func=non_loss_data_func)
                eval_duration += timers('eval-time').elapsed()
                eval_iterations += args.eval_iters
                timers('eval-time').stop()
                one_logger_utils.track_e2e_metrics()

                if args.manual_gc and args.manual_gc_eval:
                    # Collect only the objects created and used in evaluation.
                    gc.collect(generation=0)
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    enable_forward_pre_hook(model)
                    pre_hook_enabled = True
                timers('interval-time', log_level=0).start(barrier=True)


                if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
                    ft_integration.get_rank_monitor_client(
                        ft_integration.StateMachineActions.EVAL_HEARTBEAT).send_heartbeat()

            # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
            # Some of these only happen at specific iterations.
            post_training_step_callbacks(model, optimizer, opt_param_scheduler, iteration, prof,
                                        num_floating_point_operations_since_last_log_event)

            # Checkpoint and decide whether to exit.
            should_exit = checkpoint_and_decide_exit(model, optimizer, opt_param_scheduler, iteration,
                                                    num_floating_point_operations_so_far,
                                                    checkpointing_context, train_data_iterator)
            if should_exit:
                break

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard, WandB writers and one-logger.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
        ft_integration.get_rank_monitor_client().shutdown_workload_monitoring()

    maybe_finalize_async_save(blocking=True)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        sys.exit(exit_code)

    return iteration, num_floating_point_operations_so_far

import megatron.training
megatron.training.training.training_log = training_log

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
# if enable_profiler:
megatron.training.training.train = train