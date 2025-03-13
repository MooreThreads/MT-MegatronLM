# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import time
from datetime import datetime

import torch

# the number of warmup steps before the active step in each profiling cycle
profile_freq = 4
# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_profiling(args, global_step):
    # get user defined profiler settings
    enable_profiling = int(os.getenv("ENABLE_PROFILER", 0))
     # fetch profiler related env
    wait_steps = int(os.getenv("PROFILER_WAIT_STEPS", 0))
    warmup_steps = int(os.getenv("PROFILER_WARMUP_STEPS", 3))
    active_steps = int(os.getenv("PROFILER_ACTIVE_STEPS", 1))
    repeat_num = int(os.getenv("PROFILER_REPEAT_NUM", 0))
    profile_freq = int(os.getenv("PROFILER_FREQ", 1))
    current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    save_dir = os.getenv("PROFILER_SAVE_DIR", f"./profiler_result/{current_time}")
    worker_name = os.getenv(
        "PROFILER_WORKER_NAME", "rank" + str(torch.distributed.get_rank())
    )
    record_shapes = int(os.getenv("PROFILER_RECORD_SHAPES", 1))
    profile_memory = int(os.getenv("PROFILER_PROFILE_MEMORY", 0))
    with_stack = int(os.getenv("PROFILER_WITH_STACK", 1))
    with_modules = int(os.getenv("PROFILER_WITH_MODULES", 1))
    kineto_log_level = int(os.getenv("KINETO_LOG_LEVEL", 0))

    if enable_profiling:
        profile_freq = profile_freq

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(save_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            print(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.pt.trace.json")
            print(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        print(f"Profiling active. Traces will be saved at {save_dir}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # warmup, active = WARMUP, 1
        wait = profile_freq - (active_steps + warmup_steps)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                #torch.profiler.ProfilerActivity.MUSA,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup_steps, active=active_steps),
            on_trace_ready=trace_handler,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_modules=with_modules,
            start_step=global_step+1,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(args, global_step: int = 0):
    pass
    # enable_snapshot = config.profiling.enable_memory_snapshot
    # if enable_snapshot:
    #     snapshot_folder = config.profiling.save_memory_snapshot_folder
    #     snapshot_dir = os.path.join(config.job.dump_folder, snapshot_folder)
    #     if not os.path.exists(snapshot_dir):
    #         os.makedirs(snapshot_dir, exist_ok=True)
    #     rank = torch.distributed.get_rank()

    #     class MemoryProfiler:
    #         def __init__(self, step_num: int, freq: int):
    #             torch.musa.memory._record_memory_history(
    #                 max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
    #             )
    #             # when resume training, we start from the last step
    #             self.step_num = step_num
    #             self.freq = freq

    #         def step(self, exit_ctx: bool = False):
    #             self.step_num += 1
    #             if not exit_ctx and self.step_num % self.freq != 0:
    #                 return
    #             if not exit_ctx:
    #                 curr_step = self.step_num
    #                 dir_name = f"iteration_{curr_step}"
    #             else:
    #                 # dump as iteration_0_exit if OOM at iter 1
    #                 curr_step = self.step_num - 1
    #                 dir_name = f"iteration_{curr_step}_exit"
    #             curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
    #             if not os.path.exists(curr_snapshot_dir):
    #                 os.makedirs(curr_snapshot_dir, exist_ok=True)
    #             logger.info(f"Dumping memory snapshot at step {curr_step}")
    #             begin = time.monotonic()
    #             with open(
    #                 f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle", "wb"
    #             ) as output:
    #                 pickle.dump(torch.musa.memory._snapshot(), output)
    #             logger.info(
    #                 f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
    #             )

    #     logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
    #     profiler = MemoryProfiler(global_step, config.profiling.profile_freq)
    #     try:
    #         yield profiler
    #     except torch.OutOfMemoryError as e:
    #         profiler.step(exit_ctx=True)
    # else:
    #     yield None
