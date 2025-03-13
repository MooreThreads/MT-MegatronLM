#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import paramiko
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


def _process_logs(logs):
    # loss_dict = OrderedDict()
    # lr_dict = OrderedDict()
    res = {"lm_loss": {}, "z_loss": {}, "aux_loss": {}, "lr": {}}
    need_keys = {"lm loss:" : "lm_loss", "z_loss:" : "z_loss", "load_balancing_loss:" : "aux_loss", "learning rate:": "lr"}
    for num, log in enumerate(logs):
        # log_split = log.split('|')
        # if len(log_split) != LOG_LEN:
        #     continue
        log_split = log.split('] iteration')
        if len(log_split) != 2:
            continue
        iteration = int(log_split[1].split("/")[0].strip())
        for k,v in need_keys.items():
            extract_value = float(log_split[1].split(k)[1].split("|")[0].strip())
            res[v][iteration] = extract_value
    return res


def _load_local_file(base_dir, file):
    with open(os.path.join(base_dir, file), "r") as f:
        logs = f.readlines()
    res = _process_logs(logs)
    return res


def stat(cuda_info, musa_info):
    cuda, cuda_lr = cuda_info["lm_loss"], cuda_info["lr"]
    musa, musa_lr = musa_info["lm_loss"], musa_info["lr"]
    c_ks = len(cuda.keys())
    m_ks = len(musa.keys())
    ks = cuda.keys() if c_ks < m_ks else musa.keys()
    x_axis, c_axis, m_axis, diff_ratios = [], [], [], []
    c_lr, m_lr = [], []

    # ==== stat ====
    for i, k in enumerate(ks):
        if STEPS > 9000 and i >= STEPS:
            break
        x_axis.append(int(k) - 1) # iter
        c_loss = cuda[k]
        m_loss = musa[k]
        diff_ratio = abs(m_loss - c_loss) / c_loss * 100
        c_axis.append(c_loss)
        m_axis.append(m_loss)
        diff_ratios.append(diff_ratio)
        c_lr.append(cuda_lr[k])
        m_lr.append(musa_lr[k])

    # ==== plotting ====
    plt.figure(figsize=(10, 8))
    # loss
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, c_axis, color='green', linestyle='-.')
    plt.plot(x_axis, m_axis, color='orange', linestyle='-.')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1000))
    plt.title("loss compare")
    plt.ylabel("loss")
    plt.legend(["cuda", "musa"], loc='upper right')
    # lr
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, c_lr, color='green', linestyle='-.')
    plt.plot(x_axis, m_lr, color='orange', linestyle='-.')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1000))
    plt.title("lr compare")
    plt.xlabel("iterations")
    plt.ylabel("lr")
    plt.legend(["cuda", "musa"], loc='upper right')

    plt.show()
    print(f"diff: {np.mean(diff_ratios):.3f}(+-{np.var(diff_ratios):.3f})%")


if __name__ == "__main__":
    model = "mixtral"
    BASE_DIR_CUDA = "/data2/yutian.rong/projects/megatron-lm-musa-patch/examples/mixtral/output/compare_cuda"
    BASE_DIR_MUSA = "/data2/yutian.rong/projects/megatron-lm-musa-patch/examples/mixtral/output/compare_musa"
    STEPS = int(os.getenv("STEPS", 5000))

    # ======== Mixtral ==========
    # LOG_LEN = 14
    # CUDA_FILE = "2024-05-17_04:00:23/tp1_pp1_dp4_mbs1_numbs1_gbs4_gpus4.log.0.172.31.208.8"
    # CUDA_FILE = "2024-05-23_03:37:40/tp1_pp1_dp2_mbs1_numbs1_gbs2_gpus2.log.0.172.31.208.6" # 10b, 3 layers
    CUDA_FILE = "tp2_pp1_dp4_mbs2_numbs2_gbs16_gpus8.log.0.172.31.208.6" # 10b, 6 layers
    # MUSA_FILE = "2024-05-21_16:45:41/tp1_pp1_dp4_mbs1_numbs1_gbs4_gpus4.log.0.10.70.147.165"  # done
    # MUSA_FILE = "2024-05-23_15:48:09/tp1_pp1_dp2_mbs1_numbs1_gbs2_gpus2.log.0.10.70.147.165"  # 10b, 3 layers
    MUSA_FILE = "tp2_pp1_dp4_mbs2_numbs2_gbs16_gpus8.log.0.10.70.147.226" # 10b, 6 layers

    cuda_info = _load_local_file(BASE_DIR_CUDA, CUDA_FILE)
    # print(f"cuda iterations: {len(cuda_info[1])}")
    musa_info = _load_local_file(BASE_DIR_MUSA, MUSA_FILE)
    # print(f"musa iterations: {len(musa_info[1])}")
    stat(cuda_info, musa_info)