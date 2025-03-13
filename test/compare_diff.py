#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import paramiko
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


def _process_logs(log_str):
    loss_dict = OrderedDict()
    lr_dict = OrderedDict()
    logs = log_str.strip().split('\n')
    for num, log in enumerate(logs):
        log_split = log.split('|')
        if len(log_split) != LOG_LEN:
            continue
        index = log_split[0].split()[3].split('/')[0]
        loss = float(log_split[5].split(":")[1])
        lr = float(log_split[3].split(":")[1].strip())
        loss_dict[index] = loss
        lr_dict[index] = lr
    return loss_dict, lr_dict


def _load_remote_file(host, user, port=None, file_path=""):
    # prepare key file
    private_key_file = os.path.expanduser('~/.ssh/id_rsa')
    mykey = paramiko.RSAKey.from_private_key_file(private_key_file)
    # setting client
    s = paramiko.SSHClient()
    s.load_system_host_keys()
    s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # connect
    port = port if port else 22
    s.connect(host, port, user, pkey=mykey, timeout=200)
    # read remote file
    cmd = f'cat {file_path}'
    stdin, stdout, stderr = s.exec_command(cmd)

    output = stdout.read().decode()
    s.close()
    loss, lr = _process_logs(output)
    return loss, lr


def load_cuda_file(file_name):
    host_cuda = "172.31.208.6"
    user_cuda = "fan.mo"
    file_path_cuda = os.path.join(BASE_DIR_CUDA, file_name)
    cuda_loss, cuda_lr = _load_remote_file(host_cuda, user_cuda, file_path=file_path_cuda)
    return cuda_loss, cuda_lr


def load_musa_file(file_name):
    host_musa = "10.1.0.27"
    user_musa = "root"
    file_path_musa = os.path.join(BASE_DIR_MUSA, file_name)
    musa_loss, musa_lr = _load_remote_file(host_musa, user_musa, port=31968, file_path=file_path_musa)
    return musa_loss, musa_lr


def stat(cuda_info, musa_info):
    cuda, cuda_lr = cuda_info
    musa, musa_lr = musa_info
    c_ks = len(cuda.keys())
    m_ks = len(musa.keys())
    ks = cuda.keys() if c_ks < m_ks else musa.keys()
    x_axis, c_axis, m_axis, diff_ratios = [], [], [], []
    c_lr, m_lr = [], []

    # ==== stat ====
    for i, k in enumerate(ks):
        if STEPS > 1000 and i >= STEPS:
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
    BASE_DIR_CUDA = "/nfs2/fan.mo/workspace/megatron-lm-musa-patch/examples/mixtral/output"
    BASE_DIR_MUSA = "/home/dist/fan.mo/megatron-lm-musa-patch/examples/mixtral/output"
    STEPS = int(os.getenv("STEPS", 3000))

    # ======== Mixtral ==========
    LOG_LEN = 14
    # CUDA_FILE = "2024-05-17_04:00:23/tp1_pp1_dp4_mbs1_numbs1_gbs4_gpus4.log.0.172.31.208.8"
    # CUDA_FILE = "2024-05-23_03:37:40/tp1_pp1_dp2_mbs1_numbs1_gbs2_gpus2.log.0.172.31.208.6" # 10b, 3 layers
    CUDA_FILE = "2024-05-23_10:40:16/tp1_pp1_dp2_mbs1_numbs1_gbs2_gpus2.log.0.172.31.208.6" # 10b, 6 layers
    # MUSA_FILE = "2024-05-21_16:45:41/tp1_pp1_dp4_mbs1_numbs1_gbs4_gpus4.log.0.10.70.147.165"  # done
    # MUSA_FILE = "2024-05-23_15:48:09/tp1_pp1_dp2_mbs1_numbs1_gbs2_gpus2.log.0.10.70.147.165"  # 10b, 3 layers
    MUSA_FILE = "2024-05-24_11:29:07/tp1_pp1_dp2_mbs1_numbs1_gbs2_gpus2.log.0.10.70.147.165" # 10b, 6 layers

    cuda_info = load_cuda_file(CUDA_FILE)
    print(f"cuda iterations: {len(cuda_info[1])}")
    musa_info = load_musa_file(MUSA_FILE)
    print(f"musa iterations: {len(musa_info[1])}")
    stat(cuda_info, musa_info)