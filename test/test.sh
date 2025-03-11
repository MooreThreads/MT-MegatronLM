export NCCL_PROTOS=2
export CUDA_VISIBLE_DEVICES='4,5,6,7'

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=10489 test_dist.py
