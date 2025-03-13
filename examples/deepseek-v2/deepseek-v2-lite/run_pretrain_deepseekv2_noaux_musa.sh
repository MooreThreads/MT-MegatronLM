#!/bin/bash

# Runs the "175B" parameter model

# Please change the following envrioment variables
# base on the cluster configuration
set -u
  WORK_HOME=$1
  PATCH_HOME=$2
  EXPNAME=$3
  HOSTFILE=$4
  DATA_DIR=$5
  TP_SIZE=$6
  PP_SIZE=$7
  EP_SIZE=$8
  MICRO_BATCH_SIZE=$9
  GLOBAL_BATCH_SIZE=${10}
  TOKENIZED_MODEL=${11}
  RDZV_ID=${12}
set +u
export ENABLE_PROFILER=0
# export PROFILER_FREQ=4
# export MUSA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20480000

MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=$WORK_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
DATA_PATH=$DATA_DIR


LOG_PATH=$WORK_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$WORK_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$WORK_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH


export NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1) # tail for cuda/ head for musa
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk -v node_addr="$NODE_ADDR" '{ranks[$1]=(FNR-1);} END {print ranks[node_addr];}' $HOSTFILE)
export MASTER_PORT=12356
# export MUSA_LAUNCH_BLOCKING=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --log_dir $WORK_HOME/output_log/$RDZV_ID/$EXPNAME
    --redirects 3
)

MODEL_ARGS=(
    --num-layers 2  # 60 
    --hidden-size 5120
    --num-attention-heads 128
    --seq-length 4096 
    --max-position-embeddings 4096 
    --norm-epsilon 1e-6 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --vocab-size 102400 #102400
    --ffn-hidden-size 1536  # 12288
    --position-embedding-type rope 
    --no-position-embedding 
    --swiglu 
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
)

# 24414062 1T
TRAINING_ARGS=(
    --seed 42 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    --train-samples 24414062 
    --init-method-std  0.006 # 0.02 in HF config, but 0.006 in the paper 
    --use-mcore-models 
    --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --use-distributed-optimizer 
    --use-flash-attn 
    --sequence-parallel 
    --recompute-granularity full 
    --recompute-method block 
    --recompute-num-layers 1
    --distributed-backend nccl
    --multi-latent-attention
    --qk-layernorm
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 1
)
REGULARIZATION_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
)

WARMUP_STEPS=2000
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))

LEARNING_RATE_ARGS=(
    --lr 1.5e-5 
    --lr-decay-style cosine 
    --lr-warmup-samples ${WARMUP_SAMPLES} 
    --min-lr 1.5e-6 
    --initial-loss-scale 65536 
    --min-loss-scale 1.0 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE  
	--pipeline-model-parallel-size $PP_SIZE 
)

MIXED_PRECISION_ARGS=(
    --bf16 
    --attention-softmax-in-fp32 
    --no-masked-softmax-fusion 
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type NullTokenizer
    # --tokenizer-model ${TOKENIZED_MODEL}
    --split 1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 100000 
    --eval-interval 1 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
)

MOE_ARGS=(
    --num-experts 160
    --expert-model-parallel-size $EP_SIZE
    --moe-token-dispatcher-type alltoall
    --moe-router-topk-limited-devices 4
    --moe-router-num-node-group 8
    --moe-noaux-gamma 1e-3
    --moe-router-load-balancing-type seq_aux_loss
    --moe-complementary-seq-aux-loss
    --moe-router-topk 6
    --moe-router-pre-softmax #deepseek use pre-softmax
    --moe-router-use-sigmoid #deepseek use sigmoid
    --moe-router-norm-topk-prob #norm topk prob with sigmoid
    --moe-router-topk-scaling-factor 2.5 # pre-softmax need scaling
    --moe-aux-loss-coeff 3e-3
    --moe-ffn-hidden-size 1536
    --moe-shared-expert-intermediate-size 3072
    # --moe-layer-freq "([0]*1+[1]*1)*1"
    # --moe-grouped-gemm
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_deepseekv2.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]}
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]}
        ${DATA_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]}
    "
echo $cmd
$cmd
