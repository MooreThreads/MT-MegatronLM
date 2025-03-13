#!/bin/bash

set -u
  WORK_HOME=$1
  PATCH_HOME=$2
  EXPNAME=$3
  HOSTFILE=$4
  DATA_DIR=$5
  TP_SIZE=$6
  PP_SIZE=$7
  MICRO_BATCH_SIZE=$8
  GLOBAL_BATCH_SIZE=$9
  TOKENIZED_MODEL=${10}
set +u
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export NCCL_PROTOS=2
export NCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${PATCH_HOME}/Megatron-LM-240928
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH
# export MUSA_LAUNCH_BLOCKING=1

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



export NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
# export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks[$NODE_ADDR];}' $HOSTFILE)
export MASTER_PORT=12389


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT 
)

MODEL_ARGS=(
    --num-layers 8
    --hidden-size 768 
    --num-attention-heads 12 
    --seq-length 4096 
    --max-position-embeddings 4096 
    --norm-epsilon 1e-5 
    --init-method-std 0.01 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --position-embedding-type rope 
    --no-position-embedding 
    --swiglu 
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
)

# 244140625 1T
TRAINING_ARGS=(
    --seed 42 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    --train-samples 24414062 
    --init-method-std 0.0165 
    --use-mcore-models 
    --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --use-distributed-optimizer 
    --use-flash-attn 
    --sequence-parallel 
    --recompute-granularity full 
    --recompute-method block 
    --recompute-num-layers 0 
    --distributed-backend nccl 
    --transformer-impl transformer_engine
)
# --transformer-impl local transformer_engine
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

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type=SentencePieceTokenizer \
    --tokenizer-model ${TOKENIZED_MODEL} \
    --split 1
"

# DATA_ARGS=(
#     --data-path $DATA_PATH 
#     --vocab-file $VOCAB_FILE 
#     --merge-file $MERGE_FILE 
#     --split 949,50,1
# )



EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 200000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
)

# if [ -n "${WANDB_API_KEY}" ]; then
#     EVAL_AND_LOGGING_ARGS+=(
#         --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
#         --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
#     )
# fi

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]}
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]}
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]}
    "
echo $cmd
eval $cmd
