#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

TP_SIZE=1
PP_SIZE=2
WORLD_SIZE=8
MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=128
(( DP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE) ))
echo $DP_SIZE
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))
echo $GLOBAL_BATCH_SIZE

set -u
  WORK_HOME="$PWD"
  PATCH_HOME="$PWD"/../..
  EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
  DATA_PATH=/home/data/yehua/llama3/llama3_dataset/dedup-md5-pile-pile-cc_text_document
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log
  TOKENIZED_MODEL=/home/data/yehua/llama3/llama3_tokenizer
  SCRIPT_FILE=./8B/run_pretrain_llama3_musa.sh
  RDZV_ID=$CURRENT_TIME
set +u

cmd="bash -c 'cd $WORK_HOME; \
     bash $SCRIPT_FILE $WORK_HOME $PATCH_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE $TOKENIZED_MODEL $RDZV_ID"

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  # cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  cmd_ssh=$cmd" &'"
  # cmd_ssh=$cmd" '"
  echo $cmd_ssh
  ssh -f -n $host $cmd_ssh
  # echo $host, "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done
