#!/bin/bash
#This can be used for re-generation
EXP_DIR=./checkpoints
EXP_NAME=original-fp16-code-3d-1024
PIPELINE_MP_SIZE=2


STOP_STEPS=20000
TRAINING_STEPS=20000
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

NUM_EXPERTS=4
if [ -n "${3}" ]; then
    NUM_EXPERTS=$3;
fi

CAPACITY_FACTOR=2
if [ -n "${4}" ]; then
    CAPACITY_FACTOR=$4;
fi

TOP_K=1
if [ -n "${5}" ]; then
    TOP_K=$5;
fi

LOSS_WEIGHT=0.1
if [ -n "${6}" ]; then
    LOSS_WEIGHT=$6;
fi

BATCH_SIZE=32
if [ -n "${7}" ]; then
    BATCH_SIZE=$7;
fi

SEQ=1024
FLASH="--use-flash-attn  --ampipe --pipe-degree=8" 
GPUS=8

##
### Pre-training for MoE 46M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-capacity-factor=${CAPACITY_FACTOR} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-top-k=${TOP_K} \
--moe-expert-model-parallelism" 

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node $GPUS \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length ${SEQ} \
--max-position-embeddings ${SEQ} "

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size $(($BATCH_SIZE*4)) \
--train-iters ${STOP_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

DATASET=dataset/wikipedia_text_document   

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${DATASET} \
--vocab-file ./gpt2-vocab.json \
--merge-file ./gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--fp16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--no-async-tensor-model-parallel-allreduce \
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size $PIPELINE_MP_SIZE \
--fp16-lm-cross-entropy \
${FLASH}" 

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

cp ${0} ./${EXP_DIR}/script-${EXP_NAME}.sh
rm ./${EXP_DIR}/train-${EXP_NAME}.log
python -m torch.distributed.launch ${DISTRIBUTED_ARGUMENTS} \
       third_party/Megatron-LM/pretrain_gpt.py \
       ${MOE_ARGUMENTS} \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train-${EXP_NAME}.log

