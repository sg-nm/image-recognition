#!/bin/bash

ID=`date "+%m%d_%H%M_%S"`

export NCCL_SOCKET_IFNAME=bond0
export NCCL_P2P_LEVEL=1

NUM_NODES=1
WORKERS_PER_NODE=4

torchrun \
--standalone \
--nnodes=$NUM_NODES \
--nproc_per_node=$WORKERS_PER_NODE \
train_clip.py \
--cfg ./configs/vit_s.yml \
--output_dir "logs/log-"$ID \

# python train_clip.py \
# --cfg ./configs/vit_s.yml \
# --output_dir "logs/log-"$ID \