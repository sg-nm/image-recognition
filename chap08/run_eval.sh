#!/bin/bash

ID=`date "+%m%d_%H%M_%S"`

export NCCL_SOCKET_IFNAME=bond0
export NCCL_P2P_LEVEL=1

python evaluate.py \
--cfg ./configs/vit_s.yml \
--resume "./trained_model/epoch_latest.pt"