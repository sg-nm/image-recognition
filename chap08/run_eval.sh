#!/bin/bash

python evaluate.py \
--cfg ./configs/vit_s.yml \
--resume "./trained_model/epoch_latest.pt"