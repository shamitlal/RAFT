#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-things --stage thingstraj --validation sintel --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
