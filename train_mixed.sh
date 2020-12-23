#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 16 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
# python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 32 --lr 0.00002 --image_size 400 720 --wdecay 0.0001 --mixed_precision
