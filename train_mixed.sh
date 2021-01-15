#!/bin/bash
mkdir -p checkpoints
# python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/20000_raft-things.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.00001 --image_size 320 768 --wdecay 0.0001 --mixed_precision
# python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/2500_raft-things.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.000001 --image_size 320 768 --wdecay 0.0001 --mixed_precision
python -u train.py --name raft-things --stage things --validation sintel --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.00001 --image_size 320 768 --wdecay 0.0001 --mixed_precision
