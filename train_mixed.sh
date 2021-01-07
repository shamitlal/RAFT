#!/bin/bash
mkdir -p checkpoints
# python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 16 --lr 0.0001 --image_size 400 720 --wdecay 0.00001 --mixed_precision --sample_depth --use_scaler # works best for 100 scenes
python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 16 --lr 0.0001 --image_size 400 720 --wdecay 0.00001 --mixed_precision --sample_depth --use_scaler
# python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 1 --lr 0.0005 --image_size 400 720 --wdecay 0.00001 --mixed_precision --sample_depth --use_scaler
