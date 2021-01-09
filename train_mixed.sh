#!/bin/bash
mkdir -p checkpoints
# python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision 
# python -u train.py --name raft-things --stage things --validation sintel --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
# python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.00001 --mixed_precision 
# python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 16 --lr 0.0001 --image_size 400 720 --wdecay 0.00001 --mixed_precision 
# python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 3 --lr 0.0005 --image_size 400 720 --wdecay 0.00001 --mixed_precision 
python -u train.py --name raft-things --stage things --validation things --gpus 0 --num_steps 1200000 --batch_size 3 --lr 0.0001 --image_size 400 720 --wdecay 0.00001 --mixed_precision 
