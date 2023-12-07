#!/bin/bash
n_gpu=6
cls='iron'
checkpoint=train_log/linemod/checkpoints/${cls}/FFB6D_${cls}.pth.tar
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls 