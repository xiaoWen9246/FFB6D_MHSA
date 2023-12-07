#!/bin/bash
n_gpu=6  # number of gpu to use
tst_mdl=train_log/ycb/checkpoints/FFB6D.pth.tar  # checkpoint to train
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port 47769 train_ycb.py --gpus=$n_gpu -checkpoint=$tst_mdl