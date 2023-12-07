#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar  # checkpoint to test.
python3 -m torch.distributed.launch  --master_port 9999 --nproc_per_node=1 train_ycb.py --gpu '5' -eval_net -checkpoint $tst_mdl -test -test_pose # -debug