# FFB6D_MHSA

This is the source code for my work on 6D object pose estimation during my graduate studies.



<div align=center><img width="100%" src="figs/network overview.jpg"/></div>

We follow [FFB6D](https://arxiv.org/abs/2103.02242v1) as the base framework, differently, we utilize [BoTNet](https://arxiv.org/abs/2101.11605) instead of [ResNet](https://arxiv.org/abs/1512.03385) as the feature extractor of RGB images. The design of BoTNet-50 is simple: replace the final three spatial (3×3) convolutions in ResNet50 with Multi-Head Self-Attention (MHSA) layers that
implement global self-attention over a 2D featuremap. This allows us to obtain abstract and low resolution featuremaps from large images through convolutions, which are subsequently processed and aggregated using global self-attention.


## Installation
- Install CUDA
- Install the required packages：
  ```shell
  pip3 install -r requirement.txt 
  ```
- Install [apex](https://github.com/NVIDIA/apex)(Optional):
  ```shell
  git clone https://github.com/NVIDIA/apex
  cd apex
  export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # set the target architecture manually, suggested in issue https://github.com/NVIDIA/apex/issues/605#issuecomment-554453001
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  cd ..
  ```
- Install [normalSpeed](https://github.com/hfutcgncas/normalSpeed), a fast and light-weight normal map estimator:
  ```shell
  git clone https://github.com/hfutcgncas/normalSpeed.git
  cd normalSpeed/normalSpeed
  python3 setup.py install --user
  cd ..
  ```
- Compile [RandLA-Net](https://github.com/qiqihaer/RandLA-Net-pytorch) operators:
  ```shell
  cd ffb6d/models/RandLA/
  sh compile_op.sh
  ```

## Datasets

- **YCB-Video:** Download the YCB-Video Dataset from [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/). Unzip it and link the unzipped```YCB_Video_Dataset``` to ```ffb6d/datasets/ycb/YCB_Video_Dataset```:

  ```shell
  ln -s path_to_unzipped_YCB_Video_Dataset ffb6d/datasets/ycb/
  ```
## Training and evaluating
### Training on the YCB-Video Dataset
- Start training on the YCB-Video Dataset by:
  ```shell
  # commands in train_ycb.sh
  n_gpu=8  # number of gpu to use
  python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu
  ```
  The trained model checkpoints are stored in ``train_log/ycb/checkpoints/``
### Evaluating on the YCB-Video Dataset
- Start evaluating by:
  ```shell
  # commands in test_ycb.sh
  tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar  # checkpoint to test.
  python3 -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpu '0' -eval_net -checkpoint $tst_mdl -test -test_pose # -debug
  ```
  You can evaluate different checkpoints by revising the ``tst_mdl`` to the path of your target model.
### Demo/visualization on the YCB-Video Dataset
- After training your model or downloading the pre-trained model, you can start the demo by:
  ```shell
  # commands in demo_ycb.sh
  tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar
  python3 -m demo -checkpoint $tst_mdl -dataset ycb
  ```
  The visualization results will be stored in ```train_log/ycb/eval_results/pose_vis```.
## Results
- Evaluation result without any post refinement on the YCB-Video dataset:
  
  <table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Epochs </th>
      <th class="tg-c3ow" colspan="2" style="text-align: center">FFB6D</th>
      <th class="tg-c3ow" colspan="2" style="text-align: center">FFB6D_MHSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">ADDS</td>
      <td class="tg-0pky">ADD(S)</td>
      <td class="tg-0pky">ADDS</td>
      <td class="tg-0pky">ADD(S)</td>
    </tr>
    <tr>
      <td class="tg-0pky">20</td>
      <td class="tg-0pky">89.4</td>
      <td class="tg-0pky">79.1</td>
      <td class="tg-fymr" style="font-weight:bold">91.9</td>
      <td class="tg-fymr" style="font-weight:bold">84.3</td>
    </tr>
      <tr>
      <td class="tg-0pky">40</td>
      <td class="tg-0pky">91.0</td>
      <td class="tg-0pky">83.2</td>
      <td class="tg-fymr" style="font-weight:bold">93.8</td>
      <td class="tg-fymr" style="font-weight:bold">87.8</td>
    </tr>
    <tr>
      <td class="tg-0pky">60</td>
      <td class="tg-0pky">93.1</td>
      <td class="tg-0pky">86.4</td>
      <td class="tg-fymr" style="font-weight:bold">94.4</td>
      <td class="tg-fymr" style="font-weight:bold">89.1</td>
    </tr>
  </tbody>
  </table>
