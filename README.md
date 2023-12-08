# FFB6D_MHSA

This is the source code for my work on 6D object pose estimation during my graduate studies.



<div align=center><img width="100%" src="figs/network overview.jpg"/></div>

We follow [FFB6D](https://arxiv.org/abs/2103.02242v1) as the base framework, differently, we utilize [BoTNet](https://arxiv.org/abs/2101.11605) instead of [ResNet](https://arxiv.org/abs/1512.03385) as the feature extractor of RGB images. The design of BoTNet-50 is simple: replace the final three spatial (3×3) convolutions in ResNet50 with Multi-Head Self-Attention (MHSA) layers that
implement global self-attention over a 2D featuremap. This allows us to obtain abstract and low resolution featuremaps from large images through convolutions, which are subsequently processed and aggregated using global self-attention.
