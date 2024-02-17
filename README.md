# Spatial-Contextual Discrepancy Information Compensation for GAN Inversion (AAAI 2024)
<a href="https://arxiv.org/abs/2312.07079"><img src="https://img.shields.io/badge/arXiv-2312.07079-b31b1b.svg"></a>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OnU0Ox0kDV_h_qfqu5gHn0FWlwGpvQ8t?usp=sharing)

> Ziqiang Zhang, Yan Yan*, Jing-Hao Xue, Hanzi Wang
> 
> Most existing GAN inversion methods either achieve accurate reconstruction but lack editability or offer strong editability at the cost of fidelity. Hence, how to balance the distortion editability trade-off is a significant challenge for GAN inversion. To address this challenge, we introduce a novel spatial-contextual discrepancy information compensation-based GAN-inversion method (SDIC), which consists of a discrepancy information prediction network (DIPN) and a discrepancy information compensation network (DICN). SDIC follows a “compensate-and-edit” paradigm and successfully bridges the gap in image details between the original image and the reconstructed/edited image. On the one hand, DIPN encodes the multi-level spatial-contextual information of the original and initial reconstructed images and then predicts a spatial-contextual guided discrepancy map with two hourglass modules. In this way, a reliable discrepancy map that models the contextual relationship and captures fine-grained image details is learned. On the other hand, DICN incorporates the predicted discrepancy information into both the latent code and the GAN generator with different transformations, generating high-quality reconstructed/edited images. This effectively compensates for the loss of image details during GAN inversion. Both quantitative and qualitative experiments demonstrate that our proposed method achieves the excellent distortion-editability trade-off at a fast inference speed for both image inversion and editing tasks.

<img src="docs/age+/ori.png" width="200px"/>        <img src="docs/age-/ori.jpg" width="200px"/>  <img src="docs/pose/ori.jpg" width="200px"/>    <img src="docs/smile/ori.jpg" width="200px"/> 

<img src="docs/age+/age+.gif" width="200px"/>        <img src="docs/age-/age-.gif" width="200px"/>  <img src="docs/pose/pose.gif" width="200px"/>    <img src="docs/smile/smile.gif" width="200px"/> 


## Pipeline
