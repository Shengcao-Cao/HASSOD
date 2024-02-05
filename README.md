# HASSOD: Hierarchical Adaptive Self-Supervised Object Detection

This is the official PyTorch implementation of our *NeurIPS 2023* paper:

**HASSOD: Hierarchical Adaptive Self-Supervised Object Detection**

[[Project Page]](https://hassod-neurips23.github.io/) [[Paper-OpenReview]](https://openreview.net/pdf?id=sqkGJjIRfG) [[Video-YouTube]](https://www.youtube.com/watch?v=s8u7tEKg5ew) [[Video-Bilibili]](https://www.bilibili.com/video/BV1pg4y1Z7CK)

[Shengcao Cao](https://shengcao-cao.github.io/), [Dhiraj Joshi](https://research.ibm.com/people/dhiraj-joshi), [Liang-Yan Gui](https://cs.illinois.edu/about/people/faculty/lgui), [Yu-Xiong Wang](https://yxw.web.illinois.edu/)

## 🔎 Overview

![HASSOD-gif](assets/HASSOD.gif)

HASSOD is a fully self-supervised approach for object detection and instance segmentation, demonstrating a significant improvement over the previous state-of-the-art methods by discovering a more comprehensive range of objects. Moreover, HASSOD understands the part-to-whole object composition like humans do, while previous methods cannot. Notably, we improve class-agnostic Mask AR from 20.2 to 22.5 on LVIS, and from 17.0 to 26.0 on SA-1B. 

## 🛠️ Instructions
To use our code and reproduce the results, please follow these detailed documents step by step:
- [Preparation](https://github.com/Shengcao-Cao/HASSOD/blob/main/preparation.md): Prepare the environment, data, and pre-trained models
- [Reproduction](https://github.com/Shengcao-Cao/HASSOD/blob/main/reproduction.md) Produce pseudo-labels and train the object detector (download links included for our pseudo-labels and model)

## 🙏 Acknowledgements
Our code is developed based on the following repositories:
- [CutLER](https://github.com/facebookresearch/CutLER)
- [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher)
- [Detectron2](https://github.com/facebookresearch/detectron2)

We greatly appreciate their open-source work!

## ⚖️ License
This project is released under the Apache 2.0 license. Other codes from open source repository follows the original distributive licenses.

## 🌟 Citation
If you find our research interesting or use our code, data, or model in your research, please consider citing our work.
```
@inproceedings{cao2023hassod,
    title={{HASSOD}: Hierarchical Adaptive Self-Supervised Object Detection},
    author={Cao, Shengcao and Joshi, Dhiraj and Gui, Liangyan and Wang, Yu-Xiong},
    booktitle={NeurIPS},
    year={2023}
}
```
