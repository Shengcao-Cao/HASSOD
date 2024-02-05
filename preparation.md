## Preparation
This document details the necessary preparation steps including setting up the environment as well as downloading the data and pre-trained checkpoints.

### Environment
We recommend setting up the environment with [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and include the following packages:
- Python 3.8
- [PyTorch](https://pytorch.org/) 1.10.0
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [DenseCRF](https://github.com/lucasb-eyer/pydensecrf)
- Other minor dependencies
  - scikit-image 0.19.2
  - scikit-learn 1.1.1
  - setuptools 59.5.0
  - opencv-python 4.6.0.66

The versions that are used in HASSOD are listed above, but newer versions should work too. You may follow this example to set up the environment:
```
# Create and activate the environment
conda create -n hassod python=3.8
conda activate hassod

# Install PyTorch (suppose you have CUDA 11.3)
# If your CUDA version is different, check https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install Detectron2 (in your working directory, e.g., HASSOD/third_party)
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2

# Install DenseCRF (in your working directory, e.g., HASSOD/third_party)
git clone https://github.com/lucasb-eyer/pydensecrf.git
pip install -e pydensecrf

# Install other dependencies
pip install scikit-image==0.19.2 scikit-learn==1.1.1
pip install setuptools==59.5.0
pip install opencv-python==4.6.0.66
```

### Datasets

#### Training Data
The training of HASSOD is performed on [MS-COCO](https://cocodataset.org/). We need to first download the images from both the `train` split and `unlabeled` split:
- http://images.cocodataset.org/zips/train2017.zip
- http://images.cocodataset.org/zips/unlabeled2017.zip
- No need to download the training annotations since HASSOD is fully self-supervised.

Then unzip the compressed files and put the images into one folder `train+unlabeled2017`. Also create a folder `annotations/hassod` in which we will automatically generate (or download) our pseudo-labels later. The organized structure of the data should look like:
```
HASSOD
|-- datasets
    |-- coco
        |-- train+unlabeled2017
        |   |-- 000000000008.jpg
        |   |-- 000000000009.jpg
        |   |-- 000000000013.jpg
        |   |-- 000000000022.jpg
        |   |-- ...
        |
        |-- annotations
            |-- hassod
```

To make the training data visible to the detector training code (by Detectron2), create a soft link inside `detector_training`:
```
# Suppose you are now in the HASSOD directory
cd detector_training
ln -s ../datasets .
```

#### Evaluation Data
We evaluate self-supervised object detectors in a class-agnostic manner. Therefore, we need to convert the original annotation files and consider all object instances as one "object" class. We mainly follow the [previous practice of CutLER](https://github.com/facebookresearch/CutLER/blob/main/datasets/README.md).

The dataset paths are defined in `detector_training/data/datasets/builtin.py`. You may follow these examples to evaluate on other customized datasets.

##### MS-COCO, LVIS, and Objects365
Please check the data preparation steps.

##### SA1B
SA1B contains 11M images randomly split into 1,000 packs. We use the images from the first pack (with index `sa_000000.tar`) to evaluate object detectors. Download this image pack and uncompress it. Then download our pre-processed annotation file [here](). The folder structure should look like:
```
HASSOD
|-- datasets
    |-- SA1B
        |-- images
        |   |-- sa_1.jpg
        |   |-- sa_2.jpg
        |   |-- sa_3.jpg
        |   |-- ...
        |
        |-- annotations
            |-- sa1b_p0_cls_agnostic.json
```

##### MS-COCO + LVIS
In the ablation study, we evaluate various models against annotations of MS-COCO and LVIS on the `val2017` split, because they are complementary to each other. LVIS uses the same images as MS-COCO, but instead labels more classes of objects non-exhaustively. After combining the two sets of annotations and removing duplicates, there are about 20 annotations per image. In the detector training example we provide, we also evaluate on this "MS-COCO + LVIS" dataset. Download the processed annotation file [here]() and move it into `annotations/hassod`:
```
HASSOD
|-- datasets
    |-- coco
        |-- val2017
        |-- annotations
            |-- hassod
                |-- coco+lvis_cls_agnostic_instances_val2017.json
```

### DINO Checkpoints
HASSOD requires [DINO](https://github.com/facebookresearch/dino) pre-trained model checkpoints. Download the ViT-B/8 and ResNet-50 backbones:
- https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
- http://dl.fbaipublicfiles.com/cutler/checkpoints/dino_RN50_pretrain_d2_format.pkl

Put the checkpoints in a folder `checkpoints` like this:
```
HASSOD
|-- checkpoints
    |-- dino_vitbase8_pretrain.pth
    |-- dino_RN50_pretrain_d2_format.pkl
```
