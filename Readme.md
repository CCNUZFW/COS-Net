# COS-Net: Closed and Open-Set Mobile Recording Device Identification via Dual Channel Deep Representation Learning

## Introduction 
This is an implementation of the COS-Net model referring to the following paper submitted to ESWA: **COS-Net: Closed and Open-Set Mobile Recording Device Identification via Dual Channel Deep Representation Learning**

## Contributors
1. Chunyan Zeng : cyzeng@hbut.edu.cn
2. Yuandan Feng : 102400218@hbut.edu.cn 
3. Zhifeng Wang : zfwang@ccnu.edu.cn

## Environment Requirement
python == 3.10.16
torch == 2.1.1+cu118
numpy == 1.26.4
libblas == 3.9.0

## Datasets
1. The training feature file `train.csv` is placed in the `./data` directory, containing multiple MFCC features, five-level DWT approximation coefficients, and the corresponding device labels.

2. The testing feature file `test.csv` is placed in the `./data` directory, containing multiple MFCC features, five-level DWT approximation coefficients, and the corresponding device labels.

## Train and test

 Run the `COS-Net.py` to train and test.

The training and testing feature files are loaded from `./data/train.csv` and `./data/test.csv`, respectively.

