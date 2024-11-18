

# DNA Fragmentation Estimation using Lightweight Deep Learning Model

This project implements a lightweight deep learning approach for estimating the DNA fragmentation index (DFI) from sperm cell images. The methodology leverages a computationally efficient Convolutional Neural Network (CNN) for regression tasks, optimized for edge devices through model quantization.

---

## Table of Contents
- [DNA Fragmentation Estimation using Lightweight Deep Learning Model](#dna-fragmentation-estimation-using-lightweight-deep-learning-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Findings](#key-findings)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Preparing the Data](#1-preparing-the-data)
    - [2. Training the Model](#2-training-the-model)
  - [Components](#components)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Architecture](#model-architecture)
    - [Augmentation Strategies](#augmentation-strategies)
    - [Training Pipeline](#training-pipeline)
  - [Evaluation](#evaluation)
    - [Metrics](#metrics)
    - [Results](#results)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

---

## Overview

Accurate DNA fragmentation estimation is critical for improving assisted reproductive technology (ART) outcomes. This project:
- Proposes a lightweight CNN-based approach for donor-independent DFI prediction.
- Uses brightfield microscopy images to preserve sperm cell viability during testing.
- Introduces model quantization to optimize performance on edge devices while maintaining accuracy.

### Key Findings
- Achieved a Pearson correlation coefficient of **0.61** for donor-independent experiments.
- EfficientNet-B0 demonstrated the best balance of performance and computational efficiency.
- Quantized models reduce model size by ~70-80%, enabling edge deployment without significant performance loss.

---

## Features

- **Lightweight Model**: EfficientNet-B0 as the backbone for regression tasks.
- **Quantization**: 8-bit quantization for optimized edge device inference.
- **Donor-Independent Evaluation**: Ensures generalizability across unseen donor data.
- **Augmentations**: Extensive data augmentations to improve robustness.
- **Open Deployment**: ONNX runtime compatibility for flexible deployment.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pyther-hub/dfi-prediction.git
   cd dna-fragmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Preparing the Data
- Download the dataset from [McCallum et al.'s public dataset](https://figshare.com/articles/dataset/Deep_learning-based_selection_of_human_sperm_with_high_DNA_integrity/8124932).
- Structure the dataset as follows:
  ```
  dataset/
  ├── annotations.csv  # Contains columns: 'file_name', 'label', 'donnor'
  ├── images/
      ├── img1.jpg
      ├── img2.jpg
      ...
  ```

### 2. Training the Model
To train and evaluate the model:
```python
from training import RunExperiment

RunExperiment(
    set_numb=1,          # Donor ID for validation split
    model_name="efficientnet_b0",
    numb_epochs=100,
    lr=5e-4,
    do_log=True,
    aug_level=2
)
```

---

## Components

### Dataset Preparation
The dataset class (`CustomImageDataset`) loads images and labels from the annotations file, applying transformations defined for training or testing:
```python
from data_preparation import CustomImageDataset, get_augmentations

train_transform = get_augmentations('train', level_int=2)
dataset = CustomImageDataset(img_dir="dataset/images", df=pd.read_csv("annotations.csv"), transform=train_transform)
```

### Model Architecture
A lightweight CNN regression model, with EfficientNet-B0 as the backbone, predicts DFI values. It uses a custom regression head for feature extraction:
```python
from model_definitions import BaseModel

model = BaseModel(model_name="efficientnet_b0", dropout_prob=0.2)
```

### Augmentation Strategies
Three levels of augmentation ensure robustness:
- **Basic**: Resize, flips, and rotations.
- **Moderate**: Adds color jitter and affine transformations.
- **Advanced**: Includes additional augmentations for extensive variability.

### Training Pipeline
The `train_model` function manages training, validation, and logging:
- Loss: Mean Squared Error (MSE).
- Optimizer: Adam.
- Validation Metric: Pearson correlation coefficient.

---

## Evaluation

### Metrics
The evaluation focuses on the Pearson correlation coefficient to assess the linear relationship between predicted and actual DFI values.

### Results
- **Full Precision (FP32)**: EfficientNet-B0 achieved an average Pearson correlation of **0.631**.
- **Quantized (INT8)**: Reduced model size by ~70-80% with minimal performance loss (average correlation: **0.621**).

---

## Acknowledgments

This implementation is based on the methodology described in:
> Sudhanshu Rai, Samir Malakar, Dilip K. Prasad, **"DNA Fragmentation Estimation using Light-weight Deep Learning Model."**

Supported by:
- Research Council of Norway (nanoAI, Project ID: 325741)
- H2020 (OrganVision, Project ID: 964800)
- HORIZON-ERC-POC (Spermotile, Project ID: 101123485)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
