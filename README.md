# Targeted Data Augmentations for Bias Mitigation
🔗 Paper: [Targeted Data Augmentation for bias mitigation, Agnieszka Mikołajczyk-Bareła, Maria Ferlin, Michał Grochowski](https://arxiv.org/abs/2308.11386)

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This repository contains the code and resources for the paper:

**"Targeted Data Augmentation for Improving Model Robustness"**

The project introduces a novel method called **Targeted Data Augmentation (TDA)** for mitigating biases in machine learning models, particularly in deep learning for computer vision tasks. The code includes implementations of TDA and the **Counterfactual Bias Insertion (CBI)** method used for bias identification and evaluation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
- [Results](#results)
- [Reproducing Experiments](#reproducing-experiments)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Implementation of Targeted Data Augmentation (TDA) for bias mitigation.
- Implementation of Counterfactual Bias Insertion (CBI) for bias evaluation.
- Scripts for training and evaluating models on:
  - **ISIC-2020 Skin Lesion Dataset**
  - **Gender Classification Dataset**
- Support for multiple neural network architectures:
  - DenseNet121
  - EfficientNet-B2
  - Vision Transformer (ViT)

## Project Structure

```
├── data/
│   ├── isic_2020/
│   ├── gender_classification/
│   ├── masks/
│       ├── rulers/
│       ├── frames/
│       └── glasses/
├── models/
│   ├── densenet121.py
│   ├── efficientnet_b2.py
│   └── vit.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── augmentations.py
│   └── utils.py
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

## Datasets

### ISIC-2020 Skin Lesion Dataset

- **Description**: A dataset containing dermoscopic images for skin lesion classification.
- **Access**: [Kaggle SIIM-ISIC Melanoma Classification Challenge](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)

### Gender Classification Dataset

- **Description**: A dataset of cropped face images labeled by gender.
- **Access**: [Kaggle Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)

### Masks for Augmentation

- **Ruler Masks**: Used for skin lesion images.
- **Frame Masks**: Used for skin lesion images.
- **Glasses Masks**: Used for face images.

- **Access**: [Data Repository](https://mostwiedzy.pl/pl/open-research-data/bias-mitigation-benchmark-that-includes-two-datasets,227084836236401-0?_share=322e9564d0341d8a)

## Requirements

- Python 3.7 or higher
- PyTorch 1.7 or higher
- torchvision
- NumPy
- Pandas
- scikit-learn
- OpenCV
- Matplotlib

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AgaMiko/targeted-data-augmentations.git
   cd targeted-data-augmentations
   ```

2. **Set up a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the datasets**

   - Follow the links provided in the [Datasets](#datasets) section to download the datasets.
   - Place the datasets in the `data/` directory as structured above.

## Usage

### Data Preparation

1. **Prepare the ISIC-2020 Skin Lesion Dataset**

   - Extract the images and place them in `data/isic_2020/images/`.
   - Place the metadata and labels in `data/isic_2020/`.

2. **Prepare the Gender Classification Dataset**

   - Extract the images and place them in `data/gender_classification/images/`.
   - Place the labels in `data/gender_classification/`.

3. **Prepare the Masks for Augmentation**

   - Download the masks from the [Data Repository](https://mostwiedzy.pl/pl/open-research-data/bias-mitigation-benchmark-that-includes-two-datasets,227084836236401-0?_share=322e9564d0341d8a).
   - Place the masks in the corresponding directories under `data/masks/`.

### Training Models

Run the `train.py` script to train a model with or without targeted data augmentation.

**Example: Training DenseNet121 on the ISIC-2020 dataset with TDA**

```bash
python scripts/train.py \
  --model densenet121 \
  --dataset isic \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --augmentation tda \
  --bias-type frame \
  --bias-probability 0.5 \
  --output-dir results/densenet121_isic_tda_frame_p0.5
```

**Parameters:**

- `--model`: Model architecture (`densenet121`, `efficientnet_b2`, `vit`)
- `--dataset`: Dataset to use (`isic`, `gender`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate
- `--augmentation`: Augmentation type (`none`, `tda`)
- `--bias-type`: Type of bias to augment (`frame`, `ruler`, `glasses`)
- `--bias-probability`: Probability of applying the bias augmentation (e.g., `0.5`)
- `--output-dir`: Directory to save model checkpoints and logs

### Evaluating Models

Run the `evaluate.py` script to evaluate a trained model and compute bias metrics using CBI.

**Example: Evaluating the trained model**

```bash
python scripts/evaluate.py \
  --model-path results/densenet121_isic_tda_frame_p0.5/model_best.pth \
  --dataset isic \
  --batch-size 64 \
  --bias-type frame \
  --output-dir results/densenet121_isic_tda_frame_p0.5/evaluation
```

### Reproducing Experiments

To reproduce the experiments from the paper:

1. **Train the models with specified parameters**

   - Use the `train.py` script with the configurations provided in the paper's **Training Details** section.

2. **Evaluate the models**

   - Use the `evaluate.py` script to compute the standard metrics and bias influence using CBI.

3. **Scripts and Configurations**

   - All scripts and configuration files used in the experiments are available in the `scripts/` and `configs/` directories.

4. **Results**

   - The results, including metrics and logs, will be saved in the specified `--output-dir`.

## Results

The key results from the experiments can be found in the `results/` directory. Each subdirectory contains:

- Model checkpoints
- Training logs
- Evaluation metrics
- Bias influence metrics computed using CBI

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
