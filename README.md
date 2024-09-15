# Targeted Data Augmentations for Bias Mitigation

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
  - [Targeted Data Augmentation (TDA)](#targeted-data-augmentation-tda)
  - [Counterfactual Bias Insertion (CBI)](#counterfactual-bias-insertion-cbi)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
- [Results](#results)
- [Reproducing Experiments](#reproducing-experiments)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Targeted Data Augmentation (TDA):** A method to mitigate biases by intentionally introducing them during training.
- **Counterfactual Bias Insertion (CBI):** A technique to evaluate the influence of biases on model predictions.
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
│   ├── cbi.py
│   └── utils.py
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

## Datasets

### Full dataset  

* Masks for Augmentation:
* **Ruler Masks:** Used for skin lesion images, **Frame Masks:** Used for skin lesion images, **Glasses Masks:** Used for face images.
* **Gender Classification Dataset ** (A dataset of cropped face images labeled by gender) originally from [Kaggle Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)
* **ISIC-2020 Skin Lesion Dataset** (A dataset containing dermoscopic images for skin lesion classification.) originally from [Kaggle SIIM-ISIC Melanoma Classification Challenge](https://www.kaggle.com/c/siim-isic-melanoma-classification/data
* Manual bias annotations 

**Access:** [Data Repository](https://mostwiedzy.pl/pl/open-research-data/bias-mitigation-benchmark-that-includes-two-datasets,227084836236401-0?_share=322e9564d0341d8a)

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

### Targeted Data Augmentation (TDA)

**Targeted Data Augmentation (TDA)** is implemented in the `augmentations.py` script. The main idea is to randomly insert bias artifacts (e.g., frames, rulers, glasses) into training images with a specified probability.

**Implementation Details:**

- **Augmentation Functions:**

  - `add_frame(image, frame_masks)`: Adds a randomly selected frame mask to the image.
  - `add_ruler(image, ruler_masks)`: Adds a randomly selected ruler mask to the image.
  - `add_glasses(image, glasses_masks)`: Adds a randomly selected glasses mask to the image.

- **How It Works:**

  1. **Random Selection:** A mask is randomly selected from the available masks corresponding to the specified bias type.
  2. **Transformation:** The mask is resized, rotated, or scaled as needed to fit the target image.
  3. **Application:** The mask is overlaid onto the image at a random or specified position.
  4. **Probability:** The augmentation is applied with a certain probability `p`, which can be adjusted.

- **Usage in Training:**

  - During training, TDA is applied on-the-fly within the data loader. The `CustomDataset` class handles the application of TDA based on the specified parameters.

**Parameters:**

- `tda`: Boolean flag to enable or disable TDA.
- `bias_type`: Type of bias to augment (`'frame'`, `'ruler'`, `'glasses'`).
- `bias_prob`: Probability of applying the augmentation.

### Counterfactual Bias Insertion (CBI)

**Counterfactual Bias Insertion (CBI)** is implemented in the `cbi.py` script. CBI measures the influence of biases on model predictions by comparing the model's output on original images and images with inserted biases.

**Implementation Details:**

- **Process:**

  1. **Compute Original Predictions:**

     - Pass the original images through the model to obtain predictions.

  2. **Insert Biases into Images:**

     - Use the same augmentation functions (`add_frame`, `add_ruler`, `add_glasses`) to insert biases into the images.

  3. **Compute Biased Predictions:**

     - Pass the biased images through the model to obtain new predictions.

  4. **Compare Predictions:**

     - Compare the original and biased predictions to determine how many predictions changed (switched classes).

- **Usage in Evaluation:**

  - The `evaluate.py` script utilizes CBI to compute bias influence metrics.


**Running CBI Evaluation:**

In the `evaluate.py` script:

```python
from cbi import compute_cbi

# Load masks for bias insertion
masks = load_masks(args.bias_type)

# Compute CBI metrics
switched, switched_percentage = compute_cbi(model, dataloader, args.bias_type, masks)
print(f"Number of switched predictions: {switched}")
print(f"Percentage of switched predictions: {switched_percentage:.2f}%")
```

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
- `--augmentation`: Augmentation type (`'none'`, `'tda'`)
- `--bias-type`: Type of bias to augment (`'frame'`, `'ruler'`, `'glasses'`)
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


## **Additional Information:**

- **Helper Functions:**

  - `load_masks(bias_type)`: Loads the masks corresponding to the specified bias type from the `data/masks/` directory.
  - `transform_mask(mask, scale, angle)`: Applies scaling and rotation to the mask.
  - `overlay_mask(image, mask, x_offset, y_offset)`: Overlays the mask onto the image at the specified position.
  - `detect_eyes(image)`: Detects the position of the eyes in a face image for proper placement of glasses.

- **Data Augmentation Library:**

  - The augmentation functions utilize OpenCV and NumPy for image manipulation.
  - Ensure that you have OpenCV installed (`opencv-python` package).

- **Model Training Details:**

  - Training parameters such as learning rate, batch size, and number of epochs can be adjusted according to your hardware capabilities and desired performance.
  - Models are saved periodically and the best model (based on validation accuracy) is saved as `model_best.pth`.

- **Logging and Monitoring:**

  - Training and evaluation logs are saved in the `results/` directory.
  - You can use TensorBoard or other visualization tools to monitor training progress.

- **Error Handling:**

  - The scripts include basic error handling to ensure smooth execution.
  - If you encounter any issues, please refer to the FAQs or open an issue in the repository.

## FAQs

### Q: What is the purpose of TDA and how does it mitigate bias?

**A:** TDA introduces biases intentionally during training to disrupt spurious correlations between biased features and target classes. By exposing the model to biased data in a controlled manner, it learns to focus on relevant features and becomes more robust to biases.

### Q: How is CBI different from TDA?

**A:** CBI is a method used to evaluate the influence of biases on model predictions. It inserts biases into test images and compares the model's predictions before and after the insertion to assess how much the bias affects the model.

### Q: Can I use my own dataset with TDA and CBI?

**A:** Yes, you can apply TDA and CBI to any image dataset. You will need to provide appropriate masks for the biases you want to study and adjust the augmentation functions accordingly.
