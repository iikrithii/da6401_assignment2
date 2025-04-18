## Overview

This directory contains **Part A** of DA6401 Assignment 2: a fully configurable image classification pipeline built with **PyTorch Lightning**, designed to train and evaluate a 5-block convolutional neural network on the iNaturalist dataset. The code emphasizes modularity, reproducibility, and rich experiment logging to **Weights & Biases** (W&B), enabling data augmentation, batch normalization, multiple activation functions, and hyperparameter sweeps. It also provides visual interpretability tools (activation maps, prediction grids) to help you inspect model behavior.

>The implementation of this code, and the detailed analysis is available in this [report](https://api.wandb.ai/links/ns25z040-indian-institute-of-technology-madras/81xj4n40).

---

## Table of Contents

1. [Features](#features)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Configuration & Usage](#configuration--usage)
   - [Command-Line Arguments](#command-line-arguments)
   - [Training Example](#training-example)
   - [Testing Example](#testing-example)
5. [Core Components](#core-components)
   - [1. Model: ](#1-model-cnnmodel)[`CNNModel`](#1-model-cnnmodel)
   - [2. Data Loader: ](#2-data-loader-inaturalistdataloader)[`INaturalistDataLoader`](#2-data-loader-inaturalistdataloader)
   - [3. Activation Utilities](#3-activation-utilities)
   - [4. Visualization Modules](#4-visualization-modules)
6. [Extending the Codebase](#extending-the-codebase)
   - [Add New Activations](#add-new-activations)
   - [Add New Optimizers](#add-new-optimizers)
   - [Customize Augmentations](#customize-augmentations)


---

## Features

- **Configurable 5-Block CNN**: Five convolutional blocks with custom `num_filters`, `kernel_size`, optional batch normalization, a variety of activation functions (`ReLU`, `GELU`, `SiLU`, `Mish`), pooling, and dropout.
- **Dense Head**: Single fully connected layer (`dense_neurons`) plus configurable activation, optional batchnorm/dropout, and final output layer for `num_classes` classification.
- **Lightning DataModule**: Stratified train/validation split, optional data augmentation (random flips, rotations), configurable image resizing.
- **W&B Logging**: Automated tracking of train/val loss & accuracy, model checkpointing, prediction grids, and activation-map visualizations via forward‐hooks.
- **Extensible Architecture**: Easily add new activations, optimizers, transforms, or callbacks without touching core training logic.

---

## Directory Structure

```bash
├── partA/                          # Main code for Part A
│   ├── models/                     # Model definitions
│   │   └── cnnmodel.py             # `CNNModel` LightningModule
│   ├── utils/                      # Utility modules
│   │   ├── activations.py          # Get list of activations
│   │   ├── data_loader.py          # `INaturalistDataLoader` DataModule
│   │   ├── visualise.py            # Prediction-grid logger
│   │   └── visualise_layers.py     # Activation-map logger
│   ├── main.py                     # Eentrypoint for train/test
│   └── sweep.py                    # W&B sweep configuration & agent
└── requirements.txt                # Python dependencies
```
---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/iikrithii/da6401_assignment2 && cd da6401_assignment2/partA
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Login to W&B**:
   ```bash
   wandb login <your_api_key>
   ```

---

## Configuration & Usage

`partA/main.py` parses all hyperparameters and contains the entire pipeline for training or testing. You can run in **train** or **test** mode via `--mode`.

### Command-Line Arguments

| Flag                        | Type             | Default            | Description                                            |
| --------------------------- | ---------------- | ------------------ | ------------------------------------------------------ |
| `-m`, `--mode`              | `train`/`test`   | `train`            | Select between training or evaluation mode.            |
| `-dd`, `--data_dir`         | `str`            | `./data`           | Root directory, containing `train/` & `val/`.          |
| `-img`, `--img_size`        | `int`            | `256`              | Resize images to `img_size x img_size`.                |
| `-e`, `--epochs`            | `int`            | `10`               | Maximum training epochs.                               |
| `-b`, `--batch_size`        | `int`            | `32`               | Batch size for train/val/test.                         |
| `-f`, `--num_filters`       | `int` (5 values) | `[32,32,32,32,32]` | Filters in each of the 5 conv blocks.                  |
| `-k`, `--kernel_size`       | `int`            | `3`                | Kernel size for all conv layers.                       |
| `-ca`, `--conv_activation`  | `str`            | `relu`             | Conv-layer activation: `relu`, `gelu`, `silu`, `mish`. |
| `-da`, `--dense_activation` | `str`            | `relu`             | Dense-layer activation.                                |
| `-d`, `--dense_neurons`     | `int`            | `128`              | Hidden units in the dense head.                        |
| `-dr`, `--dropout_rate`     | `float`          | `0.0`              | Dropout after conv & dense layers.                     |
| `--lr`                      | `float`          | `1e-3`             | Learning rate.                                         |
| `--optimizer`               | `str`            | `adam`             | Optimizer: `adam`, `sgd`, `rmsprop`, `adamw`, `nadam`. |
| `--weight_decay`            | `float`          | `0.0`              | L2 regularization.                                     |
| `--use_aug`                 | `bool`           | `True`             | Enable data augmentation (train only).                 |
| `--use_batchnorm`           | `bool`           | `True`             | Apply BatchNorm in conv & dense layers.                |
| `-tf`, `--load_file`        | `str`            | `None`             | Path to checkpoint for testing mode.                   |
| `-wp`, `--wandb_project`    | `str`            | `wandbproject`     | W&B project name for logging.                          |
| `-we`, `--wandb_entity`     | `str`            | `wandbentity`      | W&B user or team entity.                               |

### Training Example

```bash
python partA/main.py \
  --mode train \
  --data_dir /path/to/data \
  --img_size 128 \
  --batch_size 64 \
  --epochs 20 \
  --num_filters 64 64 128 128 256 \
  --kernel_size 5 \
  --conv_activation silu \
  --dense_neurons 256 \
  --dense_activation gelu \
  --dropout_rate 0.3 \
  --use_aug True \
  --use_batchnorm True \
  --optimizer adamw \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --wandb_project DA6401A2 \
  --wandb_entity my_team
```

- Logging: Metrics and visualizations streamed live to W&B under the specified project/entity.
- Checkpoint: Best model saved automatically to `partA/save/` with hyperparameter details in filename.

### Testing Example

```bash
python partA/main.py \
  --mode test \
  --load_file partA/save/bs_64_epochs_20_aug_1_bn_1_filters_64-64-128-128-256_conv_act_silu_dense_256_dense_act_gelu_drop_0.3_lr_0.0005_optim_adamw_best_model.ckpt \
  --data_dir /path/to/data \
  --batch_size 64 \
  --img_size 128 \
  --wandb_project DA6401A2 \
  --wandb_entity my_team
```

- The script loads the checkpoint, runs `trainer.test()`, logs **test\_loss** & **test\_acc**.
- Generates and logs:
  - **Prediction Grid**: Random samples per class with boundary color / confidence.
  - **First-Layer Activations**: Heatmaps of feature maps via forward‐hooks.
  - Guided Backprop saliency maps 

---

## Core Components

### 1. Model: `CNNModel`

- Defined in `models/cnnmodel.py` as a subclass of `pl.LightningModule`.
- **Constructor parameters** mirror CLI flags: `input_channels`, `img_size`, `num_filters`, `kernel_size`, `conv_activation`, `dense_neurons`, `dense_activation`, `dropout_rate`, `use_batchnorm`, `optimizer_choice`, `lr`, `weight_decay`, `num_classes`.
- **Architecture**:
  1. **5 Convolutional Blocks**:
     - `Conv2d` with padding to maintain H×W.
     - *Optional* `BatchNorm2d`.
     - Activation via `get_activation()`.
     - `MaxPool2d(2)` halves spatial dims.
     - *Optional* `Dropout2d`.
  2. **Flatten** → Linear dense head → *Optional* `BatchNorm1d` → Activation → *Optional* `Dropout` → Output `Linear`.
- **Lightning hooks**:
  - `training_step()`: Cross-entropy loss, logs `train_loss` & `train_acc`.
  - `validation_step()`: Logs `val_loss` & `val_acc`.
  - `test_step()`: Logs `test_loss` & `test_acc`.
  - `configure_optimizers()`: Chooses among `Adam`, `SGD`, `RMSprop`, `AdamW`, `NAdam`.

### 2. Data Loader: `INaturalistDataLoader`

- Defined in `utils/data_loader.py`, subclass of `pl.LightningDataModule`.
- **Transforms**:
  - Base: `Resize(img_size)`, `ToTensor()`.
  - Augment: `RandomHorizontalFlip()`, `RandomRotation(15°)`, when `use_aug=True`.
- **Dataset**: Expects `data_dir/train` & `data_dir/val` with subfolders per class (ImageFolder format).
- **Split**: Stratified ShuffleSplit (`sklearn`) to carve `val_split` fraction from train.
- **Dataloaders**: `train_dataloader()`, `val_dataloader()`, `test_dataloader()` with `persistent_workers` for speed.

### 3. Activation Utilities

- **File**: `utils/activations.py`.
- **Function**: `get_activation(name: str) -> nn.Module` returns:
  - `ReLU()`, `GELU()`, `SiLU()`, `Mish()`.
- **Extensible**: Add new `elif` clauses for custom activations, and update CLI choices.

### 4. Visualization Modules

- **Prediction Grid** (`utils/visualise.py`):

  - Randomly samples `samples_per_class` images per true label from `test_loader`.
  - Runs forward pass → softmax → preds & confidences.
  - Plots a grid (`num_classes x samples_per_class`) with border colored green/red, annotated with predicted vs actual and confidence.


- **First-Layer Activations** (`utils/visualise_layers.py`):

  - Looks into first `Conv2d` layer.
  - Forward‐passes a random test image, captures feature maps.
  - Normalizes each map → plots in √N × √N grid.

- **Guided Backprop**: Hook & gradient-based saliency in `utils/guided_backprop.py`.


---

## Extending the Codebase

### Add New Activations

1. Update `utils/activations.py`:
   ```python
   elif activation_name == 'swish':
       return nn.SiLU()  # custom Swish implementation
   ```
2. Add `'swish'` to the `--conv_activation` and `--dense_activation` choices in `parse_arguments()` of `main.py`.

### Add New Optimizers

1. In `models/cnnmodel.py` → `configure_optimizers()`, add:
   ```python
   elif optimizer_choice == 'adagrad':
       optimizer = torch.optim.Adagrad(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
   ```
2. Extend the `--optimizer` choices in `main.py` to include `'adagrad'`.

### Customize Augmentations

Edit `utils/data_loader.py` within the `if self.use_aug:` block. For example:

```python
transforms.Compose([
    transforms.RandomResizedCrop(self.img_size),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])
```


---

