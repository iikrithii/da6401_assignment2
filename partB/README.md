# Part B: Fine-Tuning Pre-Trained Models on iNaturalist

## Overview

This **Part B** module demonstrates a robust, extensible pipeline for fine‑tuning state‑of‑the‑art pre‑trained convolutional neural networks—namely **EfficientNet V2 Small** and **ResNet-50**—on the iNaturalist image classification dataset, using **PyTorch Lightning** for clean abstraction and **Weights & Biases (W&B)** for comprehensive experiment tracking. 

>The implementation of this code, and the detailed analysis is available in this [report](https://api.wandb.ai/links/ns25z040-indian-institute-of-technology-madras/81xj4n40).

---

## Table of Contents

1. [Features](#features)  
2. [Directory Structure & Dataset Layout](#directory-structure--dataset-layout)  
3. [Installation & Dependencies](#installation--dependencies)  
5. [Usage](#usage)  
   - [Training (`train.py`)](#training-trainpy)  
   - [Testing (`test.py`)](#testing-testpy)  
6. [Detailed CLI Reference](#detailed-cli-reference)  
7. [Core Components Walkthrough](#core-components-walkthrough)  
   - [DataModule: `INaturalistDataLoader`](#datamodule-inaturalistdataloader)  
   - [LightningModule: `FineTuneModel`](#lightningmodule-finetunemodel)  
8. [Extending the Pipeline](#extending-the-pipeline)  
   - [Adding New Backbones](#adding-new-backbones)  
   - [Custom Fine‑Tuning Strategies](#custom-fine-tuning-strategies)  
   - [Advanced Augmentations](#advanced-augmentations)  
---

## Features

- **Two Pre‑Trained Backbones**: Choose between **EfficientNet V2 S** for parameter efficiency or **ResNet-50** for wide adoption.  
- **Flexible Fine‑Tuning**: Four strategies—no freeze, freeze classifier only, freeze all but last block and classifier, or unfreeze all.  
- **Stratified Splits**: Balanced train/validation subsets drawn from your `data/train/` directory, preserving class proportions.  
- **ImageNet Preprocessing**: Standard resize, tensor conversion, and normalization (mean/std) for best transfer learning performance.  
- **W&B Integration**: Automatic logging of losses, accuracies, and custom metrics under named projects and runs.  

---

## Directory Structure & Dataset Layout

```bash
partB/
├── data_loader.py           # Data preparation & splits
├── finetune.py              # Model with FT strategies, optimizers
├── train.py                 # Entrypoint to fine-tune a new model
├── test.py                  # Entrypoint to evaluate a saved checkpoint
├── requirements.txt         # Python dependencies for Part B
├── save/                    # Stores best model checkpoints
```

1. **`data/`**: Contains two subdirectories:
   - **`train/`**: All labeled training images. Each class label folder should have at least ~50 samples for stratification quality.
   - **`val/`**: Held‑out images for final evaluation.

---

## Installation & Dependencies

### 1. Clone & Navigate

   ```bash
   git clone https://github.com/iikrithii/da6401_assignment2 && cd da6401_assignment2/partB
   ```
### 2. Install Requirements

```bash
pip install -r requirements.txt
```
### 3. **Login to W&B**:
 ```bash
 wandb login <your_api_key>
 ```


---


## Usage

With data in place, run the CLI scripts below.

### Training (`train.py`)

```bash
python train.py \
  --data_dir data \
  --model_type EfficientNet \
  --img_size 224 \
  --batch_size 64 \
  --use_aug \
  --ft_strategy True \
  --ft_type last_block \
  --max_epochs 20 \
  --wandb_project DA6401A2 \
  --wandb_entity my_team
```

**What Happens**:
1. **DataModule**: loads `data/train/`, splits 80/20 for train/val, loads `data/val/` for final test.  
2. **Model**: EfficientNet V2 Small with new `Linear` output for `num_classes`.  
3. **Fine‑Tuning**: Frees the last feature block + classifier head only.  
4. **Training**: Runs 20 epochs, logs metrics (“train_loss”, “val_loss”, “val_acc”) to W&B.  
5. **Checkpointing**: Saves best model (`lowest val_loss`) to `partB/save/`.

### Testing (`test.py`)

```bash
python test.py \
  --data_dir data \
  --batch_size 32 \
  --model_checkpoint save/EfficientNet_True_last_block_64_model.ckpt \
  --wandb_project DA6401A2 \
  --wandb_entity my_team
```

**What Happens**:
- Loads the checkpoint.
- Runs `trainer.test()` on the held‑out `data/val/` set, logging `test_loss` & `test_acc`.
- Final metrics appear in your W&B dashboard under the test run.

---

## Detailed CLI Reference

All scripts use Python’s `argparse`:

### `train.py` Arguments

| Arg                 | Type             | Default       | Description                                                                                             |
|---------------------|------------------|---------------|---------------------------------------------------------------------------------------------------------|
| `--data_dir`        | `str`            | **Required**  | Path to root data folder (must contain `train/` & `val/`).                                                |
| `--model_type`      | `str`            | `EfficientNet`| Choose backbone: `EfficientNet` or `ResNet`.                                                             |
| `--img_size`        | `int`            | `224`         | Resize images to square dimensions.                                                                     |
| `--batch_size`      | `int`            | `32`          | Batch size for train/val/test loaders.                                                                  |
| `--num_workers`     | `int`            | `4`           | Number of CPU threads for data loading.                                                                 |
| `--use_aug`         | `flag`           | `False`       | Enable flip/rotation augmentations in training.                                                         |
| `--ft_strategy`     | `bool`           | `True`        | Apply fine‑tuning parameter freezing.                                                                   |
| `--ft_type`         | `str`            | `last_layer`  | FT variant: `none`, `last_layer`, `last_block`, `full`.                                                 |
| `--max_epochs`      | `int`            | `10`          | Maximum epochs for training.                                                                            |
| `--wandb_project`   | `str`            | `wandbproject`| Project name on W&B for experiment logs.                                                                |
| `--wandb_entity`    | `str`            | `wandbentity` | W&B team or user entity.                                                                                |

### `test.py` Arguments

| Arg                     | Type   | Default       | Description                                          |
|-------------------------|--------|---------------|------------------------------------------------------|
| `--model_checkpoint`    | `str`  | **Required**  | Path to the `.ckpt` to evaluate.                     |
| `--data_dir`            | `str`  | **Required**  | Root data folder (same structure as training).       |
| `--batch_size`          | `int`  | `16`          | Batch size for testing.                              |
| `--num_workers`         | `int`  | `4`           | DataLoader workers.                                  |
| `--img_size`            | `int`  | `224`         | Image resize dimension.                              |
| `--wandb_project`       | `str`  | `wandbproject`| W&B project name for logging test metrics.           |
| `--wandb_entity`        | `str`  | `wandbentity` | W&B entity/team.                                     |

---

## Core Components Walkthrough

### DataModule: `INaturalistDataLoader` (`data_loader.py`)

1. **Init Parameters**:
   - `data_dir: str` – root with `train/` & `val/`.
   - `batch_size, num_workers, img_size` – DataLoader settings.
   - `val_split` – fraction for train→val split inside `train/`.
   - `use_aug: bool` – toggle train-time augmentations.

2. **Transforms**:
   - **Base**:  
     ```python
     transforms.Compose([
       transforms.Resize((img_size, img_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
     ])
     ```
   - **Augment**: adds:
     ```python
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(15),
     ```

3. **`setup()`**:
   - Loads `ImageFolder(train/, transform)` → `full_train`.
   - Extracts labels → conducts `StratifiedShuffleSplit` for train/val indices.
   - Creates `Subset` objects: `train_dataset`, `val_dataset`.
   - Loads `ImageFolder(val/, base_transform)` as `test_dataset`.

4. **Dataloaders**:
   - `train_dataloader()`: shuffle=True, persistent_workers=True.
   - `val_dataloader()`: shuffle=False.
   - `test_dataloader()`: shuffle=False.

### LightningModule: `FineTuneModel` (`finetune.py`)

1. **Constructor**:
   ```python
   def __init__(self, model_type, num_classes, ft_strategy, ft_type, lr)
   ```  
   - Saves hyperparameters and sets up `self.lr`.
   - Loads pre-trained model
   - Replaces final classifier with `nn.Linear(in_features, num_classes)`.
   - Calls `apply_ft_strategy(ft_strategy, ft_type)`.
   - Initializes `nn.CrossEntropyLoss()`.

2. **`apply_ft_strategy()`**:
   - **none**: trains all parameters.  
   - **last_layer**: freezes backbone, unfreezes head only.  
   - **last_block**: unfreezes head + final feature block (`features[-1]` or `layer4`).  
   - **full**: unfreezes all parameters.

3. **Forward & Steps**:
   - `forward(x)`: returns `self.model(x)` logits.
   - `training_step()`: logs `train_loss`.
   - `validation_step()`: logs `val_loss` & `val_acc`  
   - `test_step()`: logs `test_loss` & `test_acc`.

---

## Extending the Pipeline

### Adding New Backbones

1. **In `finetune.py`**, extend the constructor:
   ```python
   elif model_type == 'MobileNet':
       backbone = models.mobilenet_v3_small(pretrained=True)
       # adapt classifier similarly
   ```
2. **In `train.py` & `test.py`**, update argparser:
   ```python
   parser.add_argument('--model_type', choices=['EfficientNet','ResNet','MobileNet'], ...)
   ```

### Custom Fine‑Tuning Strategies

- **Mid Layers**:
  1. Freeze all parameters.  
  2. Unfreeze a user‑specified subset (e.g., `model.features[6:8]`).
- Add logic under `apply_ft_strategy(ft_strategy, ft_type)` and update `choices` in `train.py`.

### Advanced Augmentations

```python
self.transform = transforms.Compose([
  transforms.RandomResizedCrop(self.img_size, scale=(0.8,1.0)),
  transforms.ColorJitter(brightness=0.2, saturation=0.2),
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomRotation(degrees=30),
  transforms.ToTensor(),
  transforms.Normalize(mean,std)
])
```



