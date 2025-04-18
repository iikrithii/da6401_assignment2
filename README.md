# DA6401 Assignment 2

The objective of this assignment is to a CNN model from scratch and learn how to tune the hyperparameters and visualize filters and finetune a pre-trained model. This repository explores two complementary approaches for classifying a 10-class subset of the iNaturalist dataset:

1. **Part A – Training from Scratch**  
   Design and train a fully configurable five‑block convolutional neural network (CNN) from the ground up. Perform extensive hyperparameter sweeps with Weights & Biases (W&B) to identify optimal configurations, and visualize model behavior and explanations.

2. **Part B – Fine‑Tuning Pre‑Trained Models**  
   Leverage transfer learning by adapting EfficientNet V2 Small and ResNet‑50 backbones. Experiment with different layer‑freezing strategies, compare performance against the scratch‑trained CNN, and log experiments to W&B.

>The implementation of this repository, and the detailed analysis is available in this [report](https://api.wandb.ai/links/ns25z040-indian-institute-of-technology-madras/81xj4n40).


---

## 📁 Repository Structure

```
├── partA/                         # Training a CNN from scratch
│   ├── models/                    # `CNNModel` LightningModule
│   ├── utils/                     # Activations, data loader, visualizations
│   ├── main.py                    # CLI: train & test, sweep integration
│   ├── sweep.py                   # W&B hyperparameter sweep setup
│   ├── requirements.txt           # Part A dependencies
│   └── README.md                  # Detailed docs for Part A

├── partB/                         # Fine‑tuning pre‑trained architectures
│   ├── data_loader.py             # `INaturalistDataLoader` for stratified splits
│   ├── finetune.py                # `FineTuneModel` LightningModule
│   ├── train.py                   # CLI for fine‑tuning experiments
│   ├── test.py                    # CLI for checkpoint evaluation
│   ├── requirements.txt           # Part B dependencies
│   ├── save/                      # Auto‑saved best checkpoints
│   └── README.md                  # Detailed docs for Part B

└── README.md                      # This overarching introduction & guide
```

---

## Dependencies

1. **Clone the repository**
   ```bash
   git clone https://github.com/iikrithii/da6401_assignment2.git
   cd da6401_assignment2
   ```

2. **Install dependencies**
   - **Part A**:
     ```bash
     cd partA
     pip install -r requirements.txt
     ```
   - **Part B**:
     ```bash
     cd ../partB
     pip install -r requirements.txt
     ```

## Preparing Your Data

1. **Folder Structure**: Ensure your dataset sits in `<anypath>/data/`in the following format:
   ```text
   /data/train/   # class1/, class2/, …
   /data/val/     # class1/, class2/, …
   ```
2. **Image Requirements**:
   - Formats: `JPEG`, `PNG`, etc.
   - Minimum resolution: ≥128×128 recommended.
3. **Class Balance**: Stratified splitting relies on having multiple samples per class. If a class has <5 images, consider merging or removing it.
4. **Val Split Percentage**: Default is 20% split of the **train/** folder for validation; adjust via the `val_split` argument in `INaturalistDataLoader` if needed.

## Guideleines for Implementing Code
   - **Part A**: See [partA/README.md](partA/README.md)
   - **Part B**: See [partB/README.md](partB/README.md)

---

## 📊 Results Summary

### Part A – Scratch‑trained CNN
- **Best Validation Accuracy**: 48.05%  
- **Test Accuracy**: ~44.5%  
- **Highlights**: Modular 5‑block CNN, W&B sweep revealed progressive filter depths, non‑linear activations (GELU/SILU/MISH) and moderate dropout rates (~0.04) worked best.

### Part B – Transfer Learning
| Model                         | Val Acc. | Test Acc. |
|-------------------------------|----------|-----------|
| Scratch CNN (BS=16)          | 48.0%    | 44.0%     |
| ResNet‑50 (BS=16)            | 77.0%    | 76.85%    |
| ResNet‑50 (BS=32)            | 78.1%    | 76.05%    |
| EfficientNet V2 S (BS=16)    | 75.2%    | 73.3%     |
| EfficientNet V2 S (BS=32)    | 75.9%    | 74.1%     |

**Analysis**: Fine‑tuning pre‑trained backbones yields a ~30 ppt gain in validation accuracy over scratch‑trained models, underscoring transfer learning’s power for limited data.

---



