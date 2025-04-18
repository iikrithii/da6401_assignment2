import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb
from matplotlib import gridspec
import torch.nn.functional as F

# Define your class names in the same order as the labels
CLASS_NAMES = [
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia"
]

def log_prediction_grid(model, test_loader, num_classes=10, samples_per_class=3, sort_by_confidence=False):
    model.eval()
    device = next(model.parameters()).device

    class_data = {i: [] for i in range(num_classes)}  # Store (img, label, pred, confidence)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for img, label, pred, prob in zip(images, labels, preds, probs):
                cls = label.item()
                confidence = prob[pred].item()
                if len(class_data[cls]) < samples_per_class:
                    class_data[cls].append((img.cpu(), label.item(), pred.item(), confidence))

            if all(len(class_data[i]) >= samples_per_class for i in range(num_classes)):
                break

    # Sort class rows by average confidence or class-wise accuracy if enabled
    if sort_by_confidence:
        class_conf_scores = []
        for cls in range(num_classes):
            correct = sum(1 for x in class_data[cls] if x[1] == x[2])
            acc = correct / len(class_data[cls])
            avg_conf = np.mean([x[3] for x in class_data[cls]])
            class_conf_scores.append((cls, acc, avg_conf))
        class_order = [x[0] for x in sorted(class_conf_scores, key=lambda x: (-x[1], -x[2]))]
    else:
        class_order = list(range(num_classes))

    # Create the plot
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 3.5, num_classes * 3.5))
    if samples_per_class == 1:
        axes = axes[:, None]
    
    for row_idx, cls in enumerate(class_order):
        correct_count = 0
        total_conf = 0
        for col_idx in range(samples_per_class):
            img, label, pred, confidence = class_data[cls][col_idx]
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            pred_label = CLASS_NAMES[pred]
            actual_label = CLASS_NAMES[label]
            color = "green" if pred == label else "red"
            if pred == label:
                correct_count += 1
            total_conf += confidence


            ax = axes[row_idx, col_idx]
            ax.imshow(img_np)
            for side in ["top", "bottom", "left", "right"]:
                ax.spines[side].set_color(color)
                ax.spines[side].set_linewidth(4)
            ax.set_xticks([])
            ax.set_yticks([])

            if pred == label:
                ax.text(0.5, 1.17, f"Predicted {pred_label} right!", transform=ax.transAxes,
                    color=color, fontsize=9, ha="center", va="bottom", fontweight="bold")
                ax.text(0.5, 1.05, f"with {confidence:.2f}% confidence", transform=ax.transAxes,
                    color="black", fontsize=9, ha="center", va="bottom")
            else:
                ax.text(0.5, 1.17, f"Predicted {pred_label} instead of {actual_label}", transform=ax.transAxes,
                    color=color, fontsize=9, ha="center", va="bottom", fontweight="bold")
                ax.text(0.5, 1.05, f"with {confidence:.2f}% confidence", transform=ax.transAxes,
                    color="black", fontsize=9, ha="center", va="bottom")

        avg_acc = correct_count / samples_per_class
        avg_conf = total_conf / samples_per_class


    fig.suptitle("Model Predictions Grid", fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    wandb.log({
        "Prediction Grid": fig
    })
    plt.close(fig)



