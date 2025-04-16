import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb

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

def log_prediction_grid(model, test_loader, num_classes=10, samples_per_class=3):
    """
    Log a grid of sample images and their predictions.
    The grid will have num_classes rows and samples_per_class columns.
    Adds a row header indicating the class (true label) and 
    displays the predicted label in green if correct and red if incorrect.
    
    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        wandb: The wandb module instance.
        num_classes (int): Number of classes (rows in the grid).
        samples_per_class (int): Number of samples per class (columns in the grid).
    """
    model.eval()
    # Dictionaries to store images and predictions for each class
    class_images = {i: [] for i in range(num_classes)}
    class_preds = {i: [] for i in range(num_classes)}
    class_labels = {i: [] for i in range(num_classes)}

    # Iterate over test data to collect samples until each class has enough
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for img, label, pred in zip(images, labels, preds):
                cls = label.item()
                if len(class_images[cls]) < samples_per_class:
                    class_images[cls].append(img)
                    class_preds[cls].append(pred.item())
                    class_labels[cls].append(label.item())
            # Stop if we've collected enough samples for every class
            if all(len(class_images[i]) >= samples_per_class for i in range(num_classes)):
                break

    # Create a grid with num_classes rows and samples_per_class columns
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 3, num_classes * 3))
    # If only one column is present, wrap axes in a list for consistency
    if samples_per_class == 1:
        axes = axes[:, None]
        
    for cls in range(num_classes):
        for j in range(samples_per_class):
            ax = axes[cls, j]
            # Convert tensor image to NumPy array (assuming shape [C, H, W])
            img = class_images[cls][j].cpu().numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis("off")
            # Prepare texts for predicted and actual labels
            pred_index = class_preds[cls][j]
            actual_index = class_labels[cls][j]
            pred_name = CLASS_NAMES[pred_index]
            actual_name = CLASS_NAMES[actual_index]
            # Choose color based on correctness
            color = "green" if pred_index == actual_index else "red"
            # Add predicted label text at the top of the subplot
            ax.text(0.5, 1.05, f"Pred: {pred_name}", transform=ax.transAxes, color=color,
                    fontsize=9, ha="center", va="bottom")
            # Add actual class text below the image
            ax.text(0.5, 1.14, f"Actual: {actual_name}", transform=ax.transAxes, color="black",
                    fontsize=9, ha="center", va="top")
        # # Set the row header using the leftmost subplot's y-label, centered vertically.
        # axes[cls, 0].set_ylabel(CLASS_NAMES[cls], rotation=0, labelpad=40, fontsize=12, va="center")
    wandb.log({"Prediction Grid": fig})
    plt.close(fig)
