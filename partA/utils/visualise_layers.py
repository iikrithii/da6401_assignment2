import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

def log_first_layer_filters(model, test_loader):
    """
    Visualizes the activation maps of the first convolutional layer
    for a randomly selected image from the test set.
    The activations are plotted in an 8×8 grid (assuming 64 filters).
    The figure is logged to WandB.
    """
    model.eval()
    device = next(model.parameters()).device

    # Select a random image from the test set
    images_list = []
    for imgs, _ in test_loader:
        images_list.append(imgs)
    images_all = torch.cat(images_list, dim=0)
    random_idx = np.random.randint(0, images_all.size(0))
    random_image = images_all[random_idx].unsqueeze(0).to(device)

    # Extract the first convolutional layer from model.conv_blocks
    first_conv = None
    for layer in model.conv_blocks:
        if isinstance(layer, torch.nn.Conv2d):
            first_conv = layer
            break
    if first_conv is None:
        print("No convolutional layer found.")
        return

    # Register a forward hook to capture the activations
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    hook = first_conv.register_forward_hook(hook_fn)

    # Forward pass through the model
    with torch.no_grad():
        _ = model(random_image)

    # Remove the hook
    hook.remove()

    # Get the activation maps from the hook
    activation_maps = activations[0].squeeze(0)  # Shape: (num_filters, H, W)
    num_filters = activation_maps.shape[0]
    grid_dim = int(np.ceil(np.sqrt(num_filters)))

    # Plot the activation maps
    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(grid_dim * 2.5, grid_dim * 2.5))
    axes = axes.flatten()
    for i in range(grid_dim * grid_dim):
        ax = axes[i]
        if i < num_filters:
            act_map = activation_maps[i].numpy()
            act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-5)
            ax.imshow(act_map, cmap='viridis')
            ax.set_title(f"Filter {i}", fontsize=8)
        ax.axis('off')
    plt.suptitle("First Layer Activations", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  

    wandb.log({"First Layer Activations": wandb.Image(fig)})
    plt.close(fig)
