import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

def guided_backprop(model, image, target_neuron_index):
    # Set gradients to be positive only via hooks
    def relu_hook_function(module, grad_in, grad_out):
        # Only allow positive gradients
        return (torch.clamp(grad_in[0], min=0.0),)

    hooks = []
    # Attach hook to every ReLU to modify gradients during backprop
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_backward_hook(relu_hook_function))
    
    model.zero_grad()
    image.requires_grad = True
    output = model(image.unsqueeze(0))  # shape: (1, num_classes)
    # Create a one-hot target vector for a target neuron (cycling if target_neuron_index > num_classes)
    one_hot = torch.zeros_like(output)
    one_hot[0, target_neuron_index % output.shape[1]] = 1
    output.backward(gradient=one_hot)
    guided_grad = image.grad.data.cpu().numpy()
    
    # Remove hooks to restore original gradients
    for hook in hooks:
        hook.remove()
    return guided_grad  # shape might be (C, H, W) or possibly (H, W) if single-channel

def log_guided_backprop(model, test_loader, num_neurons=10):
    model.eval()
    images, _ = next(iter(test_loader))
    image = images[3]  # choose one image for demonstration
    wandb.log({"Input_Image": wandb.Image(image)})

    guided_results = []
    for neuron in range(num_neurons):
        guided_grad = guided_backprop(model, image, target_neuron_index=neuron)
        # Normalize for visualization
        guided_grad = guided_grad - guided_grad.min()
        guided_grad = guided_grad / (guided_grad.max() + 1e-8)
        # Check if the result has three dimensions (channels, H, W)
        if guided_grad.ndim == 3:
            # Transpose to (H, W, C) for plotting
            img_to_show = guided_grad.transpose(1, 2, 0)
        else:
            img_to_show = guided_grad  # e.g., when image is grayscale or only 2D data is returned
        guided_results.append(img_to_show)
    
    # Plot results in a grid of num_neurons rows and one column
    fig, axes = plt.subplots(num_neurons, 1, figsize=(5, 5 * num_neurons))
    for idx, ax in enumerate(axes):
        ax.imshow(guided_results[idx], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Neuron {idx}")
    plt.suptitle("Guided Backpropagation for 10 Neurons (CONV5)")
    wandb.log({"Guided_Backprop": fig})
    plt.close(fig)
