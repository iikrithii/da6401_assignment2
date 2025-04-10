import torch.nn as nn

def get_activation(activation_name):
    """
    Returns the activation module corresponding to the given activation_name.
    Supported activations: 'relu', 'gelu', 'silu', 'mish'
    """
    activation_name = activation_name.lower()
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'silu':
        return nn.SiLU()
    elif activation_name == 'mish':
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")