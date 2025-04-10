import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.activations import get_activation

class CNNModel(pl.LightningModule):
    """
    A configurable CNN model with 5 convolutional blocks followed by one dense layer and an output layer.
    
    Each convolutional block contains:
      - Convolution layer (with kernel_size padding to maintain spatial dimensions)
      - (Optional) Batch Normalization
      - Activation function (configurable via activations.py)
      - Dropout 
      - MaxPooling layer (reducing spatial dimensions by a factor of 2)
      
    Final dense block:
      - A fully connected layer projecting the flattened features to dense_neurons
      - (Optional) Batch Normalization
      - Dense activation function (configurable via dense_activation)
      - Dropout (if specified)
      - Output layer with num_classes neurons (for classification)
      
    Hyperparameters:
      - input_channels: Number of channels in the input image.
      - num_filters: List of integers representing number of filters in each conv layer (must contain exactly 5 elements).
      - kernel_size: Kernel size for all conv layers.
      - activation: Activation function for convolutional layers (one of: 'ReLU', 'GELU', 'SILU', 'MISH').
      - dense_neurons: Number of neurons in the dense layer.
      - dense_activation: Activation function for the dense layer (e.g., 'ReLU', 'GELU', etc.).
      - dropout_rate: Dropout probability.
      - use_batchnorm: Whether to apply batch normalization.
      - optimizer_choice: Choice of optimizer ('adam', 'sgd', or 'rmsprop').
      - lr: Learning rate.
      - weight_decay: Weight decay for optimizer regularization.
      - num_classes: Number of output classes.
    """

    def __init__(self, input_channels=3, img_size=64, num_filters=[32, 32, 32, 32, 32],
                 kernel_size=3, kernel_sizes=None, conv_activation='relu', dense_activation="relu", dense_neurons=128, dropout_rate=0.0,
                 use_batchnorm=False, optimizer_choice="adam", lr=1e-3, weight_decay=0.0, num_classes=10):
        super(CNNModel, self).__init__()
        self.save_hyperparameters()

        # Build blocks for convolutional layers
        if kernel_sizes is None:
            kernel_sizes = [kernel_size] * len(num_filters)
        self.hparams.kernel_sizes = kernel_sizes
        layers = []
        current_in = input_channels
        for idx, num_filter in enumerate(num_filters):
            ks=kernel_sizes[idx]
            # Convolution layer
            layers.append(nn.Conv2d(current_in, num_filter, kernel_size=ks, padding=kernel_size // 2))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(num_filter))
            # Activation function from the activations module
            layers.append(get_activation(conv_activation))
            # Max pooling layer to reduce spatial dimensions by factor of 2
            layers.append(nn.MaxPool2d(2))
            # Dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            current_in = num_filter
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # Dummy input for getting the flatten dimension for the dense layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, img_size, img_size)  
            conv_out = self.conv_blocks(dummy_input)
            self.flatten_dim = conv_out.view(1, -1).shape[1]
        
        # Fully connected dense blocks with activation layers
        self.dense_block = nn.Sequential(
            nn.Linear(self.flatten_dim, dense_neurons),
            nn.BatchNorm1d(dense_neurons) if use_batchnorm else nn.Identity(),
            get_activation(dense_activation),  # Custom dense activation from activations.py
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.out = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense_block(x)
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        preds = torch.argmax(logits, dim=1)
        train_acc = (preds == labels).float().mean()
        self.log("train_acc", train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_choice = self.hparams.optimizer_choice.lower()
        if optimizer_choice == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif optimizer_choice == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif optimizer_choice == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif optimizer_choice == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif optimizer_choice == "nadam":
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Unsupported optimizer")
        return optimizer
    