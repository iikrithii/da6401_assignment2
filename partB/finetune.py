import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class FineTuneModel(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a pre-trained model.
    Supports different fine-tuning strategies.
    """
    def __init__(self, model_type="EfficientNet", num_classes=10, ft_strategy=False, ft_type="none", lr=1e-3):
        super(FineTuneModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Load the pre-trained model based on model_type.
        if model_type == "EfficientNet":
            self.model = models.efficientnet_v2_s(pretrained=True)
            in_features = self.model.classifier[1].in_features
            # Replace the final classification layer.
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_type == "ResNet":
            self.model = models.resnet50(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model type {model_type} not recognized. Choose from EfficientNet or ResNet.")

        # Apply the selected fine-tuning strategy.
        self.apply_ft_strategy(ft_strategy, ft_type)

        # Cross-entropy loss is used for classification.
        self.criterion = nn.CrossEntropyLoss()

    def apply_ft_strategy(self, ft_strategy, ft_type):
        """
        Modify the model parameters according to the desired fine-tuning strategy.
        """
        if not ft_strategy or ft_type == "none":
            # No freezing applied: train the entire model.
            print("No freezing applied. Training the entire model from scratch.")
            return

        print(f"Applying fine-tuning strategy: {ft_type}")

        if ft_type == "last_layer":
            # Freeze all layers except the final classification layer.
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            if hasattr(self.model, "classifier"):  # EfficientNet
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, "fc"):  # ResNet
                for param in self.model.fc.parameters():
                    param.requires_grad = True
        elif ft_type == "last_block":
            # Freeze all layers first.
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            # Unfreeze the classification head.
            if hasattr(self.model, "classifier"):  # EfficientNet
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
                # Unfreeze the last block of features (assuming the last element of features).
                for param in self.model.features[-1].parameters():
                    param.requires_grad = True
            elif hasattr(self.model, "fc"):  # ResNet
                for param in self.model.fc.parameters():
                    param.requires_grad = True
                # Unfreeze the last block (layer4) of ResNet.
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
        elif ft_type == "full":
            # Unfreeze all layers for full fine-tuning.
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("ft_type must be one of ['none', 'last_layer', 'last_block', 'full']")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}
    
    def configure_optimizers(self):
        # Only parameters with requires_grad=True are passed to the optimizer.
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer
