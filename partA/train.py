import argparse
import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from models.cnn_model import CNNModel
from utils.data_loader import INaturalistDataLoader
import wandb
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import Callback
import pprint



class EpochLoggerCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get("val_loss")
        val_acc = metrics.get("val_acc")
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

def train_model(config, wandb_logger):
    # Set seed for reproducibility
    seed_everything(42)

    data_dir = Path(__file__).parent.parent.parent / "data"
    
    data_loader = INaturalistDataLoader(data_dir=data_dir,
                                         batch_size=config.batch_size,
                                         img_size=config.img_size, use_aug=config.use_aug)
    data_loader.setup()
    
    # Initialize model with hyperparameters from config
    model = CNNModel(
        input_channels=3,
        img_size=config.img_size,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        conv_activation=config.conv_activation,       
        dense_neurons=config.dense_neurons,
        dense_activation=config.dense_activation,  
        dropout_rate=config.dropout_rate,
        use_batchnorm=config.use_batchnorm,
        optimizer_choice=config.optimizer,
        lr=config.lr,
        weight_decay=config.weight_decay,
        num_classes=10
    )
    # Initialize trainer
    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        enable_checkpointing=True,
        callbacks=[RichProgressBar(), EpochLoggerCallback()]
    )
    trainer.fit(model, data_loader)
    
    save_name = (
        f"save/"
        f"bs_{config.batch_size}_"
        f"epochs_{config.epochs}_"
        f"aug_{int(config.use_aug)}_"
        f"bn_{int(config.use_batchnorm)}_"
        f"filters_{'-'.join(map(str, config.num_filters))}_"
        f"conv_act_{config.conv_activation}_"
        f"dense_{config.dense_neurons}_"
        f"dense_act_{config.dense_activation}_"
        f"drop_{config.dropout_rate}_"
        f"lr_{config.lr}_"
        f"optim_{config.optimizer}_"
        f"best_model.ckpt"
    )
    trainer.save_checkpoint(save_name)

def test_model(config):
    # Load the best model checkpoint
    model = CNNModel.load_from_checkpoint("save/best_model.ckpt")
    
    # Prepare data module
    data_loader = INaturalistDataLoader(data_dir=config.data_dir,
                                         batch_size=config.batch_size,
                                         img_size=config.img_size)
    data_loader.setup()
    
    # Evaluate on test data
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )
    results = trainer.test(model, data_loader)
    print("Test results:", results)

def parse_arguments():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 - Part A")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"],
                        help="Run mode: train or test")
    parser.add_argument("-dd", "--data_dir", type=str, default="../data",
                        help="Path to the data directory")
    parser.add_argument("-img", "--img_size", type=int, default=256, help="Resized Image size for training/testing")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment2", 
                        help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ns25z040-indian-institute-of-technology-madras", 
                        help="WandB entity (user or team)")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training/testing")
    parser.add_argument("-f", "--num_filters", type=int, nargs='+', default=[32, 32, 32, 32, 32],
                        help="List of filters for each conv layer")
    parser.add_argument("-k", "--kernel_size", type=int, default=3, help="Kernel size for conv layers")
    parser.add_argument("-a", "--activation", type=str, default="relu", help="Activation function for conv layers")
    parser.add_argument("-d", "--dense_neurons", type=int, default=128, help="Number of neurons in the dense layer")
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.0, help="Dropout rate after conv layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"],
                        help="Optimizer choice: adam, sgd or rmsprop")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--use_aug", type=bool, default=False, help="Use data augmentation (default: False)")
    parser.add_argument("--use_batchnorm", type=bool, default=False, help="Use batch normalization (default: False)")

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    # Build a run name using the sweep hyperparameters
    run_name = (
        f"bs_{config.batch_size}_"
        f"epochs_{config.epochs}_"
        f"aug_{int(config.use_aug)}_"
        f"bn_{int(config.use_batchnorm)}_"
        f"filters_{'-'.join(map(str, config.num_filters))}_"
        f"conv_act_{config.conv_activation}_"
        f"dense_{config.dense_neurons}_"
        f"dense_act_{config.dense_activation}_"
        f"drop_{config.dropout_rate}_"
        f"lr_{config.lr}_"
        f"optim_{config.optimizer}"
    )
    # Set the run name in WandB
    wandb.run.name = run_name
    wandb.run.save()
    
    wandb_logger = WandbLogger(project=args.wandb_project, log_model=False, name=run_name)

    if args.mode == "train":
        train_model(config, wandb_logger)
    elif args.mode == "test":
        test_model(config)
    
    wandb.finish() 


if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,  
            'max_iter': 10,  
            'eta': 2,
        },
        'metric': {
            'name': 'val_acc',  
            'goal': 'maximize'

        },
        'parameters': {
            'epochs': {
                'values': [10]
            },
            'num_filters': {
                'values': [[32, 32, 32, 32, 32], [64, 64, 64, 64, 64], [32, 64, 128, 256, 512]]
            },

            'dense_neurons': {
                'values': [128, 256]
            },
            'dropout_rate': {
                'min': 0.0, 
                'max': 0.5,
            },
            'kernel_size': {
                'values': [3, 5]
            },
            'lr': {
                'min': 1e-5,
                'max': 1e-3,
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'conv_activation': {
                'values': ['relu', 'gelu', 'silu', 'mish']
            },
            'dense_activation': {
                'values': ['relu', 'gelu']
            },
            'use_aug': {
                'values': [True, False]
            },
            'use_batchnorm': {
                'values': [True, False]
            },
            "weight_decay": {
             "values": [0.0, 0.0001, 0.001]
            },
            "optimizer": {
                "values": ["adam", "sgd", "nadam", "adamw"]
            }
        }
    }

    pprint.pprint(sweep_config)

    # Create the sweep – make sure to specify your wandb project and entity
    # sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment2", entity="ns25z040-indian-institute-of-technology-madras")
    # print("Sweep ID:", sweep_id)
    sweep_id= "redi2h5e"
    # Run the sweep agent, which calls your main training function
    wandb.agent(sweep_id, function=main, project="DA6401_Assignment2", entity="ns25z040-indian-institute-of-technology-madras")