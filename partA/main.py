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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.visualise import log_prediction_grid
from utils.visualise_layers import log_first_layer_filters
from utils.guided_backprop import log_guided_backprop


class EpochLoggerCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get("val_loss")
        val_acc = metrics.get("val_acc")
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

def train_model(config, wandb_logger=None):
    # Set seed for reproducibility
    seed_everything(42)

    early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=7,
    verbose=True,
    mode="min"
    )

    data_dir = "/home/gokul/LLM-Distillation/wino/DLAssign2/data"
    
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
        devices=1,
        strategy="auto",
        enable_checkpointing=True,
        callbacks=[RichProgressBar(), EpochLoggerCallback(), early_stop_callback]
    )
    trainer.fit(model, data_loader)
    
    save_name = (
        f"partA/save/"
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

def test_model(config, filename =None, wandb_logger=None):

    if filename==None:
        filename = (
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
    
    
    # Load the best model checkpoint
    model = CNNModel.load_from_checkpoint(filename, strict=False)
    
    # Prepare data module
    datamodule = INaturalistDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        img_size=config.img_size,
        use_aug=False  
    )
    datamodule.setup()

    test_loader = datamodule.test_dataloader()

    # Evaluate on test data
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        logger=wandb_logger
    )

    results = trainer.test(model, datamodule)

    log_prediction_grid(
        model,
        test_loader,
        num_classes=10,
        sort_by_confidence=False
    )

    log_first_layer_filters(model, test_loader)
    
    log_guided_backprop(model, test_loader)



def parse_arguments():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 - Part A")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"],
                        help="Run mode: train or test")
    parser.add_argument("-dd", "--data_dir", type=str, default="/data",
                        help="Path to the data directory")
    parser.add_argument("-img", "--img_size", type=int, default=256, help="Resized Image size for training/testing")
    parser.add_argument("-wp", "--wandb_project", type=str, default="wandbproject", 
                        help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="wandbentity", 
                        help="WandB entity (user or team)")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training/testing")
    parser.add_argument("-f", "--num_filters", type=int, nargs='+', default=[32, 32, 32, 32, 32],
                        help="List of filters for each conv layer")
    parser.add_argument("-k", "--kernel_size", type=int, default=3, help="Kernel size for conv layers")
    parser.add_argument("-ca", "--conv_activation", type=str, default="relu", help="Activation function for conv layers")
    parser.add_argument("-da", "--dense_activation", type=str, default="relu", help="Activation function for dense layers")
    parser.add_argument("-d", "--dense_neurons", type=int, default=128, help="Number of neurons in the dense layer")
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.0, help="Dropout rate after conv layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop", "adamw", "nadam"],
                        help="Optimizer choice: adam, sgd or rmsprop")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--use_aug", type=bool, default=True, help="Use data augmentation (default: False)")
    parser.add_argument("--use_batchnorm", type=bool, default=True, help="Use batch normalization (default: False)")
    parser.add_argument("-tf","--load_file", type=str, default=None, help="Load the file for testing trained model")

    args = parser.parse_args()
    return args
 

if __name__ == "__main__":

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
    # wandb.run.name = run_name
    # wandb.run.save()
    
    # wandb_logger = WandbLogger(project=args.wandb_project, log_model=False, name=run_name)

    if args.mode == "train":
        # Set the run name in WandB
        wandb.run.name = run_name
        wandb.run.save()
        wandb_logger = WandbLogger(project=args.wandb_project, log_model=False, name=run_name)
        train_model(config, wandb_logger)
    elif args.mode == "test":
        # Set the run name in WandB
        run_name = "test_" + args.load_file.split("/")[-1].split(".ck")[0]
        wandb.run.name = run_name
        wandb.run.save()
        wandb_logger = WandbLogger(project=args.wandb_project, log_model=False, name=run_name)
        test_model(config, args.load_file, wandb_logger)
    
    if wandb.run:
        wandb.run.finish()


