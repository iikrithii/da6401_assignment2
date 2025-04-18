import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from finetune import FineTuneModel
from data_loader import INaturalistDataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning pre-trained model on iNaturalist dataset")
    # Command-line arguments for model, data location, wandb configuration, and fine-tuning strategies.
    parser.add_argument("--model_type", type=str, default="EfficientNet",
                        choices=["EfficientNet", "ResNet"],
                        help="Pre-trained model to finetune")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the iNaturalist dataset directory")
    parser.add_argument("--wandb_project", type=str, default="wandbproject",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="wandbentity",
                        help="Weights & Biases entity name")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size (square)")
    parser.add_argument("--use_aug", action="store_true",
                        help="Whether to use data augmentation")
    parser.add_argument("--ft_strategy", type=bool, default=True,
                        help="Whether to use fine-tuning strategy (i.e., freezing layers). True/False")
    parser.add_argument("--ft_type", type=str, default="last_layer",
                        choices=["none", "last_layer", "last_block", "full"],
                        help="Fine-tuning strategy type: last_layer, last_block, or full. Use 'none' if ft_strategy is False.")

    args = parser.parse_args()

    # Setup WandB logger for experiment tracking.
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)

    # Initialize data module; using our stratified split loader.
    data_module = INaturalistDataLoader(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_aug=args.use_aug
    )
    data_module.setup()

    # Create model instance with the given fine-tuning configuration.
    model = FineTuneModel(
        model_type=args.model_type,
        num_classes=len(data_module.full_train.classes),  
        ft_strategy=args.ft_strategy,
        ft_type=args.ft_type
    )

    # Setup trainer with GPU acceleration if available.
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         accelerator="gpu" if torch.cuda.is_available() else "cpu", 
                         devices="auto")
    
    trainer.fit(model, datamodule=data_module)
    save_name=f"save/{args.model_type}_{args.ft_strategy}_{args.ft_type}_{args.batch_size}_model.ckpt"
    trainer.save_checkpoint(save_name)
 
if __name__ == '__main__':
    main()