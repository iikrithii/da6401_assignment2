import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from finetune import FineTuneModel
from data_loader import INaturalistDataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model on iNaturalist dataset")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the iNaturalist dataset directory")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloader")
    parser.add_argument("--use_aug", action="store_true",
                        help="Whether to use data augmentation")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size (square)")
    parser.add_argument("--wandb_project", type=str, default="DA6401_Assignment2",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="ns25z040-indian-institute-of-technology-madras",
                        help="Weights & Biases entity name")
    args = parser.parse_args()

    # Setup WandB logger for logging test metrics.
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
    
    # Initialize the DataModule using your stratified split loader.
    data_module = INaturalistDataLoader(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        img_size=args.img_size, 
    )
    data_module.setup()

    # Load the saved model from the checkpoint.
    model = FineTuneModel.load_from_checkpoint(args.model_checkpoint)

    trainer = pl.Trainer(logger=wandb_logger, 
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices="auto",
                         strategy="auto")
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()
