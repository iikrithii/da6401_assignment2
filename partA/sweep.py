import pprint
import wandb
from train import main
                        
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
sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment2", entity="ns25z040-indian-institute-of-technology-madras")
print("Sweep ID:", sweep_id)

# Run the sweep agent, which calls your main training function
wandb.agent(sweep_id, function=main)

