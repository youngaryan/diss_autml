import os
import time
import csv
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
from class_based_fixed_speech_brain import EmotionRecognitionTrainer

# ---------------------
# Data Preparation
# ---------------------
# Load dataset and prepare label mapping
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
label_encoder_obj = LabelEncoder()
df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)

# Split into training and validation sets (80/20 split)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Constant parameters
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------
# Sweep Training Function
# ---------------------
def sweep_train():
    # Initialize a new wandb run
    wandb.init()
    
    # Get hyperparameters from wandb.config
    config = wandb.config
    lr = config.lr
    num_epochs = config.num_epochs
    unfreeze_epoch = config.unfreeze_epoch
    max_length = config.max_length

    # Set up configuration for the trainer
    run_config = {
        "batch_size": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
        "unfreeze_epoch": unfreeze_epoch,
        "max_length": max_length,
        "device": device
    }
    print(f"\nRunning trial with config: {run_config}")
    
    # Initialize trainer with current hyperparameters
    trainer = EmotionRecognitionTrainer(run_config, train_df, valid_df, mapping)
    
    start_time = time.time()
    results = trainer.train()
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    # Log the validation accuracy and other metrics to wandb
    wandb.log({
        "validation_accuracy": results.get("validation_accuracy", 0),
        "total_time": total_time
    })
    
    # Finish the run
    wandb.finish()


# ---------------------
# Set Up and Run W&B Sweep
# ---------------------
if __name__ == "__main__":
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # you can change this to 'random' or 'grid' as needed
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform',
                'min': 1e-6,
                'max': 1e-3
            },
            'num_epochs': {
                'values': list(range(1, 11))  # integers 1 through 10
            },
            'unfreeze_epoch': {
                'values': list(range(0, 6))   # integers 0 through 5
            },
            'max_length': {
                'values': [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000]
            }
        }
    }
    
    # Create the sweep and get the sweep ID
    sweep_id = wandb.sweep(sweep_config, project="emotion-recognition-hpo")
    
    # Launch the sweep agent to run 50 trials
    wandb.agent(sweep_id, function=sweep_train, count=50)
