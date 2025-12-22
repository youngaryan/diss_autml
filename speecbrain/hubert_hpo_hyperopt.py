import os
import time
import csv
import math
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from numpy.random import default_rng
from statics import SEED
import random
import numpy as np

# --- UPDATED IMPORT ---
from class_based_hubert import EmotionRecognitionTrainer
# ----------------------

# ---------------------
# Reproducibility
# ---------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------
# Data Preparation
# ---------------------
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
label_encoder_obj = LabelEncoder()
df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Define Hyperopt Search Space
# ---------------------
# Note: Reduced LR upper bound to 1e-4 and max_length to 5s for HuBERT stability
max_length_options = [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000]

space = {
    'lr': hp.loguniform('lr', math.log(1e-6), math.log(1e-4)),
    'num_epochs': hp.quniform('num_epochs', 1, 10, 1),
    'unfreeze_epoch': hp.quniform('unfreeze_epoch', 0, 5, 1),
    'max_length': hp.choice('max_length', max_length_options)
}

# ---------------------
# Objective Function
# ---------------------
def objective(params):
    # Cast float parameters from quniform to int
    params['num_epochs'] = int(params['num_epochs'])
    params['unfreeze_epoch'] = int(params['unfreeze_epoch'])

    config = {
        "batch_size": batch_size,
        "lr": params['lr'],
        "num_epochs": params['num_epochs'],
        "unfreeze_epoch": params['unfreeze_epoch'],
        "max_length": params['max_length'],
        "device": device
    }
    print(f"\nRunning trial with config: {config}")

    # Initialize the HuBERT Trainer
    trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)

    start_time = time.time()
    results = trainer.train()
    total_time = time.time() - start_time
    results["total_time"] = total_time

    validation_accuracy = results.get("validation_accuracy", 0)
    
    # Hyperopt minimizes loss, so we flip accuracy to negative
    loss = -validation_accuracy

    # Immediate CSV writing
    csv_file = os.path.join(os.path.dirname(__file__), "hpo_results_hyperopt_hubert.csv")
    fieldnames = [
        "lr", "num_epochs", "unfreeze_epoch", "max_length",
        "validation_accuracy", "total_time"
    ]
    
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
            
        writer.writerow({
            "lr": config.get("lr"),
            "num_epochs": config.get("num_epochs"),
            "unfreeze_epoch": config.get("unfreeze_epoch"),
            "max_length": config.get("max_length"),
            "validation_accuracy": validation_accuracy,
            "total_time": total_time
        })

    return {
        'loss': loss,
        'status': STATUS_OK,
        'config': config,
        'results': results
    }

# ---------------------
# Run Hyperopt Search
# ---------------------
if __name__ == "__main__":
    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=default_rng(SEED)
    )

    # Convert the index returned by hp.choice back to the actual value
    best['max_length'] = max_length_options[best['max_length']]

    print("\nBest hyperparameters found:")
    print(best)
