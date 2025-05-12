import os
import time
import csv
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# CSV file setup
csv_file = os.path.join(os.path.dirname(__file__), "hpo_results_bca1.csv")
fieldnames = [
    "trial_number", "lr", "num_epochs", "unfreeze_epoch", "max_length",
    "validation_accuracy", "total_time"
]
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# ---------------------
# Objective Function for HPO Tuning
# ---------------------
def objective(trial):
    # Suggest hyperparameter values using Optuna's API
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 10)
    unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 0, 5)
    max_length = trial.suggest_categorical("max_length", [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000])
    
    config = {
        "batch_size": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
        "unfreeze_epoch": unfreeze_epoch,
        "max_length": max_length,
        "device": device
    }
    print(f"\nRunning trial with config: {config}")
    
    # Initialize trainer with current hyperparameters
    trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)
    
    start_time = time.time()
    results = trainer.train()
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    # Store additional information for later inspection
    trial.set_user_attr("config", config)
    trial.set_user_attr("results", results)
    
    # Immediately write this trial's results to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        row = {
            "trial_number": trial.number,
            "lr": config.get("lr"),
            "num_epochs": config.get("num_epochs"),
            "unfreeze_epoch": config.get("unfreeze_epoch"),
            "max_length": config.get("max_length"),
            "validation_accuracy": results.get("validation_accuracy"),
            "total_time": results.get("total_time")
        }
        writer.writerow(row)
    
    return results.get("validation_accuracy", 0)

# ---------------------
# Run the HPO Study
# ---------------------
if __name__ == "__main__":
    # Create a study to maximize the validation accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    # Print best trial details
    best_trial = study.best_trial
    print("\nBest trial:")
    print("  Validation Accuracy: ", best_trial.value)
    print("  Config: ", best_trial.user_attrs["config"])
    print("  Full Results: ", best_trial.user_attrs["results"])
    
    print(f"HPO results written to {csv_file}")
