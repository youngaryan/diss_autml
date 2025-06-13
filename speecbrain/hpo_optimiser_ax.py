import os
import time
import numpy as np
import csv
import random
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from class_based_fixed_speech_brain import EmotionRecognitionTrainer

# Import Ax client
from ax.service.ax_client import AxClient

SEED = 42
random.seed(SEED);
np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# torch.use_deterministic_algorithms(True)

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
# Ax Experiment Setup
# ---------------------
ax_client = AxClient(random_seed=SEED)
ax_client.create_experiment(
    name="emotion_recognition_experiment",
    parameters=[
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-6, 1e-3],
            "log_scale": True,
        },
        {
            "name": "num_epochs",
            "type": "range",
            "bounds": [1, 10],
            "value_type": "int",
        },
        {
            "name": "unfreeze_epoch",
            "type": "range",
            "bounds": [0, 5],
            "value_type": "int",
        },
        {
            "name": "max_length",
            "type": "choice",
            "values": [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000],
        },
    ],
)

# Number of HPO trials
n_trials = 50

# Prepare CSV output
csv_file = os.path.join(os.path.dirname(__file__), "hpo_results_ax_bca1_1.csv")
fieldnames = [
    "trial_number", "lr", "num_epochs", "unfreeze_epoch", "max_length",
    "validation_accuracy", "total_time"
]
# Write header if file doesn't exist or is empty
file_exists = os.path.exists(csv_file)
if not file_exists or os.path.getsize(csv_file) == 0:
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# ---------------------
# HPO Loop using Ax
# ---------------------
for i in range(n_trials):
    # Get the next set of hyperparameters to evaluate
    parameters, trial_index = ax_client.get_next_trial()

    # Build config from suggested parameters
    config = {
        "batch_size": batch_size,
        "lr": parameters["lr"],
        "num_epochs": parameters["num_epochs"],
        "unfreeze_epoch": parameters["unfreeze_epoch"],
        "max_length": parameters["max_length"],
        "device": device,
    }
    print(f"\nRunning trial {trial_index} with config: {config}")

    # Initialize trainer with current hyperparameters
    trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)
    start_time = time.time()
    results = trainer.train()
    total_time = time.time() - start_time
    results["total_time"] = total_time

    # Get the validation accuracy (defaulting to 0 if not present)
    validation_accuracy = results.get("validation_accuracy", 0)

    # Report the result back to Ax (as negative to maximize accuracy)
    ax_client.complete_trial(trial_index=trial_index, raw_data=-validation_accuracy)

    # Immediately write to CSV
    trial_result = {
        "trial_number": trial_index,
        "lr": config["lr"],
        "num_epochs": config["num_epochs"],
        "unfreeze_epoch": config["unfreeze_epoch"],
        "max_length": config["max_length"],
        "validation_accuracy": validation_accuracy,
        "total_time": total_time,
    }

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(trial_result)

# ---------------------
# Retrieve and Print Best Trial Details
# ---------------------
# best_parameters, best_trial_index = ax_client.get_best_trial()
# print("\nBest trial:")
# print("  Trial Number: ", best_trial_index)
# print("  Best Parameters: ", best_parameters)

best_parameters, _ = ax_client.get_best_parameters()
best_trial_index = ax_client.experiment.best_trial.index

print("\nBest trial:")
print("  Trial Number: ", best_trial_index)
print("  Best Parameters: ", best_parameters)

# Fetch experiment data and invert the sign to get the true best validation accuracy.
experiment_data = ax_client.experiment.fetch_data().df
best_val_accuracy = -experiment_data["mean"].min()
print("  Best Validation Accuracy: ", best_val_accuracy)
