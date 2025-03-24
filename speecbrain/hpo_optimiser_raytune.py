import os
import time
import csv
import pandas as pd
import torch
from ray import tune
import ray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from class_based_fixed_speech_brain import EmotionRecognitionTrainer

# Limit the number of threads used by libraries like OpenMP
os.environ["OMP_NUM_THREADS"] = "1"

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

# Constant parameter
batch_size = 1

# ---------------------
# Ray Tune Trainable Function
# ---------------------
def train_tune(config):
    # Determine device per trial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    print(f"\nRunning trial with config: {config}")
    
    # Initialize trainer with current hyperparameters
    trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)
    
    start_time = time.time()
    results = trainer.train()
    total_time = time.time() - start_time
    results["total_time"] = total_time

    # Report metrics to Ray Tune
    tune.report(validation_accuracy=results.get("validation_accuracy", 0), total_time=total_time)

    # ---------------------
    # Write trial results immediately to CSV
    # ---------------------
    csv_file = os.path.join(os.path.dirname(__file__), "hpo_results.csv")
    fieldnames = [
        "trial_number", "lr", "num_epochs", "unfreeze_epoch", "max_length",
        "validation_accuracy", "total_time"
    ]

    trial_id = tune.get_trial_id()
    row = {
        "trial_number": trial_id,
        "lr": config.get("lr"),
        "num_epochs": config.get("num_epochs"),
        "unfreeze_epoch": config.get("unfreeze_epoch"),
        "max_length": config.get("max_length"),
        "validation_accuracy": results.get("validation_accuracy"),
        "total_time": total_time
    }

    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()  # ensure immediate disk write

# ---------------------
# Run the HPO Study with Ray Tune
# ---------------------
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    search_space = {
        "lr": tune.loguniform(1e-6, 1e-3),
        "num_epochs": tune.randint(1, 11),
        "unfreeze_epoch": tune.randint(0, 6),
        "max_length": tune.choice([2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000]),
        "batch_size": batch_size,
    }

    analysis = tune.run(
        train_tune,
        config=search_space,
        num_samples=50,
        metric="validation_accuracy",
        mode="max",
        max_concurrent_trials=4,
        resources_per_trial={"cpu": 1, "gpu": 0}
    )

    best_trial = analysis.get_best_trial("validation_accuracy", "max", "last")
    print("\nBest trial:")
    print("  Validation Accuracy: ", best_trial.last_result.get("validation_accuracy"))
    print("  Config: ", best_trial.config)
    print("  Full Results: ", best_trial.last_result)
