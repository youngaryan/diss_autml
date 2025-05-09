import os
import time
import csv
import itertools
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from class_based_fixed_speech_brain import EmotionRecognitionTrainer

if __name__ == "__main__":
    # Define grid values for each parameter you want to experiment with
    lr_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    num_epochs_values = [1, 3, 5, 7, 10]
    unfreeze_epoch_values = [0, 1, 2, 3, 4]
    max_length_values = [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000]  # seconds * sample_rate

    # Constant parameters
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and prepare label mapping
    df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
    label_encoder_obj = LabelEncoder()
    df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
    mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
    print("Label mapping:", mapping)

    # Split into training and validation sets (e.g., 80/20 split)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare CSV file to record experiment results
    csv_file = os.path.join(os.path.dirname(__file__), "result_exhaustive_bca.csv")
    fieldnames = [
        "batch_size", "lr", "num_epochs", "unfreeze_epoch", "max_length",
        "validation_accuracy", "train_time", "validation_time", "total_time"
    ]
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Iterate over all combinations of hyperparameters
        for lr, num_epochs, unfreeze_epoch, max_length in itertools.product(
            lr_values, num_epochs_values, unfreeze_epoch_values, max_length_values
        ):
            # Build configuration for the current experiment

            if num_epochs < unfreeze_epoch:
                continue
            config = {
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "unfreeze_epoch": unfreeze_epoch,
                "max_length": max_length,
                "device": device
            }
            print(f"\nRunning experiment with config: {config}")

            # Initialize trainer with the current hyperparameters
            trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)

            # Run training and time the experiment
            experiment_start = time.time()
            results = trainer.train()
            experiment_end = time.time()
            overall_experiment_time = experiment_end - experiment_start
            results["total_time"] = overall_experiment_time

            # Print experiment results
            print("Results for config:", config)
            print("  Validation Accuracy:", results.get("validation_accuracy"))
            print("  Total Training Time (s):", results.get("total_train_time"))
            print("  Total Validation Time (s):", results.get("total_val_time"))
            print("  Overall Experiment Time (s):", results.get("total_time"))

            # Append results to CSV file
            row = {
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "num_epochs": config["num_epochs"],
                "unfreeze_epoch": config["unfreeze_epoch"],
                "max_length": config["max_length"],
                "validation_accuracy": results.get("validation_accuracy"),
                "train_time": results.get("total_train_time"),
                "validation_time": results.get("total_val_time"),
                "total_time": results.get("total_time")
            }
            writer.writerow(row)
            f.flush()  # Ensure data is written immediately
            print(f"Results appended to {csv_file}")
