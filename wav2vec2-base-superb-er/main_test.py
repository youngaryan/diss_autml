import os
import time
import csv
import itertools
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoFeatureExtractor
from trainer_hf import EmotionRecognitionTrainerHF  # Your updated trainer class

if __name__ == "__main__":
    # Grid search values
    lr_values = [1e-5, 1e-6]
    num_epochs_values = [1, 3]
    max_length_values = [2 * 16000, 3 * 16000]  # seconds * sample_rate

    # Constants
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["emotion"])

    # Train/val split
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Output CSV
    csv_file = os.path.join(os.path.dirname(__file__), "result_superb.csv")
    fieldnames = [
        "batch_size", "lr", "num_epochs", "max_length",
        "validation_accuracy", "train_time", "validation_time", "total_time"
    ]
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for lr, num_epochs, max_length in itertools.product(lr_values, num_epochs_values, max_length_values):
            config = {
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "max_length": max_length,
                "device": device
            }

            print(f"\nRunning experiment with config: {config}")
            trainer = EmotionRecognitionTrainerHF(config, train_df, valid_df, label_encoder)

            experiment_start = time.time()
            results = trainer.train()
            experiment_end = time.time()
            results["total_time"] = experiment_end - experiment_start

            # Display results
            print("Results for config:", config)
            print("  Validation Accuracy:", results.get("validation_accuracy"))
            print("  Total Training Time (s):", results.get("total_train_time"))
            print("  Total Validation Time (s):", results.get("total_val_time"))
            print("  Overall Experiment Time (s):", results.get("total_time"))

            # Write to CSV
            row = {
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "num_epochs": config["num_epochs"],
                "max_length": config["max_length"],
                "validation_accuracy": results.get("validation_accuracy"),
                "train_time": results.get("total_train_time"),
                "validation_time": results.get("total_val_time"),
                "total_time": results.get("total_time")
            }
            writer.writerow(row)
            f.flush()
            print(f"Results appended to {csv_file}")
