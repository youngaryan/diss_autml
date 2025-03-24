import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from class_based_fixed_speech_brain import EmotionRecognitionTrainer
import os
import time
import csv

if __name__ == "__main__":
    # Configuration/hyperparameters (you can modify these to try different values)
    config = {
        "batch_size": 1,
        "lr": 1e-5,
        "num_epochs": 1,
        "unfreeze_epoch": 0,  # epoch to unfreeze feature extractor layers
        "max_length": 3 * 16000,  # 3 seconds at 16kHz
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Load dataset and prepare label mapping
    df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
    label_encoder_obj = LabelEncoder()
    df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
    mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
    print("Label mapping:", mapping)

    # Split into training and validation sets (e.g., 80/20 split)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize the trainer with the current hyperparameters
    trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)

    # Run training and measure overall time
    experiment_start = time.time()
    results = trainer.train()
    experiment_end = time.time()
    overall_experiment_time = experiment_end - experiment_start
    results["total_time"] = overall_experiment_time

    print("\nFinal Results:")
    print("Hyperparameters used:", results["hyperparameters"])
    print("Final Validation Accuracy:", results["validation_accuracy"])
    print("Total Training Time (s):", results["total_train_time"])
    print("Total Validation Time (s):", results["total_val_time"])
    print("Overall Time (s):", results["total_time"])

    # ---------------------------
    # Append results to res.csv file
    # ---------------------------
    csv_file = os.path.join(os.path.dirname(__file__), "res.csv")
    fieldnames = ["batch_size","lr","num_epochs" ,"unfreeze_epoch","max_length", "validation_accuracy", "train_time", "validation_time", "total_time"]
    row = {
        "batch_size": (results["hyperparameters"]["batch_size"]),
        "lr": (results["hyperparameters"]["lr"]),
        "num_epochs": (results["hyperparameters"]["num_epochs"]),
        "unfreeze_epoch": (results["hyperparameters"]["unfreeze_epoch"]),
        "max_length": (results["hyperparameters"]["max_length"]),
        "validation_accuracy": results["validation_accuracy"],
        "train_time": results["total_train_time"],
        "validation_time": results["total_val_time"],
        "total_time": results["total_time"]
    }
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to {csv_file}")