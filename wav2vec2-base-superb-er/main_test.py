# =============================
# File: dataset_superb.py
# =============================

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class EmotionDatasetHF(Dataset):
    def __init__(self, df, feature_extractor, label_encoder, sampling_rate=16000, max_length=None):
        self.df = df
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.sampling_rate = sampling_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["path"])
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        # Crop to max_length if specified
        if self.max_length is not None:
            waveform = waveform[:, :self.max_length]

        return {
            "input_values": waveform.squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    padded_inputs = pad_sequence(input_values, batch_first=True)
    return padded_inputs, labels


# =============================
# File: trainer_hf.py
# =============================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.dataset_superb import EmotionDatasetHF, collate_fn

class EmotionRecognitionTrainerHF:
    def __init__(self, config, train_df, valid_df, label_encoder):
        self.config = config
        self.device = config["device"]
        self.label_encoder = label_encoder

        # Load pretrained model and feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        self.model = AutoModelForAudioClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er",
            num_labels=len(label_encoder.classes_),
            ignore_mismatched_sizes=True
        ).to(self.device)

        # Prepare datasets
        self.train_dataset = EmotionDatasetHF(
            train_df, feature_extractor=self.feature_extractor, label_encoder=label_encoder, max_length=config["max_length"]
        )
        self.valid_dataset = EmotionDatasetHF(
            valid_df, feature_extractor=self.feature_extractor, label_encoder=label_encoder, max_length=config["max_length"]
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
        )

        # Optimizer, loss, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["num_epochs"])

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).logits
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(self.valid_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return avg_loss, accuracy

    def train(self):
        total_train_time = 0.0
        total_val_time = 0.0

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            epoch_start = time.time()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs).logits
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += targets.size(0)

                print(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

            epoch_train_time = time.time() - epoch_start
            total_train_time += epoch_train_time
            print(f"Epoch [{epoch+1}] Training Time: {epoch_train_time:.2f}s")

            val_start = time.time()
            val_loss, val_accuracy = self.validate()
            val_time = time.time() - val_start
            total_val_time += val_time

            self.scheduler.step()

            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Avg Train Loss: {total_loss / len(self.train_loader):.4f}, Train Acc: {total_correct / total_samples:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Time: {val_time:.2f}s")
            print("-" * 50)

        return {
            "hyperparameters": self.config,
            "validation_accuracy": val_accuracy,
            "total_train_time": total_train_time,
            "total_val_time": total_val_time,
            "total_time": total_train_time + total_val_time
        }


# =============================
# File: main_superb_trainer.py
# =============================

import os
import time
import csv
import itertools
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from trainer_hf import EmotionRecognitionTrainerHF

if __name__ == "__main__":
    lr_values = [1e-5, 1e-6]
    num_epochs_values = [1, 3]
    max_length_values = [2 * 16000, 3 * 16000]

    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["emotion"])

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

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

            print("Results for config:", config)
            print("  Validation Accuracy:", results.get("validation_accuracy"))
            print("  Total Training Time (s):", results.get("total_train_time"))
            print("  Total Validation Time (s):", results.get("total_val_time"))
            print("  Overall Experiment Time (s):", results.get("total_time"))

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