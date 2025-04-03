import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import time
import os
import sys

# Ensure correct relative import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_superb import EmotionDatasetHF, collate_fn

class EmotionRecognitionTrainerHF:
    def __init__(self, config, train_df, valid_df, label_encoder):
        """
        Args:
            config (dict): Training configuration.
            train_df (pd.DataFrame): Training DataFrame.
            valid_df (pd.DataFrame): Validation DataFrame.
            label_encoder (LabelEncoder): Fitted LabelEncoder for emotion labels.
        """
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
            train_df, feature_extractor=self.feature_extractor, label_encoder=label_encoder
        )
        self.valid_dataset = EmotionDatasetHF(
            valid_df, feature_extractor=self.feature_extractor, label_encoder=label_encoder
        )

        # DataLoaders with collate_fn for padding
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

            # Validation
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
