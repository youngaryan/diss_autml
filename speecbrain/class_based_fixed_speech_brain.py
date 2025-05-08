import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain.inference import EncoderClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import sys
import os
import time

# Ensure correct relative import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_speech_brain import EmotionDataset

class EmotionRecognitionTrainer:
    def __init__(self, config, train_df, valid_df, mapping):
        """
        Initialize the trainer with hyperparameters, data, and class mapping.
        
        Args:
            config (dict): Hyperparameters including batch_size, lr, num_epochs, etc.
            train_df (pd.DataFrame): DataFrame for training data.
            valid_df (pd.DataFrame): DataFrame for validation data.
            mapping (dict): Mapping for labels (e.g. from LabelEncoder).
        """
        self.config = config
        self.device = config["device"]
        self.mapping = mapping

        # ---------------------------
        # Load and configure the model
        # ---------------------------
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="pretrained_models/emotion_recognition",
            run_opts={"device": self.device}
        )
        # Freeze feature extractor layers initially
        for name, param in self.model.named_parameters():
            if "wav2vec2" in name:
                param.requires_grad = False

        # Adjust the final classification layer to match the number of classes
        in_features = self.model.mods.output_mlp.w.in_features
        num_classes = len(mapping)
        self.model.mods.output_mlp = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.model.mods.output_mlp.weight)
        self.model.to(self.device)
        
        # ---------------------------
        # Create datasets and dataloaders
        # ---------------------------
        self.train_dataset = EmotionDataset(
            train_df, feature_extractor=None, max_length=config["max_length"], label_encoder=mapping
        )
        self.valid_dataset = EmotionDataset(
            valid_df, feature_extractor=None, max_length=config["max_length"], label_encoder=mapping
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # ---------------------------
        # Set up loss, optimizer, and scheduler
        # ---------------------------
        self.criterion = nn.CrossEntropyLoss()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["num_epochs"])

    def fix_input_shape(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor has a batch dimension."""
        inputs = inputs.squeeze()
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        return inputs

    def fix_feature_shape(self, features: torch.Tensor) -> torch.Tensor:
        """Ensure features have shape [B, T, H]."""
        if features.ndim == 2:
            features = features.unsqueeze(0)
        return features

    # def validate(self):
    #     """Run validation and return average loss and accuracy."""
    #     self.model.eval()
    #     total_loss = 0.0
    #     total_correct = 0
    #     total_samples = 0

    #     with torch.no_grad():
    #         for inputs, targets in self.valid_loader:
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             inputs = self.fix_input_shape(inputs)
    #             features = self.model.mods.wav2vec2.extract_features(inputs)[0]
    #             features = self.fix_feature_shape(features)
    #             pooled = self.model.mods.avg_pool(features)
    #             if pooled.ndim == 1:
    #                 pooled = pooled.unsqueeze(0)
    #             elif pooled.ndim == 3:
    #                 pooled = pooled.squeeze(1)
    #             predictions = self.model.mods.output_mlp(pooled)
    #             loss = self.criterion(predictions, targets)
    #             total_loss += loss.item()
    #             total_correct += (predictions.argmax(dim=1) == targets).sum().item()
    #             total_samples += targets.size(0)
    #     avg_loss = total_loss / len(self.valid_loader)
    #     accuracy = total_correct / total_samples if total_samples > 0 else 0
    #     return avg_loss, accuracy

    def validate(self):
        """Run validation and return average loss and balanced accuracy."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = self.fix_input_shape(inputs)
                features = self.model.mods.wav2vec2.extract_features(inputs)[0]
                features = self.fix_feature_shape(features)
                pooled = self.model.mods.avg_pool(features)
                if pooled.ndim == 1:
                    pooled = pooled.unsqueeze(0)
                elif pooled.ndim == 3:
                    pooled = pooled.squeeze(1)
                predictions = self.model.mods.output_mlp(pooled)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

                all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.valid_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        return avg_loss, balanced_acc


    def train(self):
        """
        Run the training loop while measuring the time taken for training and validation.
        Returns:
            dict: A dictionary containing hyperparameters, final validation accuracy,
                  total training time, total validation time, and overall time.
        """
        total_train_time = 0.0
        total_val_time = 0.0
        
        for epoch in range(self.config["num_epochs"]):
            epoch_start_time = time.time()
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Optionally unfreeze feature extractor layers at a designated epoch
            if epoch == self.config["unfreeze_epoch"]:
                print(f"Epoch {epoch+1}: Unfreezing feature extractor layers...")
                for name, param in self.model.named_parameters():
                    if "wav2vec2" in name:
                        param.requires_grad = True
                # Reinitialize optimizer and scheduler with updated parameters
                params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.AdamW(params, lr=self.config["lr"])
                remaining_epochs = self.config["num_epochs"] - epoch
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=remaining_epochs)
            
            # ---------------------------
            # Training Phase
            # ---------------------------
            train_epoch_start = time.time()
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Prepare input and extract features
                inputs = self.fix_input_shape(inputs)
                features = self.model.mods.wav2vec2.extract_features(inputs)[0]
                features = self.fix_feature_shape(features)
                pooled = self.model.mods.avg_pool(features)
                if pooled.ndim == 1:
                    pooled = pooled.unsqueeze(0)
                elif pooled.ndim == 3:
                    pooled = pooled.squeeze(1)
                predictions = self.model.mods.output_mlp(pooled)
                loss = self.criterion(predictions, targets)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += (predictions.argmax(dim=1) == targets).sum().item()
                total_samples += targets.size(0)
                print(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
            train_epoch_end = time.time()
            epoch_train_time = train_epoch_end - train_epoch_start
            total_train_time += epoch_train_time
            print(f"Epoch [{epoch+1}] Training Time: {epoch_train_time:.2f} seconds")
            
            # ---------------------------
            # Validation Phase
            # ---------------------------
            val_epoch_start = time.time()
            val_loss, val_accuracy = self.validate()
            val_epoch_end = time.time()
            epoch_val_time = val_epoch_end - val_epoch_start
            total_val_time += epoch_val_time
            
            self.scheduler.step()
            epoch_loss = total_loss / len(self.train_loader)
            epoch_accuracy = total_correct / total_samples
            epoch_end_time = time.time()
            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Average Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
            print(f"Epoch [{epoch+1}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print(f"Epoch [{epoch+1}] Validation Time: {epoch_val_time:.2f} seconds")
            print("-" * 50)
        
        overall_time = total_train_time + total_val_time
        # Return results along with timing details and hyperparameters used
        return {
            "hyperparameters": self.config,
            "validation_accuracy": val_accuracy,
            "total_train_time": total_train_time,
            "total_val_time": total_val_time,
            "total_time": overall_time
        }