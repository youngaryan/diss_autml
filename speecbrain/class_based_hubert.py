import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import random
import sys
import os

# Append path to access your existing dataset code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_speech_brain import EmotionDataset
from statics import SEED

# ---------------------------
# Reproducibility Setup
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Wrapper for HuBERT + Classifier
# ---------------------------
class HubertClassifier(nn.Module):
    def __init__(self, model_name, num_classes, freeze_base=True):
        super().__init__()
        self.hubert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.hubert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

        if freeze_base:
            for param in self.hubert.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x shape: [batch, samples]
        outputs = self.hubert(x)
        # last_hidden_state: [batch, time, hidden]
        last_hidden_state = outputs.last_hidden_state
        # Mean pooling
        pooled = torch.mean(last_hidden_state, dim=1)
        logits = self.classifier(pooled)
        return logits

# ---------------------------
# The Trainer Class
# ---------------------------
class EmotionRecognitionTrainer:
    def __init__(self, config, train_df, valid_df, mapping):
        self.config = config
        self.train_df = train_df
        self.valid_df = valid_df
        self.mapping = mapping
        self.device = config["device"]
        self.num_classes = len(mapping)
        
        # Ensure seed consistency
        set_seed(SEED)
        
        # Initialize DataLoaders
        # Note: Ensure your EmotionDataset handles the max_length logic correctly for the new model
        # HuBERT expects 16kHz audio.
        self.train_dataset = EmotionDataset(
            self.train_df, 
            feature_extractor=None, 
            max_length=config["max_length"], 
            label_encoder=None # Labels are already encoded in the HPO script
        )
        self.valid_dataset = EmotionDataset(
            self.valid_df, 
            feature_extractor=None, 
            max_length=config["max_length"], 
            label_encoder=None
        )

        # Generator for dataloader reproducibility
        g = torch.Generator()
        g.manual_seed(SEED)

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            generator=g
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=config["batch_size"], 
            shuffle=False, 
            generator=g
        )

    def train(self):
        # Initialize Model
        # using the ALM/hubert-base-audioset model
        model = HubertClassifier("ALM/hubert-base-audioset", self.num_classes, freeze_base=True)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        
        # Initial Optimizer (only for classifier)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["num_epochs"])

        best_bca = 0.0
        
        for epoch in range(self.config["num_epochs"]):
            model.train()
            
            # Unfreezing Logic
            if epoch == self.config["unfreeze_epoch"]:
                print(f"   [Epoch {epoch}] Unfreezing HuBERT base...")
                for param in model.hubert.parameters():
                    param.requires_grad = True
                
                # Re-setup optimizer for all parameters
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=self.config["lr"])
                # Adjust scheduler for remaining epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config["num_epochs"] - epoch
                )

            # Training Loop
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Fix shape if necessary [Batch, Time]
                inputs = inputs.squeeze()
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Validation Loop
            val_loss, val_acc, val_bca = self.validate(model, criterion)
            
            # Track Best Performance (Balanced Accuracy)
            if val_bca > best_bca:
                best_bca = val_bca

        # Return the best metric and total time placeholder (calculated in main script)
        return {
            "validation_accuracy": best_bca, # Using Balanced Accuracy as the target metric
            "last_loss": val_loss
        }

    def validate(self, model, criterion):
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.squeeze()
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.valid_loader)
        acc = (np.array(all_preds) == np.array(all_targets)).mean()
        bca = balanced_accuracy_score(all_targets, all_preds)
        
        return avg_loss, acc, bca
