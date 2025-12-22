# achieves 90% accuracy

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Removed SpeechBrain EncoderClassifier imports
from transformers import AutoModel, AutoConfig # <--- ADDED transformers
from sklearn.metrics import balanced_accuracy_score
import joblib
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import os

# Preserve your local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statics import SEED
from data_preprocessing.dataset_speech_brain import EmotionDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Random Seed Setup
# ---------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "lr": 1e-5,  # Slightly increased for HuBERT fine-tuning stability
    "num_epochs": 5,
    "unfreeze_epoch": 2, 
    "max_length": 80000, # Ensure your dataset resamples to 16kHz
    "model_name": "ALM/hubert-base-audioset", # <--- NEW MODEL
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

g = torch.Generator()
g.manual_seed(SEED)

# ---------------------------
# Custom Model Class (Wrapper)
# ---------------------------
class HubertEmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, freeze_base=True):
        super().__init__()
        # Load the pre-trained HuBERT model
        self.hubert = AutoModel.from_pretrained(model_name)
        
        # Determine hidden size (usually 768 for base models)
        hidden_size = self.hubert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights for the classifier
        nn.init.xavier_uniform_(self.classifier.weight)

        # Freeze the base model initially
        if freeze_base:
            for param in self.hubert.parameters():
                param.requires_grad = False

    def forward(self, input_values):
        # HuBERT expects input shape: [Batch, Time]
        # It returns a BaseOutput object
        outputs = self.hubert(input_values)
        
        # last_hidden_state shape: [Batch, Time, Hidden]
        last_hidden_state = outputs.last_hidden_state
        
        # Mean Pooling over time dimension
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits

# ---------------------------
# Helper Functions
# ---------------------------
def fix_input_shape(inputs: torch.Tensor) -> torch.Tensor:
    """Ensure input tensor has a batch dimension [Batch, Time]."""
    inputs = inputs.squeeze()
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    return inputs

def validate(model, dataloader, criterion, device):
    """Run validation and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = fix_input_shape(inputs)
            
            # --- MODIFIED: Simplified Forward Pass ---
            predictions = model(inputs)
            # -----------------------------------------
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            total_correct += (predictions.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    bca = balanced_accuracy_score(all_targets, all_preds)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy, bca, all_preds, all_targets

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
device = config["device"]
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")

label_encoder_obj = LabelEncoder()
label_encoder_obj.fit(df["emotion"])
joblib.dump(label_encoder_obj, "label_encoder.joblib")

mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)
num_classes = len(mapping)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = EmotionDataset(train_df, feature_extractor=None, max_length=config["max_length"], label_encoder=label_encoder_obj)
valid_dataset = EmotionDataset(valid_df, feature_extractor=None, max_length=config["max_length"], label_encoder=label_encoder_obj)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, generator=g)

# ---------------------------
# Initialize New Model
# ---------------------------
print(f"Loading model: {config['model_name']}...")
model = HubertEmotionClassifier(config["model_name"], num_classes=num_classes)
model.to(device)

# ---------------------------
# Training Setup
# ---------------------------
criterion = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

# ---------------------------
# Training Loop
# ---------------------------
best_val_accuracy = 0.0
best_model_state_dict = None
all_preds = []
all_targets = []

for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Unfreeze logic
    if epoch == config["unfreeze_epoch"]:
        print(f"Epoch {epoch + 1}: Unfreezing feature extractor layers...")
        # Unfreeze the HuBERT base model
        for param in model.hubert.parameters():
            param.requires_grad = True
            
        # Reinitialize optimizer with new parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"] - epoch)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = fix_input_shape(inputs)
        
        optimizer.zero_grad()

        # --- MODIFIED: Simplified Forward Pass ---
        predictions = model(inputs)
        # -----------------------------------------

        loss = criterion(predictions, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (predictions.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)
        
        if batch_idx % 50 == 0:
             print(f"Epoch [{epoch+1}/{config['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    scheduler.step()
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Avg Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")

    # Validation
    val_loss, val_accuracy, bca, all_preds1, all_targets1 = validate(model, valid_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    if bca > best_val_accuracy:
        best_val_accuracy = bca
        best_model_state_dict = model.state_dict()
        all_preds = all_preds1
        all_targets = all_targets1
        print(f"New best model found at epoch {epoch+1} with val accuracy {val_accuracy:.4f}")

# ---------------------------
# Confusion Matrix & Saving
# ---------------------------
EMODB_LABELS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']

cm = confusion_matrix(all_targets, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMODB_LABELS)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix on EMODB Validation SET (Predicted vs True)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_EMODB_Validation_SET.png")
print("📊 Saved confusion matrix")

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, "best_fine_tuned_model_state_dict.pt")
    print("Best model saved.")
else:
    torch.save(model.state_dict(), "final_fine_tuned_model_state_dict.pt")
    print("Final model saved.")
