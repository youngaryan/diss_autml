# achivess 90% accuracy

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from speechbrain.inference import EncoderClassifier
from sklearn.metrics import balanced_accuracy_score
import joblib
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statics import SEED
from data_preprocessing.dataset_speech_brain import EmotionDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------
# Configuration
# ---------------------------
# 0,7.960917579180225e-06,9,5,64000,0.9301587301587302,1621.540283203125
# 1,2.2808811593092166e-05,9,0,80000,0.9611111111111111,156.26804518699646
config = {
    "batch_size": 1,
    "lr": 2.2808811593092166e-05,
    "num_epochs": 9,
    "unfreeze_epoch": 0,  # Epoch at which to unfreeze the feature extractor
    "max_length": 80000,  # 3 seconds at 16kHz
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(SEED)
# ---------------------------
# Helper Functions
# ---------------------------
def fix_input_shape(inputs: torch.Tensor) -> torch.Tensor:
    """Ensure input tensor has a batch dimension."""
    inputs = inputs.squeeze()
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    return inputs

def fix_feature_shape(features: torch.Tensor) -> torch.Tensor:
    """Ensure features have shape [B, T, H]."""
    if features.ndim == 2:
        features = features.unsqueeze(0)
    return features

def validate(model, dataloader, criterion, device):
    """Run validation and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds=[]
    all_targets=[]

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = fix_input_shape(inputs)
            features = model.mods.wav2vec2.extract_features(inputs)[0]
            features = fix_feature_shape(features)
            pooled = model.mods.avg_pool(features)
            if pooled.ndim == 1:
                pooled = pooled.unsqueeze(0)
            elif pooled.ndim == 3:
                pooled = pooled.squeeze(1)
            predictions = model.mods.output_mlp(pooled)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            total_correct += (predictions.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())


    avg_loss = total_loss / len(dataloader)
    bca = balanced_accuracy_score(all_targets, all_preds)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy, bca, all_preds,all_targets

# ---------------------------
# Device Setup
# ---------------------------
device = config["device"]

# ---------------------------
# Load Pretrained Model
# ---------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": device}
)

# Freeze feature extractor layers initially (e.g., those containing "wav2vec2")
for name, param in model.named_parameters():
    if "wav2vec2" in name:
        param.requires_grad = False

# ---------------------------
# Load and Preprocess Dataset
# ---------------------------
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")

#####################################
label_encoder_obj = LabelEncoder()
label_encoder_obj.fit(df["emotion"])  # <- Fit only

# Save the encoder for downstream use
joblib.dump(label_encoder_obj, "label_encoder.joblib")

# Optional: view mapping
print(dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_))))
# Use LabelEncoder for robustness
# Print mapping: original label -> encoded label
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)
num_classes = len(mapping)  # Expecting 7 classes

# Optionally split dataset into training and validation sets (e.g., 80/20 split)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------------------
# Adjust Final Classification Layer
# ---------------------------
# Replace final classification layer with one matching the number of classes
in_features = model.mods.output_mlp.w.in_features
model.mods.output_mlp = nn.Linear(in_features, num_classes)
# Initialize the new layer using Xavier uniform initialization
nn.init.xavier_uniform_(model.mods.output_mlp.weight)
model.to(device)

# ---------------------------
# Create Datasets and DataLoaders
# ---------------------------
train_dataset = EmotionDataset(train_df, feature_extractor=None, max_length=config["max_length"], label_encoder=label_encoder_obj)
valid_dataset = EmotionDataset(valid_df, feature_extractor=None, max_length=config["max_length"], label_encoder=label_encoder_obj)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, generator=g)

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
all_preds=[]
all_tragets=[]

for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Optionally unfreeze feature extractor layers at the designated epoch
    if epoch == config["unfreeze_epoch"]:
        print(f"Epoch {epoch + 1}: Unfreezing feature extractor layers...")
        for name, param in model.named_parameters():
            if "wav2vec2" in name:
                param.requires_grad = True
        # Reinitialize optimizer and scheduler with updated parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config["lr"])
        remaining_epochs = config["num_epochs"] - epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Fix input shape if needed
        inputs = fix_input_shape(inputs)
        
        # Forward pass: extract features and perform pooling
        features = model.mods.wav2vec2.extract_features(inputs)[0]
        features = fix_feature_shape(features)
        pooled = model.mods.avg_pool(features)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        elif pooled.ndim == 3:
            pooled = pooled.squeeze(1)
        
        predictions = model.mods.output_mlp(pooled)
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += (predictions.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    scheduler.step()
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

    # Run validation at the end of the epoch
    val_loss, val_accuracy,bca,all_preds1, all_tragets1 = validate(model, valid_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    if bca > best_val_accuracy:
        best_val_accuracy = bca
        best_model_state_dict = model.state_dict()
        all_preds=all_preds1
        all_tragets=all_tragets1
        print(f"New best model found at epoch {epoch+1} with val accuracy {val_accuracy:.4f}")


EMODB_LABELS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(all_tragets, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMODB_LABELS)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix on EMODB Validation SET (Predicted vs True)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_EMODB_Validation_SET.png")
print("📊 Saved confusion matrix as confusion_matrix_EMODB_Validation_SET.png")

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, "best_fine_tuned_model_state_dict.pt")
    print("Best model state_dict saved as 'best_fine_tuned_model_state_dict.pt'")
else:
    print("Warning: No best model was found; saving final model instead.")
    torch.save(model.state_dict(), "final_fine_tuned_model_state_dict.pt")

