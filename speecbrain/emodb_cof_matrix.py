import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain.inference import EncoderClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_speech_brain import EmotionDataset

# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "lr": 2.2808811593092166e-05,
    "num_epochs": 9,
    "unfreeze_epoch": 0,
    "max_length": 80000,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ---------------------------
# Helper Functions
# ---------------------------
def fix_input_shape(inputs: torch.Tensor) -> torch.Tensor:
    inputs = inputs.squeeze()
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    return inputs

def fix_feature_shape(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        features = features.unsqueeze(0)
    return features

def validate(model, dataloader, criterion, device):
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
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    bca = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, bca, accuracy, all_preds, all_targets

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")

# Encode emotion labels
label_encoder_obj = LabelEncoder()
df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)
num_classes = len(mapping)

# ---------------------------
# Split into Train and Validation Set
# ---------------------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emotion"])

# ---------------------------
# Load Model
# ---------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": config["device"]}
)

# Replace classifier head
in_features = model.mods.output_mlp.w.in_features
model.mods.output_mlp = nn.Linear(in_features, num_classes)
# model.load_state_dict(torch.load("best_fine_tuned_model_state_dict.pt"))
model.to(config["device"])
model.eval()

# ---------------------------
# Create Validation Dataset and Dataloader
# ---------------------------
val_dataset = EmotionDataset(val_df, feature_extractor=None, max_length=config["max_length"], label_encoder=mapping)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

# ---------------------------
# Run Validation and Plot Confusion Matrix
# ---------------------------
val_loss, bca, acc, preds, targets = validate(model, val_dataloader, criterion, config["device"])
print(f"✅ Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}, BCA: {bca:.4f}")

cm = confusion_matrix(targets, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder_obj.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on Validation Set (EMODB) BASELINE SPEECHBRAIN")
plt.tight_layout()
plt.savefig("confusion_matrix_validation.png")
# plt.show()
