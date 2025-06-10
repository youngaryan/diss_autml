import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain.inference import EncoderClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.dataset_speech_brain import EmotionDataset
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score

# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "max_length": 64000,
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
    return avg_loss, bca,  accuracy

# ---------------------------
# Load and Preprocess RAVDESS
# ---------------------------
# df = load_dataset("confit/ravdess-parquet", "fold1")
# df = pd.concat([df["train"].to_pandas(), df["test"].to_pandas()], ignore_index=True)
# df = df[~df["emotion"].isin(["surprised"])].reset_index(drop=True)










df1 = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
# 
# REVDESS TEST
# 
# from datasets import load_dataset

# df = load_dataset("confit/ravdess-parquet", "fold1")
# df_t = df["train"].to_pandas()
# df_tes = df["test"].to_pandas()
# df = pd.concat([df_t, df_tes], ignore_index=True)
# df = df[~df["emotion"].isin(["calm", "surprised"])].reset_index(drop=True)


#####################################

# Use LabelEncoder for robustness
label_encoder_obj = LabelEncoder()
df1["emotion"] = label_encoder_obj.fit_transform(df1["emotion"])
# Print mapping: original label -> encoded label
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)
num_classes = len(mapping)  # E








# Encode emotions
# label_encoder_obj = LabelEncoder()
# df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
# mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
# print("Label mapping:", mapping)
# num_classes = len(mapping)

#---------------------------
# Load Fine-Tuned Model
# ---------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": config["device"]}
)

# Adjust final classifier head and load weights
in_features = model.mods.output_mlp.w.in_features
model.mods.output_mlp = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("fine_tuned_model_state_dict.pt"))
model.to(config["device"])
model.eval()

# ---------------------------
# Validate on Entire RAVDESS
# ---------------------------
dataset = EmotionDataset(df1, feature_extractor=None, max_length=config["max_length"], label_encoder=mapping)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

test_loss, test_accuracy, raw_acc = validate(model, dataloader, criterion, config["device"])
print(f"✅ RAVDESS Test Loss: {test_loss:.4f}, Test Accuracy: {raw_acc:.4f}, bca {test_accuracy:.4f}")