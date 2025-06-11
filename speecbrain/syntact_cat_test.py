import os
import pandas as pd
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from speechbrain.inference import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import io, sys
from datasets import load_dataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.dataset_speech_brain import EmotionDataset
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path_train = os.path.join(base_dir, "syntact_cat", "db.emotion.categories.train.desired.csv")
csv_path_test = os.path.join(base_dir, "syntact_cat", "db.emotion.categories.test.desired.csv")

df_pd_train = pd.read_csv(csv_path_train)
df_pd_test= pd.read_csv(csv_path_test)

print((len(df_pd_train)))
print((len(df_pd_test)))


df_pd = pd.concat([df_pd_test,df_pd_train], ignore_index=True)

label_emo = df_pd['emotion'].unique()
print(label_emo)

count_emo = df_pd["emotion"].value_counts()

print(count_emo)
print((len(df_pd)))



##############################
config = {
    "batch_size": 1,
    "lr": 7.960917579180225e-06,
    "num_epochs": 9,
    "unfreeze_epoch": 5,  # Epoch at which to unfreeze the feature extractor
    "max_length": 64000,  # 3 seconds at 16kHz
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}




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
    """Return avg-loss, balanced-accuracy, raw-accuracy."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # ↓↓↓ single call does feature extraction, pooling, and MLP ↓↓↓
            logits, _ = model(inputs)          # logits = (batch, num_classes)

            loss = criterion(logits, targets)
            total_loss     += loss.item()
            total_correct  += (logits.argmax(1) == targets).sum().item()
            total_samples  += targets.size(0)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    raw_acc  = total_correct / total_samples if total_samples else 0
    bca      = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, bca, raw_acc


# ---------------------------
# Load and Preprocess RAVDESS
# ---------------------------
# df = load_dataset("confit/ravdess-parquet", "fold1")
# df = pd.concat([df["train"].to_pandas(), df["test"].to_pandas()], ignore_index=True)
# df = df[~df["emotion"].isin(["surprised"])].reset_index(drop=True)




# label_encoder_obj = LabelEncoder()
# df_pd["emotion"] = label_encoder_obj.fit_transform(df_pd["emotion"])
# # Print mapping: original label -> encoded label
# mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
# print("Label mapping:", mapping)
# num_classes = len(mapping)  # E



import joblib

label_encoder_obj = joblib.load("label_encoder.joblib")  # or pickle
# Load fitted LabelEncoder

# Now transform test dataset using same encoder
df_pd["emotion"] = label_encoder_obj.transform(df_pd["emotion"])

mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)
num_classes = len(mapping)  # E









model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": config["device"]},
)
model.load_state_dict(torch.load("fine_tuned_model_state_dict.pt"))
model.eval().to(config["device"])

# ---------------------------
# Validate on Entire RAVDESS
# ---------------------------
dataset = EmotionDataset(df_pd, feature_extractor=None, max_length=config["max_length"], label_encoder=label_encoder_obj)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

test_loss, test_accuracy, raw_acc = validate(model, dataloader, criterion, config["device"])
print(f"✅ RAVDESS Test Loss: {test_loss:.4f}, Test Accuracy: {raw_acc:.4f}, bca {test_accuracy:.4f}")

#['happiness' 'anger' 'sadness' 'neutral' 'boredom' 'fear']