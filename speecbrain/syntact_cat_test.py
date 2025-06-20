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
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from data_preprocessing.dataset_speech_brain import EmotionDataset
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path_train = os.path.join(base_dir, "syntact_cat", "db.emotion.categories.train.desired.csv")
csv_path_test = os.path.join(base_dir, "syntact_cat", "db.emotion.categories.test.desired.csv")

df_pd_train = pd.read_csv(csv_path_train)
df_pd_test= pd.read_csv(csv_path_test)

print((len(df_pd_train)))
print((len(df_pd_test)))


df_pd = pd.concat([df_pd_test,df_pd_train], ignore_index=True)
df_pd = df_pd_test

label_emo = df_pd['emotion'].unique()
print(label_emo)

count_emo = df_pd["emotion"].value_counts()

print(count_emo)
print((len(df_pd)))



id2label = {
    0: "fear",
    1: "disgust",
    2: "happiness",
    3: "boredom",
    4: "neutral",
    5: "sadness",
    6: "anger"
}
label2id = {v: k for k, v in id2label.items()}



##############################
config = {
    "batch_size": 1,
    "lr": 2.2808811593092166e-05,
    "num_epochs": 9,
    "unfreeze_epoch": 0,  # Epoch at which to unfreeze the feature extractor
    "max_length": 80000,  # 3 seconds at 16kHz
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


class EmotionDataset(Dataset):
    def __init__(self, dataframe, base_dir=f"{base_dir}/syntact_cat", feature_extractor=None, max_length=48000, label_encoder=None):
        self.dataframe = dataframe
        self.base_dir = base_dir  # base path to prepend
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        # Full path = base_dir + file (e.g. base/synthesized_audio/de6_happy_...)
        file_path = os.path.join(self.base_dir, sample["file"])

        waveform, sample_rate = torchaudio.load(file_path)

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # mono

        label_str = sample["emotion"]
        label  = label2id[label_str]
        # if self.label_encoder:
        #     label = self.label_encoder.transform([label_str])[0]
        # else:
        #     label = int(label_str)  # fallback

        # Pad or truncate
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.size(0) < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.size(0)))

        return waveform, torch.tensor(label, dtype=torch.long)


# import joblib

# label_encoder_obj = joblib.load("label_encoder.joblib")  # or pickle
# Load fitted LabelEncoder
# print(label_encoder_obj.classes_)   # → array([0, 1, 2, 3, 4, 5, 6])
# print(label_encoder_obj.dtype)      # int64 (or similar)

# Now transform test dataset using same encoder
# df_pd["emotion"] = label_encoder_obj.transform(df_pd["emotion"])

# mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
# print("Label mapping:", mapping)
# num_classes = len(mapping)  # E







# Load model
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": config["device"]}
)

# Patch the output layer to match training-time structure (you used nn.Linear)
in_features = model.mods.output_mlp.w.in_features  # access .w to get input dim
model.mods.output_mlp = nn.Linear(in_features, 7)  # or whatever your num_classes is

# Now it will accept standard Linear weights
model.load_state_dict(torch.load("best_fine_tuned_model_state_dict.pt"))
model.to(config["device"]).eval()

# ---------------------------
# Validate on Entire RAVDESS
# ---------------------------
dataset = EmotionDataset(df_pd, feature_extractor=None, max_length=config["max_length"], label_encoder=label2id)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

print(model.hparams.label_encoder)  # stores class names used during fine-tuning
# print(label_encoder_obj.classes_)


test_loss, test_accuracy, raw_acc = validate(model, dataloader, criterion, config["device"])
print(f"✅ RAVDESS Test Loss: {test_loss:.4f}, Test Accuracy: {raw_acc:.4f}, bca {test_accuracy:.4f}")

#['happiness' 'anger' 'sadness' 'neutral' 'boredom' 'fear']