import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from speechbrain.inference import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
import torchaudio
import os
import io
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np


# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "max_length": 64000,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ---------------------------
# Define EmotionDataset Class
# ---------------------------
class EmotionDataset(Dataset):
    def __init__(self, dataframe, feature_extractor=None, max_length=64000, label_encoder=None):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        # Decode audio bytes
        audio_bytes = sample["audio"]["bytes"]
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # Resample if needed
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0)  # Convert to mono

        # Pad or truncate
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            pad_length = self.max_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Label encoding
        label_str = sample["emotion"]
        label = self.label_encoder.transform([label_str])[0]

        return waveform, torch.tensor(label, dtype=torch.long)


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
    bca = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, bca, all_preds, all_targets

# ---------------------------
# Emotion Mapping & Label Encoding
# ---------------------------
EMODB_LABELS = ["fear", "disgust", "happiness", "boredom", "neutral", "sadness", "anger"]
label_encoder_obj = LabelEncoder()
label_encoder_obj.fit(EMODB_LABELS)

ravdess_to_emodb = {
    "angry": "anger",
    "disgust": "disgust",
    "fearful": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "calm": "boredom"  # optionally merge
}
valid_emotions = list(ravdess_to_emodb.keys())

# ---------------------------
# Load and Preprocess RAVDESS
# ---------------------------
dataset = load_dataset("confit/ravdess-parquet", "fold1")
df = pd.concat([dataset["train"].to_pandas(), dataset["test"].to_pandas()], ignore_index=True)
df = df[df["emotion"].isin(valid_emotions)].reset_index(drop=True)
df["emotion"] = df["emotion"].map(ravdess_to_emodb)

# ---------------------------
# Load Fine-Tuned EMO-DB Model
# ---------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",  # or your custom model
    savedir="pretrained_models/emodb_wav2vec2",
    run_opts={"device": config["device"]}
)
# Replace model head BEFORE loading state_dict
in_features = model.mods.output_mlp.w.weight.shape[1]  # still valid since HuggingFace loads full model
num_classes = len(EMODB_LABELS)
model.mods.output_mlp = nn.Linear(in_features, num_classes)

# Now safely load your custom EMO-DB-trained weights
#model.load_state_dict(torch.load("fine_tuned_model_state_dict.pt"))
model.load_state_dict(torch.load("best_fine_tuned_model_state_dict.pt"))
model.to(config["device"])
model.eval()

# model.load_state_dict(torch.load("fine_tuned_model_state_dict.pt"))
# model.to(config["device"])
# model.eval()

# ---------------------------
# Evaluation
# ---------------------------
dataset = EmotionDataset(df, label_encoder=label_encoder_obj, max_length=config["max_length"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

test_loss, test_bacc, all_preds, all_targets = validate(model, dataloader, criterion, config["device"])
print(f"✅ RAVDESS Generalization Test — Loss: {test_loss:.4f}, Balanced Accuracy: {test_bacc:.4f}")


# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(all_targets, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMODB_LABELS)
cm= cm[np.ix_(idx_order, idx_order)]
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix on RAVDESS (Predicted vs True)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_ravdess.png")
print("📊 Saved confusion matrix as confusion_matrix_ravdess.png")

#######################################################
emotion_counts = {label: 0 for label in EMODB_LABELS}
true_positives = {label: 0 for label in EMODB_LABELS}
for pred, target in zip(all_preds, all_targets):
    true_label = EMODB_LABELS[target]
    pred_label = EMODB_LABELS[pred]
    emotion_counts[true_label] += 1
    if pred == target:
        true_positives[true_label] += 1
labels = EMODB_LABELS
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, [emotion_counts[l] for l in labels], width, label='Total Instances')
bar2 = ax.bar(x + width/2, [true_positives[l] for l in labels], width, label='True Positives')

ax.set_ylabel('Count')
ax.set_title('Emotion-wise Total vs True Positives')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add counts on top
for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate(f'{height}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("emotion_true_positive_comparison.png")
print("📊 Saved bar chart as emotion_true_positive_comparison.png")