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
import io
from datasets import load_dataset

# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "lr": 2.2808811593092166e-05,
    "num_epochs": 9,
    "unfreeze_epoch": 0,  # Epoch at which to unfreeze the feature extractor
    "max_length": 80000,  # 3 seconds at 16kHz
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ---------------------------
# EMO-DB Emotion Labels
# ---------------------------
EMODB_LABELS = ["fear", "disgust", "happiness", "boredom", "neutral", "sadness", "anger"]
label_encoder = LabelEncoder()
label_encoder.fit(EMODB_LABELS)

# ---------------------------
# TESS to EMO-DB Mapping
# ---------------------------
# crema_to_emodb = {
#     0: "anger",
#     1: "disgust",
#     2: "fear",
#     3: "happiness",
#     4: "neutral",
#     5: "sadness",
# }
crema_to_emodb = {
    'neutral': 'neutral',
    'happy': 'happiness',
    'sad': 'sadness',
    'anger': 'anger',
    'fear': 'fear',
    'disgust': 'disgust',
    "calm": "boredom"  # optionally merge
}

# ---------------------------
# Load TESS from Hugging Face
# ---------------------------


df =load_dataset("myleslinder/crema-d",trust_remote_code=True)


if "train" not in df or len(df["train"]) == 0:
    raise ValueError("❌ Dataset is empty or missing 'train' split. Check Hugging Face download or internet connection.")

train_dataset = df["train"]
print(df["train"].features["label"])
# Convert to pandas DataFrame for processing
df_pd = train_dataset.to_pandas()

# Map labels
# df_pd["emotion"] = df_pd["label"].map(crema_to_emodb)

# print(df_pd["label"].unique())  # See what labels you actually have
# print(len(df_pd)) 
# # Drop any unmapped emotions
# df_pd = df_pd[df_pd["emotion"].notnull()].reset_index(drop=True)


label_names = df["train"].features["label"].names
df_pd["emotion_raw"] = df_pd["label"].map(lambda x: label_names[x])

# Now map to EMO-DB-compatible labels
df_pd["emotion"] = df_pd["emotion_raw"].map(crema_to_emodb)

# Drop unmapped
df_pd = df_pd[df_pd["emotion"].notnull()].reset_index(drop=True)
# ---------------------------
# Define TESSDataset
# ---------------------------
import tempfile

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

        # ✅ Use the file path provided by the Hugging Face dataset
        audio_path = sample["audio"]["path"]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0)  # Convert stereo → mono

        # Pad or truncate to fixed length
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            pad_length = self.max_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Encode the label
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
    raw_acc = total_correct / total_samples
    return avg_loss, bca, all_preds, all_targets, raw_acc

# ---------------------------
# Load Pretrained Model
# ---------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emodb_wav2vec2",
    run_opts={"device": config["device"]}
)

# Replace classification head
in_features = model.mods.output_mlp.w.weight.shape[1]
model.mods.output_mlp = nn.Linear(in_features, len(EMODB_LABELS))

# model.load_state_dict(torch.load("best_fine_tuned_model_state_dict.pt"))
model.to(config["device"])
model.eval()

# ---------------------------
# Run Evaluation
# ---------------------------
dataset = EmotionDataset(df_pd, label_encoder=label_encoder, max_length=config["max_length"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

test_loss, test_bacc, all_preds, all_targets, raw_acc = validate(model, dataloader, criterion, config["device"])
print(f"✅ TESS Generalization Test — Loss: {test_loss:.4f}, Balanced Accuracy: {test_bacc:.4f}, raw Acc {raw_acc:.4f}")

# ---------------------------
# Plot Confusion Matrix
# ---------------------------
cm = confusion_matrix(all_targets, all_preds, normalize='true')
idx_order = [label_encoder.transform([label])[0] for label in EMODB_LABELS]
cm= cm[np.ix_(idx_order, idx_order)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMODB_LABELS)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix on CREMA-D (Predicted vs True)- Zero-shot speechbrain")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_CREMA-D_ZeroShot.png")
print("📊 Saved confusion matrix as confusion_matrix_CREMA-D.png")

# ---------------------------
# Plot True Positives per Emotion
# ---------------------------
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
ax.set_title('Emotion-wise Total vs True Positives on TESS')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate(f'{height}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("emotion_true_positive_comparison_tess.png")
print("📊 Saved bar chart as emotion_true_positive_comparison_tess.png")
