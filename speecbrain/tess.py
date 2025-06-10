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
from datasets import load_dataset

# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "max_length": 64000,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ---------------------------
# EMO-DB Emotion Labels
# ---------------------------
EMODB_LABELS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
label_encoder = LabelEncoder()
label_encoder.fit(EMODB_LABELS)

# ---------------------------
# TESS to EMO-DB Mapping
# ---------------------------
tess_to_emodb = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "surprise": "boredom"  # optional mapping
}

# ---------------------------
# Load TESS from Hugging Face
# ---------------------------
def load_tess_dataset_from_hf():
    print("📥 Downloading TESS from Hugging Face...")
    dataset = load_dataset("PolyAI/TESS")
    data = []
    for item in dataset["train"]:
        filepath = item["file"]
        emotion = item["label"].lower()
        data.append({"filepath": filepath, "emotion": emotion})
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} audio samples from Hugging Face.")
    return df

df = load_tess_dataset_from_hf()

if df.empty:
    raise ValueError("❌ Dataset is empty. Check Hugging Face download or internet connection.")

df["emotion"] = df["emotion"].map(tess_to_emodb)
df = df[df["emotion"].notnull()].reset_index(drop=True)

# ---------------------------
# Define TESSDataset
# ---------------------------
class TESSDataset(Dataset):
    def __init__(self, dataframe, max_length=64000, label_encoder=None):
        self.dataframe = dataframe
        self.max_length = max_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        waveform, sample_rate = torchaudio.load(sample["filepath"])

        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0)

        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            pad_len = self.max_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

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

model.to(config["device"])
model.eval()

# ---------------------------
# Run Evaluation
# ---------------------------
dataset = TESSDataset(df, label_encoder=label_encoder, max_length=config["max_length"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
criterion = nn.CrossEntropyLoss()

test_loss, test_bacc, all_preds, all_targets = validate(model, dataloader, criterion, config["device"])
print(f"✅ TESS Generalization Test — Loss: {test_loss:.4f}, Balanced Accuracy: {test_bacc:.4f}")

# ---------------------------
# Plot Confusion Matrix
# ---------------------------
cm = confusion_matrix(all_targets, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMODB_LABELS)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix on TESS (Predicted vs True)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_tess.png")
print("📊 Saved confusion matrix as confusion_matrix_tess.png")

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
