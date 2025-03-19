import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain.inference import EncoderClassifier
from data_preprocessing.dataset_speech_brain import EmotionDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Configuration
# ---------------------------
config = {
    "batch_size": 1,
    "max_length": 3 * 16000,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ---------------------------
# Fix helper functions (same as before)
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
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy

# ---------------------------
# Load Pretrained Model (no training)
# ---------------------------
device = config["device"]
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": device}
)

# ---------------------------
# Load Dataset and Prepare Dataloader
# ---------------------------
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
label_encoder_obj = LabelEncoder()
df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)
num_classes = len(mapping)

# Replace output layer to match our dataset labels
in_features = model.mods.output_mlp.w.in_features
model.mods.output_mlp = nn.Linear(in_features, num_classes)
nn.init.xavier_uniform_(model.mods.output_mlp.weight)
model.to(device)

# Split the dataset (we will ONLY evaluate on validation set)
_, valid_df = train_test_split(df, test_size=0.2, random_state=42)
valid_dataset = EmotionDataset(valid_df, feature_extractor=None, max_length=config["max_length"], label_encoder=mapping)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# ---------------------------
# Evaluate without training
# ---------------------------
criterion = nn.CrossEntropyLoss()
val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
print(f"Zero-shot Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
