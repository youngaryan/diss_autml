
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

from data_preprocessing.dataset import EmotionDataset

# CONFIG and MODEL SETUP
model_name = 'amiriparian/ExHuBERT'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True,
                                                       )

# Replacing Classifier layer
model.classifier = nn.Linear(in_features=256, out_features=7)
# Freezing the original encoder layers and feature encoder (as in the paper) for further transfer learning
model.freeze_og_encoder()
model.freeze_feature_encoder()
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# print available memory
# print(torch.cpu.get_device_properties(0).total_memory)
print(device)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
# splits = {'session1': 'data/session1-00000-of-00001-04e11ca668d90573.parquet', 'session2': 'data/session2-00000-of-00001-f6132100b374cb18.parquet', 'session3': 'data/session3-00000-of-00001-6e102fcb5c1126b4.parquet', 'session4': 'data/session4-00000-of-00001-e39531a7c694b50d.parquet', 'session5': 'data/session5-00000-of-00001-03769060403172ce.parquet'}
# df = pd.read_parquet("hf://datasets/Zahra99/IEMOCAP_Audio/" + splits["session1"])

# Dataset and DataLoader
dataset = EmotionDataset(df, feature_extractor, max_length=3 * 16000)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load your DataFrame. Samples are shown for EmoDB and IEMOCAP from the Huggingface Hub
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
# splits = {'session1': 'data/session1-00000-of-00001-04e11ca668d90573.parquet', 'session2': 'data/session2-00000-of-00001-f6132100b374cb18.parquet', 'session3': 'data/session3-00000-of-00001-6e102fcb5c1126b4.parquet', 'session4': 'data/session4-00000-of-00001-e39531a7c694b50d.parquet', 'session5': 'data/session5-00000-of-00001-03769060403172ce.parquet'}
# df = pd.read_parquet("hf://datasets/Zahra99/IEMOCAP_Audio/" + splits["session1"])

# Dataset and DataLoader
dataset = EmotionDataset(df, feature_extractor, max_length=3 * 16000)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training setup
criterion = nn.CrossEntropyLoss()
lr = 1e-5
non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.AdamW(non_frozen_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08)

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optim.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()
        torch.cuda.empty_cache()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)
        print(f'Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}', 'total_correct', total_correct, 'total_samples', total_samples)

    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = total_correct / total_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_accuracy:.4f}')

# Example outputs:
# Epoch [3/3], Average Loss: 0.4572, Average Accuracy: 0.8249 for IEMOCAP
# Epoch [3/3], Average Loss: 0.1511, Average Accuracy: 0.9850 for EmoDB
