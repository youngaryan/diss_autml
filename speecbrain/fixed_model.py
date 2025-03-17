# this model achives over 90% accuracy
# starts with fixed parameters and unfreezes them after first epoch

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain.inference import EncoderClassifier
from data_preprocessing.dataset import EmotionDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SpeechBrain pretrained model
model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition",
    run_opts={"device": device}
)

# Initially freeze lower layers (feature extraction) to stabilize training on a small dataset
for name, param in model.named_parameters():
    if "wav2vec2" in name:
        param.requires_grad = False

# Load dataset
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")

# Encode string emotion labels to integers
df["emotion"] = df["emotion"].astype(int)
label_set = sorted(df["emotion"].unique())
label_encoder = {label: idx for idx, label in enumerate(label_set)}
print("Label mapping:", label_encoder)

# Adjust final classification layer for 7 classes (labels 0 to 6)
num_classes = len(label_encoder)  # Should be 7
model.mods.output_mlp = nn.Linear(model.mods.output_mlp.w.in_features, num_classes)

# Dataset and DataLoader
max_length = 3 * 16000  # 3 seconds at 16kHz
batch_size = 1

dataset = EmotionDataset(df, feature_extractor=None, max_length=max_length, label_encoder=label_encoder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training setup
criterion = nn.CrossEntropyLoss()
lr = 1e-5
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=lr)
num_epochs = 3
# Set up a learning rate scheduler (Cosine Annealing)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Optionally unfreeze the feature extractor layers after the first epoch
    if epoch == 1:
        print("Unfreezing feature extractor layers...")
        for name, param in model.named_parameters():
            if "wav2vec2" in name:
                param.requires_grad = True
        # Reinitialize optimizer and scheduler with the new trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - epoch)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients at the start of each iteration
        optimizer.zero_grad()
        
        # Fix input shape if needed
        inputs = inputs.squeeze()
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)  # [T] -> [1, T]

        # Forward pass: extract features
        features = model.mods.wav2vec2.extract_features(inputs)[0]
        if features.ndim == 2:
            features = features.unsqueeze(0)  # Now features: [1, T, H]

        # Average pooling over time dimension
        pooled = model.mods.avg_pool(features)  # Expected: [B, H]
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)  # [H] -> [1, H]
        elif pooled.ndim == 3:
            pooled = pooled.squeeze(1)    # [B, 1, H] -> [B, H]

        predictions = model.mods.output_mlp(pooled)  # [B, C]
        loss = criterion(predictions, targets)
        
        # Backward pass and gradient clipping for stable training
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (predictions.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

        print(f"Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Step the scheduler at the end of each epoch
    scheduler.step()
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = total_correct / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
