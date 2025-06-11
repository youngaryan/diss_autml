
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import io
import torchaudio
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import torchaudio.transforms as T


class EmotionDataset(Dataset):
    def __init__(self, dataframe, feature_extractor=None, max_length=48000, label_encoder=None):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        # Decode audio from bytes using torchaudio
        audio_bytes = sample["audio"]["bytes"]
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))


        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = T.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # remove channel dim if mono

        # Encode string label to integer
        label_str = sample["emotion"]
        if self.label_encoder and isinstance(label_str, str):
            # label = self.label_encoder[label_str]
            label = self.label_encoder.transform([label_str])[0]
        else:
            label = int(label_str)
        # label = self.label_encoder[label_str] if self.label_encoder else label_str

        # Pad or truncate waveform
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.size(0) < self.max_length:
            pad_length = self.max_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform, torch.tensor(label, dtype=torch.long)