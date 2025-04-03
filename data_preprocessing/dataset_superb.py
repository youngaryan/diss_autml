import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class EmotionDatasetHF(Dataset):
    def __init__(self, df, feature_extractor, label_encoder, sampling_rate=16000):
        self.df = df
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["path"])
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        label = self.label_encoder.transform([row["label"]])[0]
        return {
            "input_values": waveform.squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    padded_inputs = pad_sequence(input_values, batch_first=True)
    return padded_inputs, labels
