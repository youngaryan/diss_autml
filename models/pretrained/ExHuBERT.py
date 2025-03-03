import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2Processor, HubertModel

class ExHuBERT(nn.Module):
    def __init__(self, num_classes=None, freeze_exhubert=True):
        """
        Initialize the ExHuBERT model for feature extraction or classification.
        
        Args:
            num_classes (int, optional): If provided, a classifier head is added.
            freeze_exhubert (bool): If True, freezes the ExHuBERT model.
        """
        super(ExHuBERT, self).__init__()

        # Load the pre-trained model and processor
        # self.processor = Wav2Vec2Processor.from_pretrained("amiriparian/ExHuBERT")
        self.hubert = HubertModel.from_pretrained("amiriparian/ExHuBERT")

        # Freeze the feature extractor if needed
        if freeze_exhubert:
            for param in self.hubert.parameters():
                param.requires_grad = False

        # Add classification head if num_classes is given
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, waveform, sampling_rate=16000):
        """
        Forward pass of ExHuBERT for feature extraction or classification.

        Args:
            waveform (torch.Tensor): The input waveform (1D tensor).
            sampling_rate (int): Sampling rate (default: 16kHz).

        Returns:
            torch.Tensor: Features or classification logits.
        """
        # Process input waveform
        inputs = self.processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Move to the same device as the model
        device = next(self.hubert.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract features from ExHuBERT
        with torch.no_grad():
            outputs = self.hubert(**inputs)

        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

        # Pooling to reduce sequence length
        pooled_output = hidden_states.mean(dim=1)  # Global average pooling

        if self.num_classes is not None:
            return self.classifier(pooled_output)  # Return classification logits

        return pooled_output  # Return extracted features

    def extract_features(self, waveform, sampling_rate=16000):
        """
        Extracts embeddings from ExHuBERT.

        Args:
            waveform (torch.Tensor): Input waveform (1D tensor).
            sampling_rate (int): Sampling rate.

        Returns:
            torch.Tensor: Extracted features (batch_size, hidden_dim).
        """
        return self.forward(waveform, sampling_rate)

