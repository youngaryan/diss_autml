import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor


# CONFIG and MODEL SETUP
model_name = 'amiriparian/ExHuBERT'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)

# Freezing half of the encoder for further transfer learning
model.freeze_og_encoder()

sampling_rate = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Example application from a local audiofile
import numpy as np
import librosa
import torch.nn.functional as F
# Sample taken from the Toronto emotional speech set (TESS) https://tspace.library.utoronto.ca/handle/1807/24487
waveform, sr_wav = librosa.load("examples/03a04Fd.wav")
# Max Padding to 3 Seconds at 16k sampling rate for the best results
waveform = feature_extractor(waveform, sampling_rate=sampling_rate,padding = 'max_length',max_length = 48000)
waveform = waveform['input_values'][0]
waveform = waveform.reshape(1, -1)
waveform = torch.from_numpy(waveform).to(device)
with torch.no_grad():
    output = model(waveform)
    output = F.softmax(output.logits, dim = 1)
    output = output.detach().cpu().numpy().round(2)
    print(output)

    # [[0.      0.      0.      1.      0.      0.]]
    #          Low          |          High                 Arousal
    # Neg.     Neut.   Pos. |  Neg.    Neut.   Pos          Valence
    # Disgust, Neutral, Kind| Anger, Surprise, Joy          Example emotions
