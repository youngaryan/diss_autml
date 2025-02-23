# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
# import torch
# import librosa
# import numpy as np

# # Load your pretrained model
# model_name = "superb/wav2vec2-base-superb-er"
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# def predict_emotion(audio_file):
#     # Load audio file
#     speech, sr = librosa.load(audio_file, sr=16000)

#     # Extract features from audio
#     inputs = feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

#     # Get predictions
#     with torch.no_grad():
#         logits = model(**inputs).logits

#     # Get the predicted class
#     predicted_class = torch.argmax(logits, dim=-1).item()

#     # Get emotion label
#     emotion = model.config.id2label[predicted_class]
    
#     return emotion

# if __name__ == "__main__":
#     audio_path = "./database/1/wav/03a01Fa.wav"
#     emotion = predict_emotion(audio_path)
#     print(f"Predicted Emotion: {emotion}")




from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa
import os

tot =0
tr=0

# emotion
# Load your pretrained model
model_name = "superb/wav2vec2-base-superb-er"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

def predict_emotion(audio_file):
    speech, sr = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[predicted_class] ##4 emotions atm
    return emotion

def predict_emotions_in_folder(folder_path):
    predictions = {}
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

    for audio_file in wav_files:
        file_path = os.path.join(folder_path, audio_file)
        emotion = predict_emotion(file_path)
        predictions[audio_file] = emotion
        print(f"{audio_file}: {emotion}")
        tot+=1
        

    return predictions

if __name__ == "__main__":
    folder_path = "./database/1/wav"  # Replace with your audio files folder path
    predictions = predict_emotions_in_folder(folder_path)


