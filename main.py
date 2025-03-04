import data_preprocessing.download_data as download_data

def initlaise():
    download_data.download_database()
    # m = download_data.generate_metadata_for_emodb()
    # print(m)
    # s = sum(m.values())
    # print(s)

    download_data.generate_train_test_sample()

def pretrained_model(): 
    from models.pretrained.ExHuBERT import ExHuBERT
    import torchaudio

    # Load model for feature extraction
    model = ExHuBERT(num_classes=7)  # No classification head
    model.eval()

    # Load audio file
    waveform, sample_rate = torchaudio.load("examples/03a04Ad.wav")

    # Extract features
    features = model.extract_features(waveform)
    print(features.shape)  # Expe


    logits = model(waveform)
    predicted_class = logits.argmax(dim=-1).item()
    print(f"Predicted Emotion Class: {predicted_class}")

if __name__ == '__main__':
    initlaise()
    # pretrained_model()