# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal

# def plot_feature(y, sr, feature, feature_name, times=None):
#     plt.figure(figsize=(10, 4))
#     if times is None:
#         times = np.linspace(0, len(y) / sr, num=len(feature))
#     plt.plot(times, feature, label=feature_name, color='b')
#     plt.xlabel("Time (s)")
#     plt.ylabel(feature_name)
#     plt.title(f"{feature_name} over Time")
#     plt.legend()
#     plt.savefig(f"plt/feature_example/H/features_{feature_name}.png")

# def extract_pitch(y, sr):
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#     pitch_values = np.max(pitches, axis=0)
#     pitch_values[pitch_values == 0] = np.nan  # Ignore zero values for better visualization
#     plot_feature(y, sr, pitch_values, "Pitch")
#     return pitch_values

# def extract_energy(y, sr):
#     energy = librosa.feature.rms(y=y)[0]
#     plot_feature(y, sr, energy, "Energy (Loudness)")
#     return energy

# def extract_mfcc(y, sr, n_mfcc=13):
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     for i in range(n_mfcc):
#         plot_feature(y, sr, mfccs[i], f"MFCC {i+1}")
#     return mfccs

# def extract_spectral_centroid(y, sr):
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
#     plot_feature(y, sr, spectral_centroid, "Spectral Centroid")
#     return spectral_centroid

# def extract_spectral_rolloff(y, sr):
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
#     plot_feature(y, sr, rolloff, "Spectral Roll-off")
#     return rolloff

# def extract_spectral_flux(y, sr):
#     spectral_flux = np.diff(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
#     plot_feature(y, sr, spectral_flux, "Spectral Flux")
#     return spectral_flux

# def extract_chroma(y, sr):
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     for i in range(chroma.shape[0]):
#         plot_feature(y, sr, chroma[i], f"Chroma {i+1}")
#     return chroma

# def extract_zcr(y, sr):
#     zcr = librosa.feature.zero_crossing_rate(y)[0]
#     plot_feature(y, sr, zcr, "Zero-Crossing Rate (ZCR)")
#     return zcr

# def extract_jitter(y, sr):
#     pitch_values = extract_pitch(y, sr)
#     jitter = np.abs(np.diff(pitch_values))
#     jitter = jitter[~np.isnan(jitter)]  # Remove NaN values
#     plot_feature(y, sr, jitter, "Jitter")
#     return jitter

# def extract_shimmer(y, sr):
#     energy = extract_energy(y, sr)
#     shimmer = np.abs(np.diff(energy))
#     plot_feature(y, sr, shimmer, "Shimmer")
#     return shimmer

# def process_audio(file_path):
#     y, sr = librosa.load(file_path, sr=None)
    
#     features = {
#         "Pitch": extract_pitch(y, sr),
#         "Energy": extract_energy(y, sr),
#         "MFCCs": extract_mfcc(y, sr),
#         "Spectral Centroid": extract_spectral_centroid(y, sr),
#         "Spectral Roll-off": extract_spectral_rolloff(y, sr),
#         "Spectral Flux": extract_spectral_flux(y, sr),
#         "Chroma Features": extract_chroma(y, sr),
#         "Zero-Crossing Rate": extract_zcr(y, sr),
#         "Jitter": extract_jitter(y, sr),
#         "Shimmer": extract_shimmer(y, sr),
#     }
#     return features
# # Example Usage:
# # features = process_audio("example.wav")




import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import parselmouth


# Optional: for voice quality features, install praat-parselmouth via pip (pip install praat-parselmouth)


def display_graph(fig, title):
    """
    Displays the given matplotlib figure.
    """

    plt.savefig(f"plt/feature_example/S/features_{title}.png")
    return fig

def extract_mfcc(wav_file):
    """
    Extracts MFCCs from the wav file, plots the MFCC spectrogram,
    displays it, and returns the figure.
    """
    y, sr = librosa.load(wav_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set(title='MFCC')
    fig.colorbar(img, ax=ax)
    display_graph(fig,"MFCC")
    return fig

def extract_prosodic_features(wav_file):
    """
    Extracts prosodic features including pitch, energy, and a proxy for speaking rate.
    The pitch track is estimated using librosa's piptrack, energy via RMS, and
    speaking rate using onset strength. Each is plotted in a separate subplot.
    """
    y, sr = librosa.load(wav_file, sr=None)
    
    # Estimate pitch using piptrack (choose maximum magnitude per frame)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_track = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch_track.append(pitches[index, i])
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]
    
    # Estimate speaking rate proxy using onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(pitch_track)
    ax[0].set_title('Pitch Track')
    ax[1].plot(rms)
    ax[1].set_title('RMS Energy')
    ax[2].plot(onset_env)
    ax[2].set_title('Onset Strength (Proxy for Speaking Rate)')
    plt.tight_layout()
    display_graph(plt.gcf(), "prosodic_features")
    return plt.gcf()

def extract_spectral_features(wav_file):
    """
    Extracts spectral features: centroid, roll-off, and an approximation of spectral flux.
    Each feature is plotted in a subplot.
    """
    y, sr = librosa.load(wav_file, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Approximate spectral flux: compute frame-to-frame differences in magnitude
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(spectral_centroid)
    ax[0].set_title('Spectral Centroid')
    ax[1].plot(spectral_rolloff)
    ax[1].set_title('Spectral Rolloff')
    ax[2].plot(flux)
    ax[2].set_title('Spectral Flux')
    plt.tight_layout()
    display_graph(plt.gcf(),"spectral_features" )
    return plt.gcf()

def extract_voice_quality_features(wav_file):
    """
    Extracts voice quality features (jitter and shimmer) using parselmouth.
    If parselmouth is not installed, it will print a message and return None.
    Plots jitter and shimmer as bar graphs.
    """
    if parselmouth is None:
        print("parselmouth is not installed. Install it via pip for voice quality features.")
        return None

    snd = parselmouth.Sound(wav_file)
    # Extract pitch using parselmouth
    pitch = snd.to_pitch()
    # Compute jitter and shimmer using Praat functions (parameters can be adjusted)
    # jitter = parselmouth.praat.call(pitch, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    # shimmer = parselmouth.praat.call(snd, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax[0].bar(['Jitter'], [jitter])
    ax[0].set_title('Jitter')
    # ax[1].bar(['Shimmer'], [shimmer])
    ax[1].set_title('Shimmer')
    plt.tight_layout()
    display_graph(plt.gcf(), "voice_quality_features")
    return plt.gcf()

def extract_delta_features(wav_file):
    """
    Computes the first and second derivatives (delta and delta-delta) of the MFCCs,
    plots the original MFCCs, delta MFCC, and delta-delta MFCC, and displays the figure.
    """
    y, sr = librosa.load(wav_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    img1 = librosa.display.specshow(mfccs, x_axis='time', ax=ax[0])
    ax[0].set_title('MFCC')
    fig.colorbar(img1, ax=ax[0])
    img2 = librosa.display.specshow(delta_mfcc, x_axis='time', ax=ax[1])
    ax[1].set_title('Delta MFCC')
    fig.colorbar(img2, ax=ax[1])
    img3 = librosa.display.specshow(delta2_mfcc, x_axis='time', ax=ax[2])
    ax[2].set_title('Delta-Delta MFCC')
    fig.colorbar(img3, ax=ax[2])
    plt.tight_layout()
    display_graph(fig, "delta_features")
    return fig

def extract_time_frequency_representation(wav_file):
    """
    Computes a Mel-spectrogram from the audio and converts it to decibels,
    then plots the resulting spectrogram.
    """
    y, sr = librosa.load(wav_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title('Mel-Spectrogram')
    fig.colorbar(img, ax=ax)
    display_graph(fig, "frequency_representation")
    return fig

# Example usage:
if __name__ == '__main__':
    wav_file = 'examples/03a04Ta.wav'  # Replace with your .wav file path

    print("Extracting MFCCs...")
    extract_mfcc(wav_file)
    
    print("Extracting prosodic features...")
    extract_prosodic_features(wav_file)
    
    print("Extracting spectral features...")
    extract_spectral_features(wav_file)
    
    print("Extracting voice quality features...")
    extract_voice_quality_features(wav_file)
    
    print("Extracting delta features...")
    extract_delta_features(wav_file)
    
    print("Extracting time-frequency representation...")
    extract_time_frequency_representation(wav_file)


# if __name__ == '__main__':
#     features = process_audio("examples/16b10Fb.wav")
#     print(features)