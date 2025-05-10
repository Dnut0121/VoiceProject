import librosa
import numpy as np

def extract_mel(path: str, n_mels=80, hop_length=256, win_length=1024):
    y, sr = librosa.load(path, sr=None)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length
    )
    return librosa.power_to_db(mel, ref=np.max)