import os
import soundfile as sf
import numpy as np
from scipy.signal import resample
import librosa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
RAW_DIR    = os.path.join(SRC_DIR, 'data', 'raw_audio')
PROC_DIR   = os.path.join(SRC_DIR, 'data', 'processed_audio')
TARGET_SR  = 16000
os.makedirs(PROC_DIR, exist_ok=True)

def preprocess_file(fname: str):
    path_in = os.path.join(RAW_DIR, fname)
    y, sr = librosa.load(path_in, sr=None, mono=True)
    if sr != TARGET_SR:
        n = int(len(y) * TARGET_SR / sr)
        y = resample(y, n)
    out_name = os.path.splitext(fname)[0] + '.wav'
    path_out = os.path.join(PROC_DIR, out_name)
    sf.write(path_out, y.astype(np.float32), TARGET_SR)
    print(f"[Preprocess] {fname} → {out_name}")

if __name__ == '__main__':
    for f in os.listdir(RAW_DIR):
        if f.lower().endswith(('.wav', '.mp3', '.flac')):
            preprocess_file(f)