import pyworld as pw
import soundfile as sf
import numpy as np

def extract_pitch_harvest(
    path: str,
    fs: int = 16000,
    frame_period: float = 5.0,
    lower_f0: float = 71.0,
    upper_f0: float = 800.0
):
    x, sr = sf.read(path)
    if sr != fs:
        raise ValueError(f"Expected {fs}Hz, got {sr}Hz")
    f0, t = pw.harvest(
        x.astype(np.float64), fs,
        frame_period=frame_period,
        f0_floor=lower_f0,
        f0_ceil=upper_f0
    )
    f0 = pw.stonemask(x.astype(np.float64), f0, t, fs)
    return t, f0