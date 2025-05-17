import os, glob
import warnings
# 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Windows에서 심볼릭링크 권한 문제 우회
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import numpy as np
import torch
import librosa, soundfile as sf
from speechbrain.inference import EncoderClassifier

# 경로 설정
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DIR = os.path.join(data_root, "raw")
WAV_DIR = os.path.join(data_root, "wav")
PROC_DIR = os.path.join(data_root, "processed")
for d in (WAV_DIR, PROC_DIR): os.makedirs(d, exist_ok=True)

# 상수
SR = 16000
N_MELS = 80
HOP = 256
WIN = 1024

# 스피커 임베딩 모델 로드 (기본 캐시 사용, GPU/CPU 자동 선택)
device = "cuda" if torch.cuda.is_available() else "cpu"
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# 전처리 루프
for audio_path in glob.glob(os.path.join(RAW_DIR, "*")):
    name, _ = os.path.splitext(os.path.basename(audio_path))
    # 1) WAV로 변환
    wav_path = os.path.join(WAV_DIR, f"{name}.wav")
    y, sr0 = sf.read(audio_path)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    sf.write(wav_path, y, SR)

    # 2) mel-spectrogram 추출
    wav, _ = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=WIN, hop_length=HOP, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel)

    # 3) 스피커 임베딩 추출
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)  # (1, T)
    emb = spk_model.encode_batch(wav_tensor)                     # (1, D)
    emb = emb.squeeze().cpu().numpy()

    # 4) 저장
    np.savez(
        os.path.join(PROC_DIR, f"{name}.npz"),
        mel=mel_db.astype(np.float32),
        emb=emb.astype(np.float32)
    )