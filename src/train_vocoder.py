import os, glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vocoder_model import HiFiGANGenerator
import soundfile as sf
from multiprocessing import freeze_support

# 경로 및 상수
MEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
WAV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wav"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "vocoder.pth"))
SR = 16000
HOP = 256
N_FFT = 1024

class VocoderDataset(Dataset):
    def __init__(self, mel_dir, wav_dir):
        self.files = glob.glob(os.path.join(mel_dir, "*.npz"))
        self.wav_dir = wav_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel_path = self.files[idx]
        name = os.path.splitext(os.path.basename(mel_path))[0]
        data = np.load(mel_path)
        # mel: (n_mels, T)
        mel = torch.from_numpy(data['mel']).float()
        # wav: (1, T_raw)
        wav, _ = sf.read(os.path.join(self.wav_dir, f"{name}.wav"))
        wav = torch.from_numpy(wav).unsqueeze(0).float()
        return mel, wav

# MR-STFT Loss 함수
def ms_stft_loss(x, y):
    # x, y: (B, 1, T_raw)
    x = x.squeeze(1)  # (B, T_raw)
    y = y.squeeze(1)
    # 창 함수 지정 (Hann window)
    window = torch.hann_window(N_FFT, device=x.device)
    X = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)
    Y = torch.stft(y, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)
    # 프레임 길이 정렬
    min_frames = min(X.size(-1), Y.size(-1))
    X = X[..., :min_frames]
    Y = Y[..., :min_frames]
    return F.l1_loss(torch.abs(X), torch.abs(Y))

# 학습 루프 함수
def main():
    ds = VocoderDataset(MEL_DIR, WAV_DIR)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HiFiGANGenerator().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.8, 0.99))

    for epoch in range(1, 201):
        model.train()
        total_loss = 0.0
        for mel, wav in loader:
            # mel: (B, n_mels, T)
            mel, wav = mel.to(device), wav.to(device)
            # Conv1d 입력 형태 유지
            fake = model(mel)
            loss = ms_stft_loss(fake, wav)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/200 - Avg Loss: {total_loss/len(loader):.4f}")
    # 최종 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved vocoder.pth")

if __name__ == "__main__":
    freeze_support()
    main()