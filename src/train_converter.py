import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Converter

DATA_DIR = "../data/processed"
MODEL_PATH = "../models/converter.pth"
BATCH = 8
EPOCHS = 100
LR = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvDataset(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        mel = torch.from_numpy(d['mel']).permute(1,0)  # (T, n_mels)
        emb = torch.from_numpy(d['emb'])
        return mel, emb, mel

def collate(batch):
    mels, embs, tars = zip(*batch)
    # 패딩
    lengths = [m.shape[0] for m in mels]
    max_len = max(lengths)
    mel_pad = torch.zeros(len(mels), mels[0].shape[1], max_len)
    tar_pad = torch.zeros_like(mel_pad)
    for i,m in enumerate(mels): mel_pad[i,:,:m.shape[0]] = m.T
    for i,t in enumerate(tars): tar_pad[i,:,:t.shape[0]] = t.T
    return mel_pad, torch.stack(embs), tar_pad

# 데이터 로더
files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
dset = ConvDataset(files)
loader = DataLoader(dset, batch_size=BATCH, shuffle=True, collate_fn=collate)

# 모델
model = Converter().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

for ep in range(1, EPOCHS+1):
    model.train(); total=0
    for mel, emb, tar in loader:
        mel, emb, tar = mel.to(device), emb.to(device), tar.to(device)
        opt.zero_grad()
        pred = model(mel, emb)
        loss = F.l1_loss(pred, tar)
        loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {ep}/{EPOCHS}, Loss: {total/len(loader):.4f}")
    if ep % 10 == 0:
        torch.save(model.state_dict(), MODEL_PATH)

# 최종 저장
torch.save(model.state_dict(), MODEL_PATH)
print("Saved converter.pth")
