import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Converter(nn.Module):
    def __init__(self, n_mels=80, emb_dim=192, hid=256):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(n_mels, hid),
            ConvBlock(hid, hid)
        )
        self.proj = nn.Linear(emb_dim, hid)
        self.dec = nn.Sequential(
            ConvBlock(hid, hid),
            nn.Conv1d(hid, n_mels, 1)
        )
    def forward(self, mel, emb):
        # mel: (B, n_mels, T)
        h = self.enc(mel)
        e = self.proj(emb).unsqueeze(-1)
        h = h + e
        return self.dec(h)