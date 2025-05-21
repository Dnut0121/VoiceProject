import torch
import torch.nn as nn

class HiFiGANGenerator(nn.Module):
    def __init__(self, n_mels=80, channels=512,
                 upsample_rates=[8,8,2,2],
                 upsample_kernel_sizes=[16,16,4,4]):
        super().__init__()
        self.conv_pre = nn.Conv1d(n_mels, channels, kernel_size=7, padding=3)
        self.ups = nn.ModuleList()
        cur_ch = channels
        for r, k in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose1d(cur_ch, cur_ch//2,
                                       kernel_size=k, stride=r,
                                       padding=(k - r)//2,
                                       output_padding=r-1),
                    nn.LeakyReLU(0.2)
                )
            )
            cur_ch //= 2
        self.conv_post = nn.Conv1d(cur_ch, 1, kernel_size=7, padding=3)

    def forward(self, mel):
        # mel: (B, n_mels, T)
        x = self.conv_pre(mel)
        for up in self.ups:
            x = up(x)
        return torch.tanh(self.conv_post(x))