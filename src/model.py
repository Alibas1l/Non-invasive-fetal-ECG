"""1D U-Net for fetal ECG extraction.

Plain feedforward conv stacks lose fine QRS timing because a large effective
receptive field (needed to see the slow maternal QRS/baseline-wander context)
forces either deep stacks with no skip path back to full resolution, or heavy
downsampling with no way to recover sample-accurate peak locations. A U-Net
encoder-decoder with skip connections keeps a full-resolution path alongside
the wide-context path, and residual blocks let it be deep enough to separate
maternal/fetal components without vanishing gradients.

Two output heads share the decoder trunk:
  - wave_head: regresses the synthetic clean-FECG waveform (data_pipeline.synthesize_fetal_ecg)
  - qrs_head:  regresses the fetal-QRS detection bump (data_pipeline.qrs_target_signal), sigmoid-activated

The QRS head is the one that matters for the clinical endpoint (fetal heart
rate); the waveform head gives it a stronger, denser gradient signal and a
morphology check.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 15, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 15):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.res = ResidualBlock1D(out_ch, kernel_size)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res(self.proj(x))
        return self.pool(x), x  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 15):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResidualBlock1D(out_ch, kernel_size)
        self.merge = nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:  # odd-length rounding guard
            x = nn.functional.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        return self.res(x)


class FetalECGUNet(nn.Module):
    """Input:  (batch, in_channels, T) real, z-scored abdominal signal.
    Output: dict with 'wave' (batch, T) and 'qrs' (batch, T), both squeezed
    to 1 channel; 'qrs' is passed through sigmoid (detection probability)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16, depth: int = 4):
        super().__init__()
        self.depth = depth

        chs = [base_channels * (2**i) for i in range(depth)]  # e.g. [16, 32, 64, 128]
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for ch in chs:
            self.downs.append(DownBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = ResidualBlock1D(prev_ch, kernel_size=15, dilation=2)

        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(UpBlock(prev_ch, ch, ch))
            prev_ch = ch

        self.wave_head = nn.Conv1d(prev_ch, 1, kernel_size=1)
        self.qrs_head = nn.Conv1d(prev_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        wave = self.wave_head(x).squeeze(1)
        qrs = torch.sigmoid(self.qrs_head(x)).squeeze(1)
        return {"wave": wave, "qrs": qrs}


if __name__ == "__main__":
    model = FetalECGUNet(in_channels=3, base_channels=16, depth=4)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 3, 2000)
    out = model(x)
    print(f"params={n_params:,}")
    print(f"wave shape={tuple(out['wave'].shape)}, qrs shape={tuple(out['qrs'].shape)}")
