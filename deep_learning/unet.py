import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        inter_x = self.double_conv(x)

        return self.max_pool(inter_x), inter_x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_below, x_left):
        upsampled = self.upsample(x_below)
        h, w = upsampled.shape[2], upsampled.shape[3]
        H, W = x_left.shape[2], x_left.shape[3]

        offset_w = W - w
        offset_h = H - h

        # left, right, top, bottom
        padded_upsampled = F.pad(
            upsampled,
            [
                offset_w // 2,
                offset_w - offset_w // 2,
                offset_h // 2,
                offset_h - offset_h // 2,
            ],
        )

        concatenated = torch.cat([x_left, padded_upsampled], dim=1)

        return self.double_conv(concatenated)
