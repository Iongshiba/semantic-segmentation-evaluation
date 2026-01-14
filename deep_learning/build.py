import torch.nn as nn

from .unet import DoubleConv, Up, Down


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout=0.3):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        enc_channels = [64, 128, 256, 512, 1024]

        self.downs.append(Down(n_channels, enc_channels[0]))

        for i in range(len(enc_channels) - 1):
            self.downs.append(Down(enc_channels[i], enc_channels[i + 1]))

        for i in range(len(enc_channels) - 1, 0, -1):
            self.ups.append(Up(enc_channels[i], enc_channels[i - 1], dropout=dropout))

        # Final convolution
        self.final_conv = nn.Conv2d(enc_channels[0], n_classes, kernel_size=1)

    def forward(self, x, verbose=False):
        skip_connections = []

        # contracting
        for i, down in enumerate(self.downs):
            x, skip = down(x)
            skip_connections.append(skip)
            if verbose:
                print(f"Down_{i}:\t\tx-{x.shape}\t\tskip-{skip.shape}")

        # bottleneck
        x = skip_connections[-1]
        skip_connections = skip_connections[:-1]

        # expansive
        for i, up in enumerate(self.ups):
            skip = skip_connections.pop()
            x = up(x, skip)
            if verbose:
                print(f"Up_{i}:\t\tx-{x.shape}")

        x = self.final_conv(x)
        return x
