import torch
import torch.nn as nn
import torch.nn.functional as F
from Simulated.Cortex.D_demosaicing.D_Abstract import AbstractDemosaicing
from . import register_class

@register_class("Sandbox")
class SandboxDemosaicing(AbstractDemosaicing):
    def __init__(self, params, device):
        super(SandboxDemosaicing, self).__init__(params, device)

        self.device = device
        self.demosaicing = UNet(dim_latent=params['CorticalModel']['latent_dim']).to(device=self.device)

    def demosaic(self, C_encoded_pa):
        ip = self.demosaicing(C_encoded_pa)
        return ip


class UNet(nn.Module):
    def __init__(self, dim_latent):
        super(UNet, self).__init__()
        self.inc = DoubleConv(dim_latent, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.up1 = Up(64, 32, False)
        self.up2 = Up(32, 16, False)
        self.outc = OutConv(16, dim_latent)
            
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False).to(memory_format=torch.channels_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False).to(memory_format=torch.channels_last),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )       

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(memory_format=torch.channels_last)

    def forward(self, x):
        x = self.conv(x)
        return x