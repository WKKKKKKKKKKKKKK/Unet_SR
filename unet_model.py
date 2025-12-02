import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double conv block: Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Conv block + MaxPool"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Upsample + concat + conv block"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    PyTorch version of your Keras U-Net.
    create_big=True => larger U-Net (31M parameters)
    create_big=False => small U-Net (~7M parameters)
    """

    def __init__(self, in_channels=3, out_channels=3, create_big=False):
        super(UNet, self).__init__()
        self.create_big = create_big

        # Encoder
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)

        if create_big:
            self.e4 = EncoderBlock(256, 512)
            self.bridge = ConvBlock(512, 1024)
        else:
            self.bridge = ConvBlock(256, 512)

        # Decoder
        if create_big:
            self.d1 = DecoderBlock(1024, 512)
            self.d2 = DecoderBlock(512, 256)
            self.d3 = DecoderBlock(256, 128)
            self.d4 = DecoderBlock(128, 64)
        else:
            self.d1 = DecoderBlock(512, 256)
            self.d2 = DecoderBlock(256, 128)
            self.d3 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode
        e1, p1 = self.e1(x)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)

        if self.create_big:
            e4, p4 = self.e4(p3)
            bridge = self.bridge(p4)
            d1 = self.d1(bridge, e4)
            d2 = self.d2(d1, e3)
            d3 = self.d3(d2, e2)
            d_final = self.d4(d3, e1)
        else:
            bridge = self.bridge(p3)
            d1 = self.d1(bridge, e3)
            d2 = self.d2(d1, e2)
            d_final = self.d3(d2, e1)

        return self.final_conv(d_final)
