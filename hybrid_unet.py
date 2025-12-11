import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Basic Conv Block
# ----------------------------------------
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

# ----------------------------------------
# Encoder Block
# ----------------------------------------
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

# ----------------------------------------
# Decoder Block
# ----------------------------------------
class DecoderBlock(nn.Module):
    """Upsample + concat + conv block with size alignment"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐尺寸
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ----------------------------------------
# Transformer Block (ViT-style)
# ----------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=2048):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        h = self.norm1(x_flat)
        h, _ = self.attn(h, h, h)
        x_flat = x_flat + h

        h = self.norm2(x_flat)
        h = self.mlp(h)
        x_flat = x_flat + h

        return x_flat.transpose(1, 2).reshape(B, C, H, W)

# ----------------------------------------
# Boundary Enhancement Module (BEM)
# ----------------------------------------
class BEM(nn.Module):
    def __init__(self, cnn_ch=64, trans_ch=64):
        super(BEM, self).__init__()
        self.conv_cnn = nn.Sequential(
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1),
            nn.BatchNorm2d(cnn_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1)
        )
        self.conv_trans = nn.Sequential(
            nn.Conv2d(trans_ch, trans_ch, 3, padding=1),
            nn.BatchNorm2d(trans_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(trans_ch, trans_ch, 3, padding=1)
        )

    def forward(self, f_cnn, f_trans):
        f1 = self.conv_cnn(f_cnn)
        f2 = self.conv_trans(f_trans)
        return F.relu(f1 + f2)

# ----------------------------------------
# UNet with Transformer + BEM
# ----------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, create_big=False, trans_downsample=4):
        super(UNet, self).__init__()
        self.create_big = create_big
        self.trans_downsample = trans_downsample

        # ---- Encoder ----
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)

        if create_big:
            self.e4 = EncoderBlock(256, 512)
            self.bridge = ConvBlock(512, 1024)
        else:
            self.bridge = ConvBlock(256, 512)

        # Transformer + BEM for first stage
        self.transformer1 = TransformerBlock(dim=64)
        self.bem1 = BEM(cnn_ch=64, trans_ch=64)

        # ---- Decoder ----
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
        # ---- Stage 1 ----
        f1, p1 = self.e1(x)

        # Downsample for transformer
        if self.trans_downsample > 1:
            f1_ds = F.interpolate(f1, scale_factor=1/self.trans_downsample, mode='bilinear', align_corners=False)
            f1_trans_ds = self.transformer1(f1_ds)
            f1_trans = F.interpolate(f1_trans_ds, size=f1.shape[2:], mode='bilinear', align_corners=False)
        else:
            f1_trans = self.transformer1(f1)

        f1_bem = self.bem1(f1, f1_trans)

        # ---- Encoder ----
        f2, p2 = self.e2(f1_bem)
        f3, p3 = self.e3(f2)

        if self.create_big:
            f4, p4 = self.e4(f3)
            bridge = self.bridge(p4)
            d1 = self.d1(bridge, f4)
            d2 = self.d2(d1, f3)
            d3 = self.d3(d2, f2)
            d_final = self.d4(d3, f1_bem)
        else:
            bridge = self.bridge(p3)
            d1 = self.d1(bridge, f3)
            d2 = self.d2(d1, f2)
            d_final = self.d3(d2, f1_bem)

        return self.final_conv(d_final)
