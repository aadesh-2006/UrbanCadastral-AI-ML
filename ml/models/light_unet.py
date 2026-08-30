"""
LightUNet: Ultra-lightweight U-Net architecture for semantic segmentation.

Designed specifically for CPU training and low-latency inference:
- ~1.94 Million parameters (vs 31M for standard UNet, 40M+ for DeepLabV3)
- Multi-scale skip connections preserve sharp building boundaries
- Minimal RAM footprint (< 150MB during backpropagation)
- Pure PyTorch implementation (zero external C++ binary dependencies)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightUNet(nn.Module):
    """
    Lightweight 4-stage U-Net with base_filters=16.
    Input: (B, 3, H, W)
    Output: (B, num_classes, H, W) logits
    """
    def __init__(self, in_channels=3, num_classes=1, base_filters=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, base_filters)           # 16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(base_filters, base_filters * 2)      # 32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)  # 64
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)  # 128
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16) # 256

        # Decoder (Expansive Path)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        # 1x1 Convolution Head
        self.out_head = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_head(d1)

def get_model(in_channels=3, num_classes=1, base_filters=16):
    return LightUNet(in_channels=in_channels, num_classes=num_classes, base_filters=base_filters)
