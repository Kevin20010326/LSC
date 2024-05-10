import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.decoder1 = DoubleConv(512, 256)
        self.decoder2 = DoubleConv(256, 128)
        self.decoder3 = DoubleConv(128, 64)
        self.decoder4 = DoubleConv(64, out_channels)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.maxpool(x1))
        x3 = self.encoder3(self.maxpool(x2))
        x4 = self.encoder4(self.maxpool(x3))
        x = self.decoder1(self.upsample(x4) + x3)
        x = self.decoder2(self.upsample(x) + x2)
        x = self.decoder3(self.upsample(x) + x1)
        x = self.decoder4(self.upsample(x))
        return x
