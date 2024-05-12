import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

        # 添加一个卷积层，将 x3 的通道数调整为 256
        self.adjust_channels = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.maxpool(x1))
        x3 = self.encoder3(self.maxpool(x2))
        x4 = self.encoder4(self.maxpool(x3))

        print("x3 shape:", x3.shape)
        print("x4 shape:", x4.shape)

        
        # 调整 x1、x2 和 x3 的尺寸，使其与 x4 的尺寸匹配
        x1 = F.interpolate(x1, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # 对 x3 进行通道数调整
        x3 = self.adjust_channels(x3)

        # 对 x4 进行上采样操作
        x4 = self.upsample1(x4)
        x4 = F.pad(x4, (0, 1, 0, 1))  # 在右边和底部填充1个像素

        x = self.decoder1(torch.cat([self.upsample1(x4), x3], dim=1))
        x = self.decoder2(torch.cat([self.upsample2(x), x2], dim=1))
        x = self.decoder3(torch.cat([self.upsample3(x), x1], dim=1))
        x = self.decoder4(self.upsample4(x))
        return x
