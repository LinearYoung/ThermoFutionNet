import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """卷积块: Conv + BN + ReLU"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VisibleEncoder(nn.Module):
    """可见光图像编码器"""

    def __init__(self):
        super(VisibleEncoder, self).__init__()
        # 输入: [3, H, W]
        self.enc1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )  # [64, H, W]
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )  # [128, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )  # [256, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )  # [512, H/8, W/8]

    def forward(self, x):
        features = []
        x1 = self.enc1(x)
        features.append(x1)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        features.append(x2)
        x = self.pool2(x2)

        x3 = self.enc3(x)
        features.append(x3)
        x = self.pool3(x3)

        x4 = self.enc4(x)
        features.append(x4)

        return features  # [x1, x2, x3, x4]


class InfraredEncoder(nn.Module):
    """红外图像编码器"""

    def __init__(self):
        super(InfraredEncoder, self).__init__()
        # 输入: [1, H, W]
        self.enc1 = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 64)
        )  # [64, H, W]
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )  # [128, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )  # [256, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )  # [512, H/8, W/8]

    def forward(self, x):
        features = []
        x1 = self.enc1(x)
        features.append(x1)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        features.append(x2)
        x = self.pool2(x2)

        x3 = self.enc3(x)
        features.append(x3)
        x = self.pool3(x3)

        x4 = self.enc4(x)
        features.append(x4)

        return features  # [x1, x2, x3, x4]
