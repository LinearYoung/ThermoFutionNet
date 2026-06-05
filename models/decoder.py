import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ConvBlock


class Decoder(nn.Module):
    """U-Net风格解码器"""

    def __init__(self):
        super(Decoder, self).__init__()
        # 上采样 + 卷积块
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64)
        )

        # 最后输出层
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fused_features):
        """
        Args:
            fused_features: 融合后的多尺度特征列表 [f1, f2, f3, f4]
                           f1: [B, 64, H, W]
                           f2: [B, 128, H/2, W/2]
                           f3: [B, 256, H/4, W/4]
                           f4: [B, 512, H/8, W/8]
        Returns:
            out: 融合后的RGB图像 [B, 3, H, W]
        """
        f1, f2, f3, f4 = fused_features

        # 解码阶段4
        x = self.up4(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.dec4(x)

        # 解码阶段3
        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.dec3(x)

        # 解码阶段2
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec2(x)

        # 输出层
        out = self.final_conv(x)
        out = self.sigmoid(out)

        return out
