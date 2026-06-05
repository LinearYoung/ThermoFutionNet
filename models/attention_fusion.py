import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """注意力融合模块"""

    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        # 卷积生成注意力图
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f_vis, f_ir):
        """
        Args:
            f_vis: 可见光特征 [B, C, H, W]
            f_ir: 红外特征 [B, C, H, W]
        Returns:
            f_fused: 融合特征 [B, C, H, W]
            a_ir: 红外注意力图 [B, 1, H, W]
        """
        # 拼接特征
        f_concat = torch.cat([f_vis, f_ir], dim=1)

        # 生成红外注意力图
        a_ir = self.attention_conv(f_concat)

        # 可见光注意力图
        a_vis = 1 - a_ir

        # 融合特征
        f_fused = a_vis * f_vis + a_ir * f_ir

        return f_fused, a_ir
