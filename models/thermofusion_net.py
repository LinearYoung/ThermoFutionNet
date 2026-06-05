import torch
import torch.nn as nn
from .encoders import VisibleEncoder, InfraredEncoder
from .attention_fusion import AttentionFusion
from .decoder import Decoder


class ThermoFusionNet(nn.Module):
    """ThermoFusionNet 主网络"""

    def __init__(self):
        super(ThermoFusionNet, self).__init__()
        self.visible_encoder = VisibleEncoder()
        self.infrared_encoder = InfraredEncoder()

        # 多尺度注意力融合模块
        self.fusion1 = AttentionFusion(64)
        self.fusion2 = AttentionFusion(128)
        self.fusion3 = AttentionFusion(256)
        self.fusion4 = AttentionFusion(512)

        self.decoder = Decoder()

    def forward(self, visible, infrared):
        """
        Args:
            visible: 可见光RGB图像 [B, 3, H, W]
            infrared: 红外图像 [B, 1, H, W]
        Returns:
            fused: 融合后的RGB图像 [B, 3, H, W]
            attention_maps: 各尺度注意力图列表
        """
        # 编码
        vis_features = self.visible_encoder(visible)
        ir_features = self.infrared_encoder(infrared)

        # 多尺度融合
        fused_features = []
        attention_maps = []

        f1, a1 = self.fusion1(vis_features[0], ir_features[0])
        fused_features.append(f1)
        attention_maps.append(a1)

        f2, a2 = self.fusion2(vis_features[1], ir_features[1])
        fused_features.append(f2)
        attention_maps.append(a2)

        f3, a3 = self.fusion3(vis_features[2], ir_features[2])
        fused_features.append(f3)
        attention_maps.append(a3)

        f4, a4 = self.fusion4(vis_features[3], ir_features[3])
        fused_features.append(f4)
        attention_maps.append(a4)

        # 解码
        fused = self.decoder(fused_features)

        return fused, attention_maps
