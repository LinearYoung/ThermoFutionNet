import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLoss(nn.Module):
    """多模态融合损失函数"""

    def __init__(self, lambda_intensity=1.0, lambda_gradient=5.0,
                 lambda_color=1.0, lambda_thermal=2.0):
        super(FusionLoss, self).__init__()
        self.lambda_intensity = lambda_intensity
        self.lambda_gradient = lambda_gradient
        self.lambda_color = lambda_color
        self.lambda_thermal = lambda_thermal

    def rgb_to_gray(self, x):
        """将RGB图像转换为灰度图像"""
        # 使用标准的RGB到灰度转换公式
        return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

    def sobel_gradient(self, x):
        """使用Sobel算子计算梯度"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_x = sobel_x.to(x.device)
        sobel_y = sobel_y.to(x.device)

        # 计算x和y方向的梯度
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)

        # 梯度幅值
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return gradient

    def forward(self, fused, visible, infrared):
        """
        Args:
            fused: 融合后的RGB图像 [B, 3, H, W]
            visible: 可见光RGB图像 [B, 3, H, W]
            infrared: 红外图像 [B, 1, H, W]
        Returns:
            loss: 总损失
            loss_dict: 各部分损失字典
        """
        # 转换为灰度
        fused_gray = self.rgb_to_gray(fused)
        visible_gray = self.rgb_to_gray(visible)

        # 1. Intensity Loss
        # 取可见光和红外的最大值作为目标
        target_intensity = torch.max(visible_gray, infrared)
        l_intensity = F.l1_loss(fused_gray, target_intensity)

        # 2. Gradient Loss
        fused_grad = self.sobel_gradient(fused_gray)
        visible_grad = self.sobel_gradient(visible_gray)
        infrared_grad = self.sobel_gradient(infrared)
        target_grad = torch.max(visible_grad, infrared_grad)
        l_gradient = F.l1_loss(fused_grad, target_grad)

        # 3. Color Loss
        l_color = F.l1_loss(fused, visible)

        # 4. Thermal Loss
        # 生成热区域mask，取红外图像top 20%的亮度区域
        batch_size = infrared.shape[0]
        thermal_mask = torch.zeros_like(infrared)
        for i in range(batch_size):
            ir_flat = infrared[i].view(-1)
            threshold = torch.quantile(ir_flat, 0.8)
            thermal_mask[i] = (infrared[i] >= threshold).float()

        # 在热区域让融合图像更接近红外
        l_thermal = F.l1_loss(fused_gray * thermal_mask, infrared * thermal_mask)

        # 总损失
        loss = (self.lambda_intensity * l_intensity +
                self.lambda_gradient * l_gradient +
                self.lambda_color * l_color +
                self.lambda_thermal * l_thermal)

        loss_dict = {
            'total': loss.item(),
            'intensity': l_intensity.item(),
            'gradient': l_gradient.item(),
            'color': l_color.item(),
            'thermal': l_thermal.item()
        }

        return loss, loss_dict
