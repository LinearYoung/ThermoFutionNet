import numpy as np
import torch
from PIL import Image


def tensor_to_numpy(tensor):
    """将Tensor转换为numpy数组"""
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取batch中的第一个
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


def tensor_to_image(tensor):
    """将Tensor转换为PIL Image"""
    img = tensor_to_numpy(tensor)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]  # 单通道灰度图
    return Image.fromarray(img)


def rgb_to_gray(x):
    """将RGB Tensor转换为灰度"""
    return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]


def normalize_for_display(x):
    """归一化到 [0,1] 用于显示"""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu()
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    return torch.clamp(x, 0, 1)


def make_comparison_grid(visible, infrared, attention, fused):
    """
    创建四列对比图：Visible | Infrared | Attention | Fused
    """
    vis_np = tensor_to_numpy(visible)
    ir_np = tensor_to_numpy(infrared)
    att_np = tensor_to_numpy(attention)
    fused_np = tensor_to_numpy(fused)

    # 确保都是3通道
    if vis_np.shape[2] == 1:
        vis_np = np.concatenate([vis_np] * 3, axis=2)
    if ir_np.shape[2] == 1:
        ir_np = np.concatenate([ir_np] * 3, axis=2)
    if att_np.shape[2] == 1:
        # 注意力图用热力图
        import cv2
        att_gray = (att_np[:, :, 0] * 255).astype(np.uint8)
        att_np = cv2.applyColorMap(att_gray, cv2.COLORMAP_JET)
        att_np = cv2.cvtColor(att_np, cv2.COLOR_BGR2RGB) / 255.0
    if fused_np.shape[2] == 1:
        fused_np = np.concatenate([fused_np] * 3, axis=2)

    # 拼接
    h, w = vis_np.shape[:2]
    grid = np.zeros((h, w * 4, 3), dtype=np.float32)
    grid[:, :w] = vis_np
    grid[:, w:w * 2] = ir_np
    grid[:, w * 2:w * 3] = att_np
    grid[:, w * 3:w * 4] = fused_np

    # 转为uint8
    grid = (grid * 255).astype(np.uint8)
    return Image.fromarray(grid)


def save_image(tensor, save_path):
    """保存单个图像"""
    img = tensor_to_image(tensor)
    img.save(save_path)


def save_comparison_image(visible, infrared, attention, fused, save_path):
    """保存对比图"""
    grid = make_comparison_grid(visible, infrared, attention, fused)
    grid.save(save_path)


def save_comparison(visible, infrared, attention, fused, save_path):
    """兼容旧接口"""
    save_comparison_image(visible, infrared, attention, fused, save_path)
