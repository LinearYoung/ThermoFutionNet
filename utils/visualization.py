import matplotlib.pyplot as plt
import numpy as np
from .image_utils import tensor_to_numpy


def visualize_result(visible, infrared, attention, fused, save_path=None):
    """可视化结果"""
    vis_np = tensor_to_numpy(visible)
    ir_np = tensor_to_numpy(infrared)
    att_np = tensor_to_numpy(attention)
    fused_np = tensor_to_numpy(fused)

    # 创建子图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(vis_np)
    axes[0].set_title('Visible')
    axes[0].axis('off')

    if ir_np.shape[2] == 1:
        axes[1].imshow(ir_np[:, :, 0], cmap='gray')
    else:
        axes[1].imshow(ir_np)
    axes[1].set_title('Infrared')
    axes[1].axis('off')

    if att_np.shape[2] == 1:
        im = axes[2].imshow(att_np[:, :, 0], cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].imshow(att_np)
    axes[2].set_title('Attention Map')
    axes[2].axis('off')

    axes[3].imshow(fused_np)
    axes[3].set_title('Fused Result')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
