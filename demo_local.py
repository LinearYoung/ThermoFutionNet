import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import IRVisibleDataset
from models import ThermoFusionNet
from utils import save_image, save_comparison_image


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_checkpoint(model, checkpoint_path, device):
    """加载checkpoint，兼容多种格式"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 尝试直接加载
            try:
                model.load_state_dict(checkpoint)
            except:
                raise ValueError("Unknown checkpoint format")
    else:
        model.load_state_dict(checkpoint)

    return model


def main():
    parser = argparse.ArgumentParser(description='Local demo for ThermoFusionNet')
    parser.add_argument('--config', type=str, default='configs/train_local.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/local/best.pth',
                        help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/local_demo',
                        help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    fused_dir = os.path.join(args.output_dir, 'fused')
    compare_dir = os.path.join(args.output_dir, 'compare')
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available() and config['device'] == 'cuda':
        print(f"Warning: CUDA not available, falling back to CPU")
    print(f"Using device: {device}")

    # 数据加载
    try:
        test_dataset = IRVisibleDataset(
            data_root=config['data_root'],
            split='test',
            img_size=config['img_size']
        )
    except Exception as e:
        print(f"Error: Failed to load test dataset.")
        print(f"Please check if data_root='{config['data_root']}' contains test/visible and test/infrared directories.")
        print(f"Original error: {str(e)}")
        return

    if len(test_dataset) == 0:
        print("Error: No test images found.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    print(f"Test samples: {len(test_dataset)}")

    # 模型初始化
    model = ThermoFusionNet().to(device)

    # 加载checkpoint
    try:
        model = load_model_checkpoint(model, args.checkpoint, device)
    except Exception as e:
        print(f"Error: Failed to load checkpoint.")
        print(f"Please check if checkpoint path '{args.checkpoint}' is correct.")
        print(f"Original error: {str(e)}")
        return

    model.eval()

    # 推理
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc='Demo')):
            visible = batch['visible'].to(device)
            infrared = batch['infrared'].to(device)
            filename = batch['filename'][0]

            fused, attention_maps = model(visible, infrared)
            attention = attention_maps[0]  # 使用第一个尺度的注意力图

            # 保存
            prefix = f'{idx + 1:04d}'
            save_path_fused = os.path.join(fused_dir, f'fused_{prefix}_{filename}')
            save_path_compare = os.path.join(compare_dir, f'compare_{prefix}_{filename}')

            save_image(fused, save_path_fused)
            save_comparison_image(visible, infrared, attention, fused, save_path_compare)

    print(f"\nDemo completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
