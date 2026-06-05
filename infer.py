import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from models import ThermoFusionNet
from utils import save_image, save_comparison_image


def parse_args():
    parser = argparse.ArgumentParser(description='Infer ThermoFusionNet on single pair')
    parser.add_argument('--visible', type=str, required=True,
                        help='Path to visible image')
    parser.add_argument('--infrared', type=str, required=True,
                        help='Path to infrared image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='results/infer_result.png',
                        help='Path to output fused image')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    return parser.parse_args()


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
    args = parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型初始化
    model = ThermoFusionNet().to(device)

    # 加载检查点
    try:
        model = load_model_checkpoint(model, args.checkpoint, device)
    except Exception as e:
        print(f"Error: Failed to load checkpoint.")
        print(f"Please check if checkpoint path '{args.checkpoint}' is correct.")
        print(f"Original error: {str(e)}")
        return

    model.eval()

    # 图像变换
    visible_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    infrared_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # 读取图像
    visible_img = Image.open(args.visible).convert('RGB')
    infrared_img = Image.open(args.infrared).convert('L')

    # 应用变换
    visible_tensor = visible_transform(visible_img).unsqueeze(0).to(device)
    infrared_tensor = infrared_transform(infrared_img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        fused, attention_maps = model(visible_tensor, infrared_tensor)
        attention = attention_maps[0]

    # 保存融合图像
    save_image(fused, args.output)

    # 保存对比图
    output_dir = os.path.dirname(args.output)
    output_name = os.path.splitext(os.path.basename(args.output))[0]
    compare_path = os.path.join(output_dir, f'{output_name}_compare.png')
    save_comparison_image(visible_tensor, infrared_tensor, attention, fused, compare_path)

    print(f"Fused image saved to: {args.output}")
    print(f"Comparison image saved to: {compare_path}")


if __name__ == '__main__':
    main()
