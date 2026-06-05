import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import IRVisibleDataset
from models import ThermoFusionNet
from losses import FusionLoss
from utils import save_comparison_image


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Smoke test for ThermoFusionNet')
    parser.add_argument('--config', type=str, default='configs/train_local.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Create output directory
    os.makedirs('results/smoke_test', exist_ok=True)

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available() and config['device'] == 'cuda':
        print(f"Warning: CUDA not available, falling back to CPU")
    print(f"Using device: {device}")

    # Dataset and dataloader
    dataset = IRVisibleDataset(
        data_root=config['data_root'],
        split='train',
        img_size=config['img_size']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    # Get a batch
    batch = next(iter(dataloader))
    visible = batch['visible'].to(device)
    infrared = batch['infrared'].to(device)

    print(f"Visible shape: {visible.shape}")
    print(f"Infrared shape: {infrared.shape}")

    # Create model
    model = ThermoFusionNet().to(device)

    # Forward pass
    model.train()
    fused, attention_maps = model(visible, infrared)
    attention = attention_maps[0]  # Use first scale

    print(f"Fused shape: {fused.shape}")
    print(f"Attention shape: {attention.shape}")

    # Loss
    loss_weights = config['loss_weights']
    criterion = FusionLoss(
        lambda_intensity=loss_weights['lambda_intensity'],
        lambda_gradient=loss_weights['lambda_gradient'],
        lambda_color=loss_weights['lambda_color'],
        lambda_thermal=loss_weights['lambda_thermal']
    )
    loss, loss_dict = criterion(fused, visible, infrared)

    print(f"\nLosses:")
    print(f"  Total loss: {loss_dict['total']:.6f}")
    print(f"  Intensity loss: {loss_dict['intensity']:.6f}")
    print(f"  Gradient loss: {loss_dict['gradient']:.6f}")
    print(f"  Color loss: {loss_dict['color']:.6f}")
    print(f"  Thermal loss: {loss_dict['thermal']:.6f}")

    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save comparison image
    save_comparison_image(visible, infrared, attention, fused, 'results/smoke_test/smoke_test_compare.png')
    print(f"\nComparison image saved to results/smoke_test/smoke_test_compare.png")

    print("\nSmoke test passed.")


if __name__ == '__main__':
    main()
