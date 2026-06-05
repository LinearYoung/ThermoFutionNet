import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import IRVisibleDataset
from models import ThermoFusionNet
from losses import FusionLoss
from utils import save_checkpoint, load_checkpoint, Logger, save_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='Train ThermoFusionNet')
    parser.add_argument('--config', type=str, default='configs/train_4090.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_output_dirs(config):
    """获取输出目录，兼容旧配置"""
    if 'output' in config:
        checkpoint_dir = config['output'].get('checkpoint_dir', './checkpoints')
        result_dir = config['output'].get('result_dir', './results/train')
        log_dir = config['output'].get('log_dir', './logs')
    else:
        checkpoint_dir = './checkpoints'
        result_dir = './results/train'
        log_dir = './logs'
    return checkpoint_dir, result_dir, log_dir


def main():
    args = parse_args()
    config = load_config(args.config)

    # 获取输出目录
    checkpoint_dir, result_dir, log_dir = get_output_dirs(config)

    # 创建必要的目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 设置设备
    if not torch.cuda.is_available() and config['device'] == 'cuda':
        print(f"Warning: CUDA not available, falling back to CPU")
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据加载
    train_dataset = IRVisibleDataset(
        data_root=config['data_root'],
        split='train',
        img_size=config['img_size']
    )
    val_dataset = IRVisibleDataset(
        data_root=config['data_root'],
        split='val',
        img_size=config['img_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 模型初始化
    model = ThermoFusionNet().to(device)

    # 损失函数
    loss_weights = config['loss_weights']
    criterion = FusionLoss(
        lambda_intensity=loss_weights['lambda_intensity'],
        lambda_gradient=loss_weights['lambda_gradient'],
        lambda_color=loss_weights['lambda_color'],
        lambda_thermal=loss_weights['lambda_thermal']
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # AMP - only enable if CUDA available
    amp_enabled = config['amp'] and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 日志记录器
    logger = Logger(log_dir)

    # 断点续训
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, args.resume, device
        )

    # 训练循环
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # 训练阶段
        model.train()
        train_losses = {
            'total': 0.0,
            'intensity': 0.0,
            'gradient': 0.0,
            'color': 0.0,
            'thermal': 0.0
        }

        train_bar = tqdm(train_loader, desc='Training')
        for batch in train_bar:
            visible = batch['visible'].to(device)
            infrared = batch['infrared'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                fused, attention_maps = model(visible, infrared)
                loss, loss_dict = criterion(fused, visible, infrared)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新损失
            for k in train_losses:
                train_losses[k] += loss_dict[k]

            train_bar.set_postfix({'loss': loss_dict['total']})

        # 平均训练损失
        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # 验证阶段
        model.eval()
        val_losses = {
            'total': 0.0,
            'intensity': 0.0,
            'gradient': 0.0,
            'color': 0.0,
            'thermal': 0.0
        }

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for batch in val_bar:
                visible = batch['visible'].to(device)
                infrared = batch['infrared'].to(device)

                fused, attention_maps = model(visible, infrared)
                loss, loss_dict = criterion(fused, visible, infrared)

                for k in val_losses:
                    val_losses[k] += loss_dict[k]

                val_bar.set_postfix({'loss': loss_dict['total']})

        # 平均验证损失
        for k in val_losses:
            val_losses[k] /= len(val_loader)

        # 打印损失
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"Intensity: {train_losses['intensity']:.4f}, "
              f"Gradient: {train_losses['gradient']:.4f}, "
              f"Color: {train_losses['color']:.4f}, "
              f"Thermal: {train_losses['thermal']:.4f}")
        print(f"Val   - Total: {val_losses['total']:.4f}, "
              f"Intensity: {val_losses['intensity']:.4f}, "
              f"Gradient: {val_losses['gradient']:.4f}, "
              f"Color: {val_losses['color']:.4f}, "
              f"Thermal: {val_losses['thermal']:.4f}")

        # 记录日志
        logger.log_scalar('train/total_loss', train_losses['total'], epoch + 1)
        logger.log_scalar('train/intensity_loss', train_losses['intensity'], epoch + 1)
        logger.log_scalar('train/gradient_loss', train_losses['gradient'], epoch + 1)
        logger.log_scalar('train/color_loss', train_losses['color'], epoch + 1)
        logger.log_scalar('train/thermal_loss', train_losses['thermal'], epoch + 1)

        logger.log_scalar('val/total_loss', val_losses['total'], epoch + 1)
        logger.log_scalar('val/intensity_loss', val_losses['intensity'], epoch + 1)
        logger.log_scalar('val/gradient_loss', val_losses['gradient'], epoch + 1)
        logger.log_scalar('val/color_loss', val_losses['color'], epoch + 1)
        logger.log_scalar('val/thermal_loss', val_losses['thermal'], epoch + 1)

        # 保存检查点
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        save_checkpoint(
            model, optimizer, epoch + 1, val_losses['total'],
            latest_path,
            is_best=False,
            best_val_loss=best_val_loss,
            config=config
        )

        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = os.path.join(checkpoint_dir, 'best.pth')
            save_checkpoint(
                model, optimizer, epoch + 1, val_losses['total'],
                best_path,
                is_best=True,
                best_val_loss=best_val_loss,
                config=config
            )
            print(f"Best model saved! Val Loss: {best_val_loss:.4f}")

        # 可视化
        if (epoch + 1) % config['vis_interval'] == 0 and len(val_dataset) > 0:
            model.eval()
            with torch.no_grad():
                sample = val_dataset[0]
                visible = sample['visible'].unsqueeze(0).to(device)
                infrared = sample['infrared'].unsqueeze(0).to(device)
                fused, attention_maps = model(visible, infrared)
                # 使用第一个尺度的注意力图
                attention = attention_maps[0]
                # 使用带前导零的文件名
                save_comparison(
                    visible, infrared, attention, fused,
                    os.path.join(result_dir, f'epoch_{epoch + 1:03d}_compare.png')
                )

    logger.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
