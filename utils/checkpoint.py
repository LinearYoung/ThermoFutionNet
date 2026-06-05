import torch
import os


def save_checkpoint(model, optimizer, epoch, loss, save_path, is_best=False, best_val_loss=None, config=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.replace('latest', 'best')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"Loaded checkpoint from epoch {epoch}, loss: {loss}")
    return epoch, best_val_loss
