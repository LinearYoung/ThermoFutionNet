import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class IRVisibleDataset(Dataset):
    """红外-可见光图像数据集"""

    def __init__(self, data_root, split='train', img_size=256):
        """
        Args:
            data_root: 数据根目录
            split: 'train', 'val', or 'test'
            img_size: 图像尺寸
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size

        # 构建图像路径
        visible_dir = os.path.join(data_root, split, 'visible')
        infrared_dir = os.path.join(data_root, split, 'infrared')

        # 获取所有图像文件
        self.visible_files = sorted(glob.glob(os.path.join(visible_dir, '*.png')) +
                                    glob.glob(os.path.join(visible_dir, '*.jpg')))
        self.infrared_files = sorted(glob.glob(os.path.join(infrared_dir, '*.png')) +
                                     glob.glob(os.path.join(infrared_dir, '*.jpg')))

        # 验证文件名匹配
        self._validate_files()

        # 定义变换
        self.visible_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.infrared_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def _validate_files(self):
        """验证可见光和红外图像文件名是否匹配"""
        assert len(self.visible_files) == len(self.infrared_files), \
            f"Visible and infrared image counts don't match: {len(self.visible_files)} vs {len(self.infrared_files)}"

        for vis_file, ir_file in zip(self.visible_files, self.infrared_files):
            vis_name = os.path.splitext(os.path.basename(vis_file))[0]
            ir_name = os.path.splitext(os.path.basename(ir_file))[0]
            assert vis_name == ir_name, f"File names don't match: {vis_name} vs {ir_name}"

    def __len__(self):
        return len(self.visible_files)

    def __getitem__(self, idx):
        visible_path = self.visible_files[idx]
        infrared_path = self.infrared_files[idx]

        # 读取图像
        visible_img = Image.open(visible_path).convert('RGB')
        infrared_img = Image.open(infrared_path).convert('L')

        # 应用变换
        visible_tensor = self.visible_transform(visible_img)
        infrared_tensor = self.infrared_transform(infrared_img)

        return {
            'visible': visible_tensor,
            'infrared': infrared_tensor,
            'filename': os.path.basename(visible_path)
        }
