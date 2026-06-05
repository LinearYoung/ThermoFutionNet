import os
import argparse
import random
import shutil
from pathlib import Path


def get_image_files(directory):
    """获取目录下所有图像文件，支持多种格式"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'}
    image_files = []
    for f in os.listdir(directory):
        if os.path.splitext(f)[1] in valid_extensions:
            image_files.append(f)
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description='Prepare MSRS dataset')
    parser.add_argument('--msrs_root', type=str, default='./MSRS-main',
                        help='Path to MSRS-main root directory')
    parser.add_argument('--out_root', type=str, default='./data',
                        help='Output root directory')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    args = parser.parse_args()

    random.seed(args.seed)

    # 定义路径
    msrs_train_vi = os.path.join(args.msrs_root, 'train', 'vi')
    msrs_train_ir = os.path.join(args.msrs_root, 'train', 'ir')
    msrs_test_vi = os.path.join(args.msrs_root, 'test', 'vi')
    msrs_test_ir = os.path.join(args.msrs_root, 'test', 'ir')

    # 检查源目录是否存在
    for path in [msrs_train_vi, msrs_train_ir, msrs_test_vi, msrs_test_ir]:
        if not os.path.exists(path):
            print(f"Warning: Source directory not found: {path}")

    # 获取训练集文件
    train_vi_files = get_image_files(msrs_train_vi)
    train_ir_files = get_image_files(msrs_train_ir)

    # 检查训练集配对
    train_vi_set = set(train_vi_files)
    train_ir_set = set(train_ir_files)
    train_paired_files = sorted(list(train_vi_set & train_ir_set))

    if len(train_vi_files) != len(train_ir_files):
        print(f"Warning: Train visible ({len(train_vi_files)}) and infrared ({len(train_ir_files)}) counts differ")
    if len(train_paired_files) < len(train_vi_files) or len(train_paired_files) < len(train_ir_files):
        print(f"Warning: Only {len(train_paired_files)} paired images found in train set")

    # 划分训练和验证
    num_train = int(len(train_paired_files) * (1 - args.val_ratio))
    random.shuffle(train_paired_files)
    train_files = train_paired_files[:num_train]
    val_files = train_paired_files[num_train:]

    # 获取测试集文件
    test_vi_files = get_image_files(msrs_test_vi)
    test_ir_files = get_image_files(msrs_test_ir)

    # 检查测试集配对
    test_vi_set = set(test_vi_files)
    test_ir_set = set(test_ir_files)
    test_paired_files = sorted(list(test_vi_set & test_ir_set))

    if len(test_vi_files) != len(test_ir_files):
        print(f"Warning: Test visible ({len(test_vi_files)}) and infrared ({len(test_ir_files)}) counts differ")
    if len(test_paired_files) < len(test_vi_files) or len(test_paired_files) < len(test_ir_files):
        print(f"Warning: Only {len(test_paired_files)} paired images found in test set")

    # 定义输出目录
    out_dirs = {
        'train_visible': os.path.join(args.out_root, 'train', 'visible'),
        'train_infrared': os.path.join(args.out_root, 'train', 'infrared'),
        'val_visible': os.path.join(args.out_root, 'val', 'visible'),
        'val_infrared': os.path.join(args.out_root, 'val', 'infrared'),
        'test_visible': os.path.join(args.out_root, 'test', 'visible'),
        'test_infrared': os.path.join(args.out_root, 'test', 'infrared'),
    }

    # 检查是否需要覆盖
    for dir_path in out_dirs.values():
        if os.path.exists(dir_path) and os.listdir(dir_path):
            if args.overwrite:
                shutil.rmtree(dir_path)
            else:
                print(f"Error: Directory {dir_path} exists and is not empty. Use --overwrite to proceed.")
                return

    # 创建输出目录
    for dir_path in out_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 复制训练集
    for filename in train_files:
        shutil.copy2(os.path.join(msrs_train_vi, filename), os.path.join(out_dirs['train_visible'], filename))
        shutil.copy2(os.path.join(msrs_train_ir, filename), os.path.join(out_dirs['train_infrared'], filename))

    # 复制验证集
    for filename in val_files:
        shutil.copy2(os.path.join(msrs_train_vi, filename), os.path.join(out_dirs['val_visible'], filename))
        shutil.copy2(os.path.join(msrs_train_ir, filename), os.path.join(out_dirs['val_infrared'], filename))

    # 复制测试集
    for filename in test_paired_files:
        shutil.copy2(os.path.join(msrs_test_vi, filename), os.path.join(out_dirs['test_visible'], filename))
        shutil.copy2(os.path.join(msrs_test_ir, filename), os.path.join(out_dirs['test_infrared'], filename))

    # 打印统计
    print(f"\nDataset preparation completed!")
    print(f"Train pairs: {len(train_files)}")
    print(f"Val pairs: {len(val_files)}")
    print(f"Test pairs: {len(test_paired_files)}")


if __name__ == '__main__':
    main()
