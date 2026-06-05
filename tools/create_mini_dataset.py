import os
import argparse
import random
import shutil


def get_image_files(directory):
    """获取目录下所有图像文件"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'}
    image_files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if os.path.splitext(f)[1] in valid_extensions:
                image_files.append(f)
    return sorted(image_files)


def copy_pair(src_vis, src_ir, dst_vis, dst_ir, filename):
    """复制一对图像"""
    shutil.copy2(os.path.join(src_vis, filename), os.path.join(dst_vis, filename))
    shutil.copy2(os.path.join(src_ir, filename), os.path.join(dst_ir, filename))


def main():
    parser = argparse.ArgumentParser(description='Create mini dataset for debugging')
    parser.add_argument('--src_root', type=str, default='./data',
                        help='Source dataset root directory')
    parser.add_argument('--out_root', type=str, default='./mini_data',
                        help='Output root directory')
    parser.add_argument('--num_train', type=int, default=20,
                        help='Number of train pairs')
    parser.add_argument('--num_val', type=int, default=5,
                        help='Number of val pairs')
    parser.add_argument('--num_test', type=int, default=5,
                        help='Number of test pairs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    args = parser.parse_args()

    random.seed(args.seed)

    # 定义源路径
    src_train_vis = os.path.join(args.src_root, 'train', 'visible')
    src_train_ir = os.path.join(args.src_root, 'train', 'infrared')
    src_val_vis = os.path.join(args.src_root, 'val', 'visible')
    src_val_ir = os.path.join(args.src_root, 'val', 'infrared')
    src_test_vis = os.path.join(args.src_root, 'test', 'visible')
    src_test_ir = os.path.join(args.src_root, 'test', 'infrared')

    # 获取源文件
    train_vis_files = get_image_files(src_train_vis)
    train_ir_files = get_image_files(src_train_ir)
    val_vis_files = get_image_files(src_val_vis)
    val_ir_files = get_image_files(src_val_ir)
    test_vis_files = get_image_files(src_test_vis)
    test_ir_files = get_image_files(src_test_ir)

    # 检查配对
    train_paired = sorted(list(set(train_vis_files) & set(train_ir_files)))
    val_paired = sorted(list(set(val_vis_files) & set(val_ir_files)))
    test_paired = sorted(list(set(test_vis_files) & set(test_ir_files)))

    # 抽样
    num_train_actual = min(args.num_train, len(train_paired))
    num_val_actual = min(args.num_val, len(val_paired))
    num_test_actual = min(args.num_test, len(test_paired))

    if num_train_actual < args.num_train:
        print(f"Warning: Requested {args.num_train} train pairs, but only {len(train_paired)} available")
    if num_val_actual < args.num_val:
        print(f"Warning: Requested {args.num_val} val pairs, but only {len(val_paired)} available")
    if num_test_actual < args.num_test:
        print(f"Warning: Requested {args.num_test} test pairs, but only {len(test_paired)} available")

    random.shuffle(train_paired)
    random.shuffle(val_paired)
    random.shuffle(test_paired)

    selected_train = train_paired[:num_train_actual]
    selected_val = val_paired[:num_val_actual]
    selected_test = test_paired[:num_test_actual]

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

    # 复制文件
    for filename in selected_train:
        copy_pair(src_train_vis, src_train_ir, out_dirs['train_visible'], out_dirs['train_infrared'], filename)

    for filename in selected_val:
        copy_pair(src_val_vis, src_val_ir, out_dirs['val_visible'], out_dirs['val_infrared'], filename)

    for filename in selected_test:
        copy_pair(src_test_vis, src_test_ir, out_dirs['test_visible'], out_dirs['test_infrared'], filename)

    # 打印统计
    print(f"\nMini dataset created!")
    print(f"Train pairs: {len(selected_train)}")
    print(f"Val pairs: {len(selected_val)}")
    print(f"Test pairs: {len(selected_test)}")


if __name__ == '__main__':
    main()
