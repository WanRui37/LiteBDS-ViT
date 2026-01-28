import json
import os
from pathlib import Path

import cv2


def organize_flowers_dataset(json_path, src_img_dir, dst_root_dir):
    """
    根据JSON标注生成目录结构，并复制图片到对应类别目录
    Args:
        json_path: JSON标注文件路径（包含train/val划分和类别信息）
        src_img_dir: 原始图片目录（/mnt/data/small_dataset/flower/flowers-102/jpg/）
        dst_root_dir: 目标根目录（将在该目录下生成train/val及class子目录）
    """
    # 1. 加载JSON数据
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到JSON文件 {json_path}")
        return
    except json.JSONDecodeError:
        print("错误：JSON文件格式无效，请检查文件内容")
        return

    # 验证JSON结构（确保包含train和val键）
    required_splits = ["train", "val", "test"]
    for split in required_splits:
        if split not in dataset_info:
            print(f"错误：JSON文件缺少 {split} 字段")
            return

    # 2. 定义类别目录生成函数（统一格式：class_00001 ~ class_00102）
    def create_class_dirs(split_dir):
        """在指定split目录下创建所有类别子目录（1-102类）"""
        for class_id in range(1, 103):  # 102类，从1到102
            class_dir_name = f"{class_id:d}"  # 格式化为5位数字（00001~00102）
            class_dir = os.path.join(split_dir, class_dir_name)
            Path(class_dir).mkdir(parents=True, exist_ok=True)  # 递归创建，已存在则跳过
        print(f"已创建 {split_dir} 下所有类别目录")

    # 3. 遍历train/val划分，复制图片到对应目录
    for split in required_splits:
        # 创建split根目录（train/val）
        split_dir = os.path.join(dst_root_dir, split)
        Path(split_dir).mkdir(parents=True, exist_ok=True)

        # 创建该split下的所有类别目录
        create_class_dirs(split_dir)

        # 处理该split下的所有图片
        image_entries = dataset_info[split]
        total = len(image_entries)
        success_count = 0
        fail_count = 0
        print(f"\n开始处理 {split} 集（共 {total} 张图片）...")

        for entry in image_entries:
            # 解析JSON条目：[图片名, 原始类别编号（0开始）, 类别名]
            if len(entry) != 3:
                print(f"警告：跳过无效条目 {entry}（格式错误）")
                fail_count += 1
                continue

            img_name, orig_class_id, class_name = entry

            # 转换类别编号：0→1，76→77，最终1-102（对应class_00001~class_00102）
            dst_class_id = orig_class_id + 1
            if not (1 <= dst_class_id <= 102):
                print(f"警告：跳过图片 {img_name}（无效类别编号 {orig_class_id}）")
                fail_count += 1
                print(f"错误：类别编号 {orig_class_id} 超出范围（应在0-101之间）")
                exit(0)

            # 构建源图片路径和目标路径
            src_img_path = os.path.join(src_img_dir, img_name)
            class_dir_name = f"{dst_class_id:d}"
            dst_img_path = os.path.join(split_dir, class_dir_name, img_name)

            # 读取图片，resize后保存
            try:
                if os.path.exists(src_img_path):
                    img = cv2.imread(src_img_path)
                    if img is not None:
                        # img = cv2.resize(img, (256, 256))
                        cv2.imwrite(dst_img_path, img)
                        success_count += 1
                    else:
                        print(f"警告：无法读取图片 {img_name}")
                        fail_count += 1
                else:
                    print(f"警告：跳过 {img_name}（源文件不存在）")
                    fail_count += 1
            except Exception as e:
                print(f"警告：处理 {img_name} 失败 - {str(e)}")
                fail_count += 1

        # 打印该split的处理结果
        print(f"{split} 集处理完成：成功 {success_count} 张，失败 {fail_count} 张")

    print(f"\n所有数据集处理完成！目标目录：{dst_root_dir}")
    print("目录结构：")
    print(f"- {dst_root_dir}")
    print("  - train")
    print("    - class_00001 ~ class_00102")
    print("  - val")
    print("    - class_00001 ~ class_00102")
    print("  - test")
    print("    - class_00001 ~ class_00102")


if __name__ == "__main__":
    # -------------------------- 请根据实际情况修改以下参数 --------------------------
    JSON_PATH = "/mnt/data/small_dataset/flower_tca/flowers-102/split_zhou_OxfordFlowers.json"  # 你的JSON标注文件路径
    SRC_IMG_DIR = "/mnt/data/small_dataset/flower_tca/flowers-102/jpg/"  # 原始图片目录（固定）
    DST_ROOT_DIR = "/mnt/data/small_dataset/flower_tca/flowers-102/"  # 目标根目录（可自定义）
    # --------------------------------------------------------------------------------

    # 执行数据集整理
    organize_flowers_dataset(json_path=JSON_PATH, src_img_dir=SRC_IMG_DIR, dst_root_dir=DST_ROOT_DIR)
