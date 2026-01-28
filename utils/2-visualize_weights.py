#!/usr/bin/env python3
"""
可视化QuantizeLinear权重分布的工具
"""

import argparse
import os

# 添加项目路径到Python路径
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from quant_vision_transformer import *
except:
    from .quant_vision_transformer import *


def load_checkpoint(checkpoint_path):
    """加载checkpoint文件"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 打印checkpoint信息
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if "model" in checkpoint:
        print(f"Model state dict keys (first 10): {list(checkpoint['model'].keys())[:10]}")

    return checkpoint


def extract_quantizelinear_weights(model_state_dict):
    """从模型状态字典中提取QuantizeLinear层的权重"""
    quant_weights = OrderedDict()

    for name, param in model_state_dict.items():
        # 查找QuantizeLinear相关的权重
        if "weight" in name and ("fc1" in name or "fc2" in name or "qkv" in name or "proj" in name):
            # 检查是否是QuantizeLinear的权重
            if param.dim() >= 2:  # 确保是权重矩阵
                quant_weights[name] = param.detach().cpu().numpy()
                print(f"Found QuantizeLinear weight: {name}, shape: {param.shape}")

    return quant_weights


def plot_weight_distribution(weights_dict, output_dir, model_version):
    """绘制权重分布图"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置绘图风格
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 为每个权重创建单独的图
    for weight_name, weight_data in weights_dict.items():
        # 展平权重数据用于直方图
        flat_weights = weight_data.flatten()

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 直方图
        ax1.hist(flat_weights, bins=256, alpha=0.7, edgecolor="black")
        ax1.set_title(f"{weight_name}\nWeight Distribution Histogram", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"Mean: {flat_weights.mean():.4f}\nStd: {flat_weights.std():.4f}\n"
        stats_text += f"Min: {flat_weights.min():.4f}\nMax: {flat_weights.max():.4f}\n"
        stats_text += f"Shape: {weight_data.shape}"
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 箱线图
        ax2.boxplot(flat_weights, vert=True)
        ax2.set_title(f"{weight_name}\nWeight Distribution Boxplot", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Weight Value")
        ax2.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        # 清理文件名
        safe_weight_name = weight_name.replace(".", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"{model_version}_{safe_weight_name}_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved distribution plot: {output_path}")


def create_summary_plot(weights_dict, output_dir, model_version):
    """创建所有权重的汇总图"""
    if not weights_dict:
        print("No weights found for summary plot")
        return

    # 创建汇总统计图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 统计信息
    names = []
    means = []
    stds = []
    mins = []
    maxs = []

    for weight_name, weight_data in weights_dict.items():
        flat_weights = weight_data.flatten()
        names.append(weight_name)
        means.append(flat_weights.mean())
        stds.append(flat_weights.std())
        mins.append(flat_weights.min())
        maxs.append(flat_weights.max())

    # 1. 均值比较
    axes[0].barh(range(len(means)), means)
    axes[0].set_yticks(range(len(means)))
    axes[0].set_yticklabels([name.split(".")[-1] for name in names], fontsize=8)
    axes[0].set_title("Mean Weight Values")
    axes[0].set_xlabel("Mean Value")
    axes[0].grid(True, alpha=0.3)

    # 2. 标准差比较
    axes[1].barh(range(len(stds)), stds)
    axes[1].set_yticks(range(len(stds)))
    axes[1].set_yticklabels([name.split(".")[-1] for name in names], fontsize=8)
    axes[1].set_title("Weight Standard Deviations")
    axes[1].set_xlabel("Standard Deviation")
    axes[1].grid(True, alpha=0.3)

    # 3. 值范围比较
    for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
        axes[2].plot([min_val, max_val], [i, i], "o-", linewidth=2)
    axes[2].set_yticks(range(len(mins)))
    axes[2].set_yticklabels([name.split(".")[-1] for name in names], fontsize=8)
    axes[2].set_title("Weight Value Ranges")
    axes[2].set_xlabel("Weight Value")
    axes[2].grid(True, alpha=0.3)

    # 4. 所有权重的直方图叠加
    for weight_name, weight_data in weights_dict.items():
        flat_weights = weight_data.flatten()
        # 标准化以便比较
        normalized_weights = (flat_weights - flat_weights.mean()) / flat_weights.std()
        axes[3].hist(normalized_weights, bins=50, alpha=0.5, label=weight_name.split(".")[-1])

    axes[3].set_title("Normalized Weight Distributions (All Layers)")
    axes[3].set_xlabel("Normalized Weight Value")
    axes[3].set_ylabel("Frequency")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f"QuantizeLinear Weight Analysis - {model_version}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 保存汇总图
    output_path = os.path.join(output_dir, f"{model_version}_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plot: {output_path}")

    # # 保存统计信息到JSON文件
    # stats_data = {}
    # for i, name in enumerate(names):
    #     stats_data[name] = {
    #         'mean': float(means[i]),
    #         'std': float(stds[i]),
    #         'min': float(mins[i]),
    #         'max': float(maxs[i]),
    #         'shape': list(weights_dict[name].shape)
    #     }

    # stats_path = os.path.join(output_dir, f'{model_version}_statistics.json')
    # with open(stats_path, 'w') as f:
    #     json.dump(stats_data, f, indent=2)

    # print(f"Saved statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize QuantizeLinear weight distributions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the best_checkpoint.pth file")
    parser.add_argument(
        "--output-dir", type=str, default="./weight_visualizations", help="Output directory for visualization results"
    )
    parser.add_argument(
        "--model-version", type=str, default="unknown_model", help="Version of the model for title generation"
    )

    args = parser.parse_args()

    try:
        # 加载checkpoint
        checkpoint = load_checkpoint(args.checkpoint)

        # 提取模型状态字典
        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        else:
            model_state_dict = checkpoint

        # 提取QuantizeLinear权重
        quant_weights = extract_quantizelinear_weights(model_state_dict)

        if not quant_weights:
            print("No QuantizeLinear weights found in the checkpoint!")
            return

        print(f"\nFound {len(quant_weights)} QuantizeLinear weight tensors")

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 绘制单个权重分布图
        plot_weight_distribution(quant_weights, args.output_dir, args.model_version)

        # 创建汇总图
        create_summary_plot(quant_weights, args.output_dir, args.model_version)

        print(f"\nVisualization completed! Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
