#!/usr/bin/env python3
"""
计算QuantizeLinear权重矩阵秩的工具
"""

import argparse
import json
import os

# 添加项目路径到Python路径
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import seaborn as sns
import torch
import torch.nn as nn

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
        # 查找MLP相关的权重（fc1和fc2）
        if "weight" in name and any(x in name for x in ["mlp.fc1", "mlp.fc2", "attn.qkv", "attn.proj"]):
            # 检查是否是权重矩阵
            if param.dim() >= 2:  # 确保是权重矩阵
                quant_weights[name] = param.detach().cpu().numpy()
                print(f"Found MLP weight: {name}, shape: {param.shape}")

    return quant_weights


def calculate_effective_rank(matrix, method="shannon"):
    """计算矩阵的有效秩（基于奇异值分布）

    Args:
        matrix: 输入矩阵
        method: 计算方法，可选 'shannon'（香农熵）或 'stable'（稳定秩）
    """
    if matrix.size == 0:
        return 0.0

    try:
        # 计算奇异值
        singular_values = np.linalg.svd(matrix, compute_uv=False)

        # 归一化奇异值（使其和为1）
        singular_values_norm = singular_values / np.sum(singular_values)

        if method == "shannon":
            # 基于香农熵的有效秩
            # 避免log(0)的情况
            singular_values_norm = np.maximum(singular_values_norm, 1e-12)
            entropy = -np.sum(singular_values_norm * np.log(singular_values_norm))
            effective_rank = np.exp(entropy)

        elif method == "stable":
            # 稳定秩（stable rank）
            fro_norm_sq = np.sum(singular_values**2)
            op_norm_sq = np.max(singular_values) ** 2
            effective_rank = fro_norm_sq / op_norm_sq if op_norm_sq > 0 else 0

        else:
            raise ValueError(f"Unknown method: {method}")

        return effective_rank

    except Exception as e:
        print(f"Error calculating effective rank: {e}")
        return 0.0


def calculate_matrix_rank(matrix, tol=None):
    """计算矩阵的秩"""
    if matrix.size == 0:
        return 0

    # 使用SVD计算矩阵的秩
    if tol is None:
        # 默认使用机器精度作为容差
        tol = max(matrix.shape) * np.finfo(matrix.dtype).eps

    try:
        rank = np.linalg.matrix_rank(matrix, tol=tol)
        return rank
    except Exception as e:
        print(f"Error calculating rank: {e}")
        return 0


def calculate_grouped_rank(weight_matrix, num_groups_weight):
    """根据num_groups_weight分组计算秩"""
    if num_groups_weight <= 1:
        # 如果num_groups_weight=1，直接计算整个矩阵的秩
        rank = calculate_matrix_rank(weight_matrix)
        print(f"  Full matrix rank: {rank}")
        return rank

    # 检查矩阵形状是否可以被num_groups_weight整除
    # 对于MLP权重，我们通常按照输出维度（行数）进行分组
    if weight_matrix.shape[0] % num_groups_weight != 0:
        print(f"Warning: shape[0] ({weight_matrix.shape[0]}) not divisible by num_groups_weight ({num_groups_weight})")
        # 如果不能整除，使用整个矩阵
        rank = calculate_matrix_rank(weight_matrix)
        print(f"  Full matrix rank: {rank}")
        return rank

    # 按照shape[0]维度（行）分组
    group_size = weight_matrix.shape[0] // num_groups_weight
    total_rank = 0

    print(f"  Grouping: {weight_matrix.shape[0]} -> {num_groups_weight} groups of size {group_size}")

    for i in range(num_groups_weight):
        # 提取当前组的子矩阵（按行分组）
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        group_matrix = weight_matrix[start_idx:end_idx, :]  # 修复：按行分组

        # 计算当前组的秩
        group_rank = calculate_matrix_rank(group_matrix)
        total_rank += group_rank

        print(f"    Group {i+1}: shape {group_matrix.shape}, rank: {group_rank}")

    print(f"  Total grouped rank: {total_rank}")
    return total_rank


def analyze_mlp_weights_rank(weights_dict, num_groups_weight=1):
    """分析MLP权重矩阵的秩"""
    if not weights_dict:
        print("No MLP weights found for rank analysis!")
        return

    # print(f"\nAnalyzing MLP weights rank with num_groups_weight={num_groups_weight}")
    # print("=" * 80)

    rank_results = OrderedDict()

    for weight_name, weight_data in weights_dict.items():
        # print(f"\nAnalyzing: {weight_name}")
        # print(f"  Shape: {weight_data.shape}")
        # print(f"  Data type: {weight_data.dtype}")

        # 计算矩阵的秩
        if num_groups_weight > 1:
            rank = calculate_grouped_rank(weight_data, num_groups_weight)
        else:
            rank = calculate_matrix_rank(weight_data)
            # print(f"  Full matrix rank: {rank}")

        # 计算有效秩
        effective_rank_shannon = calculate_effective_rank(weight_data, method="shannon")
        effective_rank_stable = calculate_effective_rank(weight_data, method="stable")

        # 计算秩与最小维度的比例
        min_dim = min(weight_data.shape)
        rank_ratio = rank / min_dim if min_dim > 0 else 0
        effective_ratio_shannon = effective_rank_shannon / min_dim if min_dim > 0 else 0
        effective_ratio_stable = effective_rank_stable / min_dim if min_dim > 0 else 0

        # 存储结果
        rank_results[weight_name] = {
            "shape": weight_data.shape,
            "rank": rank,
            "effective_rank_shannon": effective_rank_shannon,
            "effective_rank_stable": effective_rank_stable,
            "min_dimension": min_dim,
            "rank_ratio": rank_ratio,
            "effective_ratio_shannon": effective_ratio_shannon,
            "effective_ratio_stable": effective_ratio_stable,
            "num_groups_weight": num_groups_weight,
        }

        # print(f"  Full rank: {rank}")
        # print(f"  Effective rank (Shannon): {effective_rank_shannon:.4f}")
        # print(f"  Effective rank (Stable): {effective_rank_stable:.4f}")
        # print(f"  Rank ratio: {rank_ratio:.4f} ({rank}/{min_dim})")
        # print(f"  Effective ratio (Shannon): {effective_ratio_shannon:.4f}")
        # print(f"  Effective ratio (Stable): {effective_ratio_stable:.4f}")
        # print("-" * 50)

    return rank_results


def print_rank_summary(rank_results):
    """打印秩分析摘要"""
    if not rank_results:
        return

    print("\n" + "=" * 80)
    print("MLP WEIGHTS RANK SUMMARY")
    print("=" * 80)

    # 打印表格头
    print(
        f"{'Weight Name':<30} {'Shape':<15} {'Rank':<6} {'EffRank(S)':<10} {'EffRank(T)':<10} {'Ratio':<8} {'EffRatio(S)':<12} {'EffRatio(T)':<12}"
    )
    print("-" * 120)

    for weight_name, result in rank_results.items():
        shape_str = f"{result['shape'][0]}x{result['shape'][1]}"
        print(
            f"{weight_name:<30} {shape_str:<15} {result['rank']:<6} "
            f"{result['effective_rank_shannon']:<10.2f} {result['effective_rank_stable']:<10.2f} "
            f"{result['rank_ratio']:<8.4f} {result['effective_ratio_shannon']:<12.4f} {result['effective_ratio_stable']:<12.4f}"
        )

    # 计算平均秩比例
    avg_ratio = np.mean([result["rank_ratio"] for result in rank_results.values()])
    avg_eff_ratio_shannon = np.mean([result["effective_ratio_shannon"] for result in rank_results.values()])
    avg_eff_ratio_stable = np.mean([result["effective_ratio_stable"] for result in rank_results.values()])

    print("\nAverage ratios:")
    print(f"  Rank ratio: {avg_ratio:.4f}")
    print(f"  Effective ratio (Shannon): {avg_eff_ratio_shannon:.4f}")
    print(f"  Effective ratio (Stable): {avg_eff_ratio_stable:.4f}")

    # 分析秩分布
    ranks = [result["rank"] for result in rank_results.values()]
    eff_ranks_shannon = [result["effective_rank_shannon"] for result in rank_results.values()]
    eff_ranks_stable = [result["effective_rank_stable"] for result in rank_results.values()]
    min_dims = [result["min_dimension"] for result in rank_results.values()]

    print("\nRank statistics:")
    print(f"  Min rank: {min(ranks)}")
    print(f"  Max rank: {max(ranks)}")
    print(f"  Avg rank: {np.mean(ranks):.2f}")
    print(f"  Full rank layers: {sum(1 for r in ranks if r == min_dims[ranks.index(r)])}/{len(ranks)}")

    print("\nEffective rank statistics (Shannon):")
    print(f"  Min effective rank: {min(eff_ranks_shannon):.2f}")
    print(f"  Max effective rank: {max(eff_ranks_shannon):.2f}")
    print(f"  Avg effective rank: {np.mean(eff_ranks_shannon):.2f}")

    print("\nEffective rank statistics (Stable):")
    print(f"  Min effective rank: {min(eff_ranks_stable):.2f}")
    print(f"  Max effective rank: {max(eff_ranks_stable):.2f}")
    print(f"  Avg effective rank: {np.mean(eff_ranks_stable):.2f}")


def save_rank_results(rank_results, output_dir, model_version):
    """保存秩分析结果到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存到JSON文件
    results_path = os.path.join(output_dir, f"{model_version}_rank_analysis.json")

    # 转换numpy类型为Python原生类型
    serializable_results = {}
    for key, result in rank_results.items():
        serializable_results[key] = {
            "shape": [int(dim) for dim in result["shape"]],
            "rank": int(result["rank"]),
            "min_dimension": int(result["min_dimension"]),
            "rank_ratio": float(result["rank_ratio"]),
            "num_groups_weight": int(result["num_groups_weight"]),
        }

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Rank results saved to: {results_path}")

    # 保存简明的文本摘要
    summary_path = os.path.join(output_dir, f"{model_version}_rank_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MLP WEIGHTS RANK ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for weight_name, result in serializable_results.items():
            f.write(f"{weight_name}\n")
            f.write(f"  Shape: {result['shape'][0]}x{result['shape'][1]}\n")
            f.write(f"  Rank: {result['rank']}\n")
            f.write(f"  Min Dimension: {result['min_dimension']}\n")
            f.write(f"  Rank Ratio: {result['rank_ratio']:.4f}\n")
            f.write(f"  Num Groups Weight: {result['num_groups_weight']}\n")
            f.write("-" * 40 + "\n")

    print(f"Rank summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate MLP weights matrix rank")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the best_checkpoint.pth file")
    parser.add_argument(
        "--output-dir", type=str, default="./rank_analysis", help="Output directory for rank analysis results"
    )
    parser.add_argument("--model-version", type=str, default="unknown_model", help="Name of the model for file naming")
    parser.add_argument("--num-groups-weight", type=int, default=1, help="Number of groups for weight matrix analysis")

    args = parser.parse_args()

    try:
        # 加载checkpoint
        checkpoint = load_checkpoint(args.checkpoint)

        # 提取模型状态字典
        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        else:
            model_state_dict = checkpoint

        # 提取MLP权重（fc1和fc2）
        mlp_weights = extract_quantizelinear_weights(model_state_dict)

        if not mlp_weights:
            print("No MLP weights (fc1, fc2) found in the checkpoint!")
            return

        print(f"\nFound {len(mlp_weights)} MLP weight tensors")

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 分析MLP权重矩阵的秩
        rank_results = analyze_mlp_weights_rank(mlp_weights, args.num_groups_weight)

        # 打印摘要
        print_rank_summary(rank_results)

        # 保存结果
        save_rank_results(rank_results, args.output_dir, args.model_version)

        print(f"\nRank analysis completed! Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
