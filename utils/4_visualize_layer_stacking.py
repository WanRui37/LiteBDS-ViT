import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

class SimpleConfig:
    """简化的配置类，用于模拟BDS实验"""
    def __init__(self):
        self.ffn_group_num = 8
        self.attn_group_num = 8
        self.head_group_num = 8
        self.ffn_linear_method = "block_diag"
        self.attn_linear_method = "block_diag"
        self.head_linear_method = "block_diag"
        self.ffn_fc1_shift_step = 1
        self.ffn_fc2_shift_step = 1
        self.attn_qkv_shift_step = 1
        self.attn_proj_shift_step = 1
        self.head_shift_step = 1
        self.learnable_groups = False

class MockLinearQ(nn.Module):
    """模拟LinearQ层，只关注组传播特性"""
    def __init__(self, in_features, out_features, num_groups=8, shift_step=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.shift_step = shift_step
        
        # 假设每个组有相同的特征数
        self.features_per_group = in_features // num_groups
        assert in_features % num_groups == 0, "in_features必须能被num_groups整除"
        assert out_features % num_groups == 0, "out_features必须能被num_groups整除"
    
    def forward(self, x):
        """模拟LinearQ的前向传播，只关注组间信息传播"""
        # x的形状: (batch_size, in_features)
        batch_size = x.shape[0]
        
        # 将输入特征按组分割
        x_groups = x.view(batch_size, self.num_groups, self.features_per_group)
        
        # 模拟组移位操作（BDS的核心）
        shifted_groups = torch.roll(x_groups, shifts=self.shift_step, dims=1)
        
        # 模拟组内线性变换（简化：只保留组激活状态）
        # 这里我们只关心哪些组有信息，不关心具体数值
        output_groups = (shifted_groups.sum(dim=2, keepdim=True) > 0).float()
        
        # 扩展回输出特征维度
        output = output_groups.repeat(1, 1, self.out_features // self.num_groups)
        output = output.view(batch_size, self.out_features)
        
        return output

class MockQBlock(nn.Module):
    """模拟Q_Block，包含LinearQ和非线性激活"""
    def __init__(self, dim, num_groups=8, shift_step=1):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        
        # 模拟两个LinearQ层（类似于Q_Block中的结构）
        self.linear1 = MockLinearQ(dim, dim, num_groups, shift_step)
        self.linear2 = MockLinearQ(dim, dim, num_groups, shift_step)
        
        # 模拟非线性激活（GELU）
        self.activation = nn.GELU()
        
        # 模拟层归一化（简化版）
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """模拟Q_Block的前向传播"""
        # 第一个LinearQ + 激活
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        
        # 第二个LinearQ
        x = self.linear2(x)
        
        # 残差连接
        x = x + residual
        
        return x

def simulate_layer_stacking(layer_type, num_layers=8, dim=64, num_groups=8, shift_step=1):
    """
    模拟层堆叠的信息传播过程
    
    Args:
        layer_type: 'linearq' 或 'qblock'
        num_layers: 堆叠的层数
        dim: 特征维度
        num_groups: 组数
        shift_step: 移位步长
    
    Returns:
        coverage_list: 每层的组覆盖比例列表
        activation_maps: 每层的激活状态图
    """
    
    # 初始化输入：只有第一个组有信息
    x = torch.zeros(1, dim)
    x[0, :dim//num_groups] = 1.0  # 只有第一个组激活
    
    coverage_list = []
    activation_maps = []
    
    # 创建层实例
    if layer_type == 'linearq':
        layers = [MockLinearQ(dim, dim, num_groups, shift_step) for _ in range(num_layers)]
    else:  # qblock
        layers = [MockQBlock(dim, num_groups, shift_step) for _ in range(num_layers)]
    
    # 记录初始状态
    initial_coverage = (x > 0).float().sum(dim=1).item() / dim
    coverage_list.append(initial_coverage)
    activation_maps.append((x > 0).float().squeeze().numpy())
    
    # 逐层传播
    for i, layer in enumerate(layers):
        x = layer.forward(x)
        
        # 计算当前层的覆盖比例
        coverage = (x > 0).float().sum(dim=1).item() / dim
        coverage_list.append(coverage)
        
        # 记录激活状态
        activation_maps.append((x > 0).float().squeeze().numpy())
    
    return coverage_list, activation_maps

def visualize_comparison():
    """可视化LinearQ和Q_Block的堆叠效果对比"""
    parser = argparse.ArgumentParser(description="Visualize QuantizeLinear weight distributions")
    parser.add_argument(
        "--output-dir", type=str, default="./visualize_layer_stacking", help="Output directory for visualization results"
    )
    args = parser.parse_args()

    # 实验参数
    num_layers = 8
    dim = 384
    num_groups = 8
    shift_step = 1
    output_dir = args.output_dir
    
    # 运行两种类型的堆叠实验
    linearq_coverage, linearq_maps = simulate_layer_stacking(
        'linearq', num_layers, dim, num_groups, shift_step
    )
    
    qblock_coverage, qblock_maps = simulate_layer_stacking(
        'qblock', num_layers, dim, num_groups, shift_step
    )
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 覆盖比例对比图
    axes[0, 0].plot(range(len(linearq_coverage)), linearq_coverage, 
                   marker='o', label='LinearQ', linewidth=2)
    axes[0, 0].plot(range(len(qblock_coverage)), qblock_coverage, 
                   marker='s', label='Q_Block', linewidth=2)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Group Coverage Ratio')
    axes[0, 0].set_title('Coverage Comparison: LinearQ vs Q_Block')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. LinearQ激活热图
    im1 = axes[0, 1].imshow(np.array(linearq_maps).T, cmap='Blues', aspect='auto')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Feature Index')
    axes[0, 1].set_title('LinearQ: Activation Propagation')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. Q_Block激活热图
    im2 = axes[1, 0].imshow(np.array(qblock_maps).T, cmap='Reds', aspect='auto')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Feature Index')
    axes[1, 0].set_title('Q_Block: Activation Propagation')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 4. 差异热图
    diff_maps = np.array(qblock_maps) - np.array(linearq_maps)
    im3 = axes[1, 1].imshow(diff_maps.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Feature Index')
    axes[1, 1].set_title('Difference: Q_Block - LinearQ')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'layer_stacking_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("=== 堆叠实验统计信息 ===")
    print(f"实验配置: dim={dim}, groups={num_groups}, layers={num_layers}, shift_step={shift_step}")
    print(f"LinearQ最终覆盖比例: {linearq_coverage[-1]:.3f}")
    print(f"Q_Block最终覆盖比例: {qblock_coverage[-1]:.3f}")
    print(f"覆盖差异: {qblock_coverage[-1] - linearq_coverage[-1]:.3f}")
    
    # 分析传播速度
    linearq_full_coverage_layer = next(i for i, cov in enumerate(linearq_coverage) if cov >= 1.0)
    qblock_full_coverage_layer = next(i for i, cov in enumerate(qblock_coverage) if cov >= 1.0)
    
    print(f"LinearQ达到全覆盖的层数: {linearq_full_coverage_layer}")
    print(f"Q_Block达到全覆盖的层数: {qblock_full_coverage_layer}")
    print(f"传播速度差异: {linearq_full_coverage_layer - qblock_full_coverage_layer} 层")

if __name__ == "__main__":
    visualize_comparison()