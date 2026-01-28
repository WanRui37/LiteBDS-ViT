#!/usr/bin/env python3
"""
生成--layer_groups参数的JSON格式
输入格式：
  module.blocks.0.attn.qkv: 1
  module.blocks.0.attn.proj: 1
  module.blocks.0.mlp.fc1: 1
  module.blocks.0.mlp.fc2: 1
  ...
输出格式：
  {"attn_qkv":[1,4,2,...],"attn_proj":[1,8,4,...],"mlp_fc1":[1,8,8,...],"mlp_fc2":[1,8,8,...]}
"""
import json

# 用户提供的输入文本
input_text = """
    module.blocks.0.attn.qkv: 1
    module.blocks.0.attn.proj: 1
    module.blocks.0.mlp.fc1: 1
    module.blocks.0.mlp.fc2: 1
    module.blocks.1.attn.qkv: 4
    module.blocks.1.attn.proj: 8
    module.blocks.1.mlp.fc1: 8
    module.blocks.1.mlp.fc2: 8
    module.blocks.2.attn.qkv: 2
    module.blocks.2.attn.proj: 4
    module.blocks.2.mlp.fc1: 8
    module.blocks.2.mlp.fc2: 8
    module.blocks.3.attn.qkv: 3
    module.blocks.3.attn.proj: 4
    module.blocks.3.mlp.fc1: 8
    module.blocks.3.mlp.fc2: 8
    module.blocks.4.attn.qkv: 4
    module.blocks.4.attn.proj: 8
    module.blocks.4.mlp.fc1: 8
    module.blocks.4.mlp.fc2: 8
    module.blocks.5.attn.qkv: 2
    module.blocks.5.attn.proj: 4
    module.blocks.5.mlp.fc1: 8
    module.blocks.5.mlp.fc2: 8
    module.blocks.6.attn.qkv: 4
    module.blocks.6.attn.proj: 4
    module.blocks.6.mlp.fc1: 8
    module.blocks.6.mlp.fc2: 8
    module.blocks.7.attn.qkv: 4
    module.blocks.7.attn.proj: 8
    module.blocks.7.mlp.fc1: 8
    module.blocks.7.mlp.fc2: 8
    module.blocks.8.attn.qkv: 3
    module.blocks.8.attn.proj: 4
    module.blocks.8.mlp.fc1: 8
    module.blocks.8.mlp.fc2: 8
    module.blocks.9.attn.qkv: 3
    module.blocks.9.attn.proj: 8
    module.blocks.9.mlp.fc1: 8
    module.blocks.9.mlp.fc2: 8
    module.blocks.10.attn.qkv: 4
    module.blocks.10.attn.proj: 8
    module.blocks.10.mlp.fc1: 8
    module.blocks.10.mlp.fc2: 8
    module.blocks.11.attn.qkv: 3
    module.blocks.11.attn.proj: 8
    module.blocks.11.mlp.fc1: 8
    module.blocks.11.mlp.fc2: 8
"""


def generate_layer_groups(input_text):
    # 初始化分组字典
    layer_groups = {"attn_qkv": [], "attn_proj": [], "mlp_fc1": [], "mlp_fc2": []}

    # 解析输入文本
    lines = input_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 分割模块路径和分组数
        module_path, group_num = line.split(":")
        module_path = module_path.strip()
        group_num = int(group_num.strip())

        # 解析模块路径
        parts = module_path.split(".")
        block_idx = int(parts[2])  # 获取block索引
        layer_type = parts[3]  # attn或mlp
        layer_name = parts[4]  # qkv/proj 或 fc1/fc2

        # 构建键名
        key = f"{layer_type}_{layer_name}"

        # 确保列表长度足够
        while len(layer_groups[key]) <= block_idx:
            layer_groups[key].append(0)

        # 设置分组数
        layer_groups[key][block_idx] = group_num

    return layer_groups


def main():
    # 生成layer_groups
    layer_groups = generate_layer_groups(input_text)

    # 转换为JSON字符串，确保没有空格
    json_str = json.dumps(layer_groups, separators=(",", ":"))

    # 生成适合shell脚本使用的转义JSON字符串（对双引号进行转义）
    shell_escaped_json = json_str.replace('"', '\\"')

    print("生成的--layer_groups参数值：")
    print(json_str)

    print("\n在shell脚本中使用的转义格式：")
    print(shell_escaped_json)

    print("\n在命令行中使用示例：")
    print(f"python main.py --model fourbits_deit_small_patch16_224 --layer_groups '{json_str}' ...")

    print("\n在shell脚本中使用示例：")
    print(f'python main.py --model fourbits_deit_small_patch16_224 --layer_groups "{shell_escaped_json}" ...')


if __name__ == "__main__":
    main()
