from pprint import pprint

import torch


def print_linearq_groups_from_checkpoint(checkpoint_path):
    """
    从checkpoint中分析并打印所有LinearQ层的num_groups信息

    Args:
        checkpoint_path: checkpoint文件路径
    """
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"成功加载checkpoint: {checkpoint_path}\n")

        if "model" not in checkpoint:
            print("错误: checkpoint中未找到模型状态字典")
            return

        model_state_dict = checkpoint["model"]

        print("=== LinearQ层num_groups分析 ===")
        print("正在分析模型中的LinearQ层分组信息...\n")

        # 统计所有包含num_groups信息的层
        linearq_layers = {}
        total_layers = 0
        layers_with_groups = 0

        for key, value in model_state_dict.items():
            total_layers += 1

            # 查找包含num_groups信息的键
            if "num_groups" in key:
                layer_name = key.replace(".num_groups", "")
                linearq_layers[layer_name] = {
                    "num_groups": value.item() if hasattr(value, "item") else value,
                    "weight_key": None,
                    "in_features": None,
                    "out_features": None,
                }
                layers_with_groups += 1

            # 查找包含log_num_groups信息的键（可学习分组）
            elif "log_num_groups" in key:
                layer_name = key.replace(".log_num_groups", "")
                log_num_groups = value.item() if hasattr(value, "item") else value
                num_groups = int(torch.exp(torch.tensor(log_num_groups)).round().item())
                linearq_layers[layer_name] = {
                    "num_groups": num_groups,
                    "log_num_groups": log_num_groups,
                    "weight_key": None,
                    "in_features": None,
                    "out_features": None,
                    "learnable": True,
                }
                layers_with_groups += 1

        # 查找对应的权重张量来获取特征维度信息
        for key in model_state_dict.keys():
            if ".weight" in key and not ("num_groups" in key or "log_num_groups" in key):
                # 尝试匹配层名
                for layer_name in linearq_layers.keys():
                    if key.startswith(layer_name) and key.endswith(".weight"):
                        linearq_layers[layer_name]["weight_key"] = key
                        weight_tensor = model_state_dict[key]
                        if len(weight_tensor.shape) == 2:
                            linearq_layers[layer_name]["out_features"] = weight_tensor.shape[0]
                            linearq_layers[layer_name]["in_features"] = weight_tensor.shape[1]
                        break

        # 打印结果
        if linearq_layers:
            print(f"找到 {len(linearq_layers)} 个LinearQ层的分组信息:\n")

            total_original_params = 0
            total_current_params = 0
            total_original_flops = 0
            total_current_flops = 0

            for layer_name, info in sorted(linearq_layers.items()):
                num_groups = info["num_groups"]
                in_features = info["in_features"]
                out_features = info["out_features"]
                learnable = info.get("learnable", False)

                if in_features and out_features:
                    original_params = in_features * out_features
                    original_flops = in_features * out_features
                    current_params = original_params / num_groups
                    current_flops = original_flops / num_groups

                    total_original_params += original_params
                    total_current_params += current_params
                    total_original_flops += original_flops
                    total_current_flops += current_flops

                    print(f"层名: {layer_name}")
                    print(f"  输入特征: {in_features}, 输出特征: {out_features}")
                    print(f"  num_groups: {num_groups} {'(可学习)' if learnable else ''}")
                    if learnable:
                        print(f"  log_num_groups: {info.get('log_num_groups', 'N/A'):.4f}")
                    print(f"  原始参数量: {original_params:,}")
                    print(f"  当前参数量: {current_params:,.0f}")
                    print(f"  原始FLOPs: {original_flops:,}")
                    print(f"  当前FLOPs: {current_flops:,.0f}")
                    print(f"  减少比例: {num_groups:.1f}x")
                    print("-" * 60)
                else:
                    print(f"层名: {layer_name}")
                    print(f"  num_groups: {num_groups} {'(可学习)' if learnable else ''}")
                    if learnable:
                        print(f"  log_num_groups: {info.get('log_num_groups', 'N/A'):.4f}")
                    print("  警告: 无法获取特征维度信息")
                    print("-" * 60)

            # 打印汇总信息
            if total_original_params > 0:
                print("\n=== 汇总信息 ===")
                print(f"总原始参数量: {total_original_params:,}")
                print(f"总当前参数量: {total_current_params:,.0f}")
                print(f"总原始FLOPs: {total_original_flops:,}")
                print(f"总当前FLOPs: {total_current_flops:,.0f}")
                overall_reduction = total_original_params / total_current_params if total_current_params > 0 else 1
                print(f"整体减少比例: {overall_reduction:.1f}x")
        else:
            print("未找到LinearQ层的分组信息")

        print(f"\n分析完成。总共检查了 {total_layers} 个层，其中 {layers_with_groups} 个层包含分组信息。")

    except FileNotFoundError:
        print(f"错误: 未找到checkpoint文件 {checkpoint_path}")
    except Exception as e:
        print(f"分析LinearQ分组信息失败: {str(e)}")


def print_checkpoint_args(checkpoint_path):
    """
    打印checkpoint文件中保存的参数信息

    Args:
        checkpoint_path: checkpoint文件路径
    """
    try:
        # 加载checkpoint（指定map_location避免设备不匹配问题）
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"成功加载checkpoint: {checkpoint_path}\n")

        # 打印checkpoint包含的所有键
        print(f"checkpoint包含的内容: {list(checkpoint.keys())}\n")

        # 逐个解析并打印各部分信息
        if "epoch" in checkpoint:
            print(f"训练轮次 (epoch): {checkpoint['epoch']}\n")

        if "args" in checkpoint:
            print("训练参数 (args):")
            # 将Namespace转换为字典并美化打印
            pprint(vars(checkpoint["args"]), indent=4)
            print()

        if "model" in checkpoint:
            model_keys = list(checkpoint["model"].keys())
            print("模型状态字典 (model.state_dict):")
            print(f"  包含 {len(model_keys)} 个键值对")
            print(f"  前5个键示例: {model_keys[:5]}...\n")

        if "optimizer" in checkpoint:
            optim_dict = checkpoint["optimizer"]
            print("优化器状态字典 (optimizer.state_dict):")
            print(f"  参数组数量 (param_groups): {len(optim_dict.get('param_groups', []))}")
            print(f"  状态键数量 (state): {len(optim_dict.get('state', []))}\n")

        if "lr_scheduler" in checkpoint:
            scheduler_keys = list(checkpoint["lr_scheduler"].keys())
            print("学习率调度器状态字典 (lr_scheduler.state_dict):")
            print(f"  包含键: {scheduler_keys}\n")

        if "scaler" in checkpoint:
            scaler_keys = list(checkpoint["scaler"].keys())
            print("损失缩放器状态字典 (scaler.state_dict):")
            print(f"  包含键: {scaler_keys}\n")

    except FileNotFoundError:
        print(f"错误: 未找到checkpoint文件 {checkpoint_path}")
    except Exception as e:
        print(f"加载checkpoint失败: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="打印checkpoint参数")
    # default="teacher/cifar/1W_32A.pth"
    # default="teacher/flower/1W1A.pth",
    # default="result/flower_w8a8_train_v1.4/best_checkpoint.pth",
    parser.add_argument(
        "--checkpoint_path",
        default="result/flower_w8a8_train_v1.4/best_checkpoint.pth",
        type=str,
        help="checkpoint文件路径",
    )
    parser.add_argument("--analyze_groups", action="store_true", help="分析LinearQ层的num_groups信息")

    args = parser.parse_args()

    if args.analyze_groups:
        print_linearq_groups_from_checkpoint(args.checkpoint_path)
    else:
        print_checkpoint_args(args.checkpoint_path)
