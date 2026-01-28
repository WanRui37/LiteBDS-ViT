# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

from Quant import LinearQ


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distillation_type: str,
        alpha: float,
        tau: float,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    # We provide the teacher's targets in log probability because we use log_target=True
                    # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                    # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.numel()
            )
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class DistillationLoss_shift(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distillation_type: str,
        alpha: float,
        tau: float,
        rank_reg_weight,
    ):
        super().__init__()
        self.rank_reg_weight = rank_reg_weight
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.model = model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    # We provide the teacher's targets in log probability because we use log_target=True
                    # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                    # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.numel()
            )
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        # 新增：计算有效秩正则项
        rank_reg = 0.0
        for name, param in self.model.named_parameters():
            # if any(key in name for key in ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']) and 'weight' in name and param.dim() == 2:
            if any(key in name for key in ["attn.qkv", "mlp.fc1"]) and "weight" in name and param.dim() == 2:
                rank = self.effective_rank_ratio(param)
                rank_reg += rank
        rank_reg *= self.rank_reg_weight
        # print(f"Rank reg: {rank_reg.item()}")

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha + rank_reg
        return loss

    def effective_rank_ratio(
        self, weight: torch.Tensor, group: int = 1, eps: float = 1e-6, cal_type: str = "classical"
    ) -> torch.Tensor:
        if cal_type != "classical":
            raise NotImplementedError("当前仅支持 'classical' 计算类型")

        out_g_total = weight.shape[0]
        in_g = weight.shape[1]

        assert out_g_total % group == 0, f"权重矩阵列数 {out_g_total} 不能被分组数 {group} 整除"
        out_g = out_g_total // group

        weight_blocks = weight.view(group, out_g, in_g)  # shape: [g, out_g, in_g]
        _, S, _ = torch.linalg.svd(weight_blocks, full_matrices=False)  # shape: [g, k]，k=min(out_g, in_g)
        S_filtered = S.masked_fill(S < eps, 0.0)
        global_S = S_filtered.flatten()
        global_S = global_S[global_S > eps]  # 最终保留的有效奇异值

        # 极端情况：无有效奇异值
        if len(global_S) == 0:
            return torch.tensor(0.0, device=weight.device)

        s_sq = global_S.square()  # 向量化平方（比**2更高效）
        s_sq_sum = s_sq.sum()

        if s_sq_sum < eps:
            return torch.tensor(0.0, device=weight.device)

        p = s_sq / s_sq_sum
        p_clipped = p.clamp(min=eps, max=1.0)  # 向量化clip（比torch.clamp快）
        entropy = -(p_clipped * p_clipped.log()).sum()  # 向量化log+求和

        return entropy.exp()


class DistillationLoss_bi(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        # self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels, teacher_outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        # with torch.no_grad():
        #     teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    # We provide the teacher's targets in log probability because we use log_target=True
                    # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                    # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.numel()
            )
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class BlockDiagonalLoss(torch.nn.Module):
    """
    基于梯度感知的分块对角矩阵损失函数
    先进特性：
    1. 梯度重要性评估 - 通过梯度L2范数评估参数重要性
    2. 自适应分组策略 - 根据梯度大小动态调整分组数量
    3. 梯度累积统计 - 训练过程中累积梯度信息
    4. 温度调节的软分组分配 - 使用softmax进行平滑的组分配
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distillation_type: str,
        alpha: float,
        tau: float,
        complexity_weight: float = 1,
        complexity_bias: float = 0.05,
        gradient_weight: float = 0,
        target_reduction: float = 8.0,
        gradient_accumulation_steps: int = 100,
        temperature: float = 1.0,
        min_groups: int = 1,
        max_groups: int = 12,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.model = model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.complexity_weight = complexity_weight
        self.complexity_bias = complexity_bias
        self.gradient_weight = gradient_weight
        self.target_reduction = target_reduction
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.temperature = temperature
        self.min_groups = min_groups
        self.max_groups = max_groups

        # 梯度统计信息
        self.gradient_stats = {}
        self.step_counter = 0

        # 注册梯度hook
        self.register_gradient_hooks()

    def register_gradient_hooks(self):
        """为所有LinearQ层注册梯度hook"""
        for name, module in self.model.named_modules():
            if isinstance(module, LinearQ) and any(
                key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
            ):
                # 初始化梯度统计
                self.gradient_stats[name] = {
                    "gradient_norms": [],
                    "gradient_variances": [],
                    "importance_scores": [],
                    "optimal_groups": (
                        module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1
                    ),
                }

                # 注册梯度hook
                module.weight.register_hook(lambda grad, name=name: self.gradient_hook(grad, name))

    def gradient_hook(self, grad, layer_name):
        """梯度hook函数，收集梯度信息"""
        if grad is not None and self.training:
            # 计算梯度L2范数
            grad_norm = grad.norm(p=2).item()

            # 计算梯度方差（重要性指标）
            grad_variance = grad.var().item() if grad.numel() > 1 else 0.0

            # 计算梯度重要性分数（结合范数和方差）
            importance_score = grad_norm * (1 + 0.1 * grad_variance)

            # 存储梯度统计信息
            if layer_name in self.gradient_stats:
                stats = self.gradient_stats[layer_name]
                stats["gradient_norms"].append(grad_norm)
                stats["gradient_variances"].append(grad_variance)
                stats["importance_scores"].append(importance_score)

                # 限制存储的历史步数
                if len(stats["gradient_norms"]) > self.gradient_accumulation_steps:
                    stats["gradient_norms"] = stats["gradient_norms"][-self.gradient_accumulation_steps :]
                    stats["gradient_variances"] = stats["gradient_variances"][-self.gradient_accumulation_steps :]
                    stats["importance_scores"] = stats["importance_scores"][-self.gradient_accumulation_steps :]

    def calculate_gradient_aware_groups(self, layer_name, module):
        """基于梯度信息计算最优分组数量

        改进版：
        1. 重新设计重要性分数计算方式，使用梯度范数的平均值作为重要性
        2. 采用非线性映射函数，使重要性对分组数的影响更敏感
        3. 增加分组数多样性，避免过度集中在高分组数
        """
        if layer_name not in self.gradient_stats:
            return module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1

        stats = self.gradient_stats[layer_name]
        if len(stats["importance_scores"]) < 10:  # 最少需要一些梯度信息
            return module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1

        # 计算平均重要性分数
        avg_importance = sum(stats["importance_scores"]) / len(stats["importance_scores"])

        # 收集所有层的平均重要性分数
        all_importances = [
            sum(s["importance_scores"]) / len(s["importance_scores"])
            for s in self.gradient_stats.values()
            if len(s["importance_scores"]) >= 10
        ]

        if not all_importances:
            return module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1

        # 计算重要性分数的分布参数
        import numpy as np

        mean_importance = np.mean(all_importances)
        std_importance = np.std(all_importances) if len(all_importances) > 1 else 1.0

        # 使用Z-score标准化重要性分数，使其更符合正态分布
        if std_importance > 0:
            normalized_importance = (avg_importance - mean_importance) / std_importance
            # 将Z-score转换到[0, 1]区间，使用sigmoid函数增加非线性
            normalized_importance = 1.0 / (1.0 + np.exp(-normalized_importance))
        else:
            normalized_importance = 0.5

        # 计算分组数映射因子，使用指数函数增加非线性
        # 这样可以让重要性分数高的层更容易获得较小的分组数
        mapping_factor = np.exp(normalized_importance * np.log(self.max_groups))

        # 计算目标分组数：重要性越高，分组数越少
        # 这里使用了非线性映射，使得分组数分布更均匀
        target_groups_float = self.max_groups / mapping_factor
        target_groups_float = max(self.min_groups, min(self.max_groups, target_groups_float))

        # 确保分组数量能被输入输出维度整除
        optimal_groups = max(self.min_groups, min(self.max_groups, int(round(target_groups_float))))

        # 调整分组数量以确保整除性
        # 搜索范围扩大，优先选择接近目标值的可整除分组数
        possible_groups = []
        for g in range(self.min_groups, self.max_groups + 1):
            if module.in_features % g == 0 and module.out_features % g == 0:
                possible_groups.append(g)

        if not possible_groups:
            optimal_groups = self.min_groups
        else:
            # 选择最接近目标值的分组数
            optimal_groups = min(possible_groups, key=lambda x: abs(x - target_groups_float))

        # 更新统计信息
        stats["optimal_groups"] = optimal_groups

        return optimal_groups

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == "none":
            distillation_loss = 0.0
        else:
            if outputs_kd is None:
                raise ValueError(
                    "When knowledge distillation is enabled, the model is "
                    "expected to return a Tuple[Tensor, Tensor] with the output of the "
                    "class_token and the dist_token"
                )

            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            if self.distillation_type == "soft":
                T = self.tau
                distillation_loss = (
                    F.kl_div(
                        F.log_softmax(outputs_kd / T, dim=1),
                        F.log_softmax(teacher_outputs / T, dim=1),
                        reduction="sum",
                        log_target=True,
                    )
                    * (T * T)
                    / outputs_kd.numel()
                )
            elif self.distillation_type == "hard":
                distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        # 总损失 = 基础损失 + 蒸馏损失
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        self.step_counter += 1

        return loss

    def calculate_complexity_loss(self):
        """计算计算复杂度和参数量的正则化损失"""
        total_original_complexity = 0.0
        total_current_complexity = 0.0

        # 遍历所有LinearQ层，计算总复杂度
        for name, module in self.model.named_modules():
            if (
                isinstance(module, LinearQ)
                and hasattr(module, "get_complexity_reduction")
                and any(key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
            ):
                # 获取当前分组和最优分组
                current_groups = (
                    module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1
                )

                # 原始复杂度（全连接层）
                original_complexity = module.in_features * module.out_features

                # 当前复杂度（基于梯度感知的分组）
                current_complexity = original_complexity / current_groups

                total_original_complexity += original_complexity
                total_current_complexity += current_complexity

        # 如果没有LinearQ层，返回0
        if total_original_complexity == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        # 整体复杂度比例
        overall_ratio = total_current_complexity / total_original_complexity
        target_overall_ratio = (1.0 / self.target_reduction) - self.complexity_bias

        if overall_ratio <= target_overall_ratio:
            complexity_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        else:
            complexity_loss = F.mse_loss(
                torch.tensor(overall_ratio, device=next(self.model.parameters()).device),
                torch.tensor(target_overall_ratio, device=next(self.model.parameters()).device),
            )

        # complexity_loss = F.mse_loss(
        #     torch.tensor(overall_ratio, device=next(self.model.parameters()).device),
        #     torch.tensor(target_overall_ratio, device=next(self.model.parameters()).device),
        # )
        print(
            f"""Complexity Loss: {complexity_loss.item():.6f}, "
              f"Overall Ratio: {overall_ratio:.6f}, "
              f"Target Ratio: {target_overall_ratio:.6f}"""
        )
        return complexity_loss

    def calculate_gradient_aware_loss(self):
        """计算梯度感知的分组损失"""
        if self.step_counter < 10:  # 前几步需要积累梯度信息
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        total_loss = 0.0
        layer_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, LinearQ) and any(
                key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
            ):
                if name in self.gradient_stats and len(self.gradient_stats[name]["importance_scores"]) >= 10:
                    stats = self.gradient_stats[name]

                    current_groups = (
                        module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1
                    )
                    optimal_groups = self.calculate_gradient_aware_groups(name, module)

                    # 计算分组差异损失（鼓励接近最优分组）
                    group_diff = abs(current_groups - optimal_groups) / max(current_groups, optimal_groups)
                    layer_loss = F.mse_loss(
                        torch.tensor(current_groups / self.max_groups, device=next(self.model.parameters()).device),
                        torch.tensor(optimal_groups / self.max_groups, device=next(self.model.parameters()).device),
                    )

                    total_loss += layer_loss
                    layer_count += 1

        if layer_count == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        print(f"Gradient-Aware Loss: {total_loss.item() / layer_count:.6f}")

        return total_loss / layer_count

    def print_gradient_aware_info(self):
        """打印梯度感知分组信息"""
        print("\n=== Gradient-Aware Block Diagonal Information ===")
        print(f"Accumulation Steps: {self.step_counter}")

        for name, stats in self.gradient_stats.items():
            if len(stats["importance_scores"]) >= 10:
                avg_importance = sum(stats["importance_scores"]) / len(stats["importance_scores"])
                current_groups = None

                # 获取当前模块的分组信息
                for module_name, module in self.model.named_modules():
                    if module_name == name and isinstance(module, LinearQ):
                        current_groups = (
                            module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1
                        )
                        break

                print(f"Layer: {name}")
                print(f"  Importance Score: {avg_importance:.4f}")
                print(f"  Current Groups: {current_groups}")
                print(f"  Optimal Groups: {stats['optimal_groups']}")
                print(f"  Gradient Norm: {sum(stats['gradient_norms'])/len(stats['gradient_norms']):.6f}")
                print("-" * 50)

    def get_current_groups(self):
        current_groups_dict = {}
        total_original_complexity = 0.0
        total_current_complexity = 0.0

        # 遍历所有LinearQ层，计算最优分组
        for name, module in self.model.named_modules():
            if (
                isinstance(module, LinearQ)
                and hasattr(module, "get_complexity_reduction")
                and any(key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
            ):
                current_groups = (
                    module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1
                )
                current_groups_dict[name] = current_groups

                original_complexity = module.in_features * module.out_features
                current_complexity = original_complexity / current_groups

                total_original_complexity += original_complexity
                total_current_complexity += current_complexity
        overall_ratio = total_current_complexity / total_original_complexity

        return current_groups_dict, overall_ratio

    def update_linearq_groups(self):
        """基于梯度信息和全局复杂度约束更新LinearQ层的分组数量"""
        # 计算全局最优分组配置
        global_optimal_groups = self.calculate_global_optimal_groups()

        for name, module in self.model.named_modules():
            if (
                isinstance(module, LinearQ)
                and hasattr(module, "learnable_groups")
                and module.learnable_groups
                and any(key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
            ):
                if name in global_optimal_groups:
                    optimal_groups = global_optimal_groups[name]
                    # print(f"Optimal Groups for {name}: {optimal_groups}")

                    # 更新统计信息
                    if name in self.gradient_stats:
                        self.gradient_stats[name]["optimal_groups"] = optimal_groups

                    # 更新可学习的num_groups参数
                    with torch.no_grad():
                        module.log_num_groups.data.copy_(torch.tensor(float(optimal_groups)))

    def calculate_global_optimal_groups(self):
        """计算全局最优分组配置，确保整体复杂度不超过目标值

        这是一个全局优化问题：
        - 目标：在满足整体复杂度约束的前提下，最大化模型性能
        - 约束：total_complexity <= target_complexity
        - 优化变量：每层的分组数
        - 优先级：基于梯度重要性，重要层应保持较少的分组数
        - 稳定性：加入双向调整机制和稳定性约束，防止分组数快速下降
        """
        # 首先计算每层的梯度最优分组和复杂度贡献
        layer_info = []
        total_original_complexity = 0.0
        target_overall_ratio = (1.0 / self.target_reduction) - self.complexity_bias
        # print(f"Target Overall Ratio: {target_overall_ratio:.6f}")

        # 收集所有LinearQ层的信息
        for name, module in self.model.named_modules():
            if (
                isinstance(module, LinearQ)
                and hasattr(module, "get_complexity_reduction")
                and any(key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
            ):
                # 计算梯度最优分组数
                grad_optimal_groups = self.calculate_gradient_aware_groups(name, module)

                # 获取当前实际分组数作为参考
                current_actual_groups = (
                    module.get_effective_num_groups() if hasattr(module, "get_effective_num_groups") else 1
                )

                # 计算当前层的原始复杂度
                original_complexity = module.in_features * module.out_features
                total_original_complexity += original_complexity

                # 计算重要性分数（用于优先级排序）
                importance_score = 1.0
                if name in self.gradient_stats and len(self.gradient_stats[name]["importance_scores"]) >= 10:
                    importance_score = sum(self.gradient_stats[name]["importance_scores"]) / len(
                        self.gradient_stats[name]["importance_scores"]
                    )

                # 获取当前层支持的分组数范围
                min_groups = 1
                max_groups = 12  # 从get_effective_num_groups中获取的最大值

                # 收集所有可能的分组数选项及其对应的复杂度
                possible_groups = []
                for g in range(min_groups, max_groups + 1):
                    if module.in_features % g == 0 and module.out_features % g == 0:
                        complexity = original_complexity / g
                        possible_groups.append((g, complexity))

                layer_info.append(
                    {
                        "name": name,
                        "module": module,
                        "original_complexity": original_complexity,
                        "importance_score": importance_score,
                        "grad_optimal_groups": grad_optimal_groups,
                        "current_actual_groups": current_actual_groups,
                        "possible_groups": possible_groups,
                    }
                )

        # 计算目标复杂度
        target_complexity = total_original_complexity * target_overall_ratio

        # 复杂度缓冲区间：允许复杂度在目标值的90%-100%之间波动
        # 确保复杂度不会超过目标值，同时避免微小波动导致频繁调整
        complexity_lower_bound = target_complexity * 0.9

        # 按重要性分数排序（从高到低）
        layer_info.sort(key=lambda x: x["importance_score"], reverse=True)

        # 初始配置：使用梯度最优分组数作为初始值，保持动态调整能力
        current_groups = {}
        total_current_complexity = 0.0

        for layer in layer_info:
            # 使用梯度最优分组数作为初始值，确保每次都能基于最新梯度信息调整
            current_groups[layer["name"]] = layer["grad_optimal_groups"]
            # 找到对应分组数的复杂度
            for g, c in layer["possible_groups"]:
                if g == layer["grad_optimal_groups"]:
                    total_current_complexity += c
                    break

        # 计算当前复杂度与目标复杂度的差异
        complexity_diff = total_current_complexity - target_complexity
        # print(f"complexity_diff: {complexity_diff:.6f}")

        # 如果在缓冲区间内，直接返回当前分组配置（增加稳定性）
        if complexity_lower_bound <= total_current_complexity <= target_complexity:
            # print(f"当前复杂度在缓冲区间内: {total_current_complexity:.6f}，直接返回")
            return current_groups

        # 否则，需要调整分组数以满足约束
        # print(f"当前复杂度: {total_current_complexity:.6f}, 目标复杂度: {target_complexity:.6f}, 差异: {complexity_diff:.6f}")

        # 调整机制
        if total_current_complexity > target_complexity:  # 当前复杂度超过目标，需要减少复杂度（增加分组数）
            # 按重要性从低到高调整分组数（优先调整不重要的层）
            for layer in reversed(layer_info):
                if total_current_complexity <= target_complexity:
                    break

                current_group = current_groups[layer["name"]]

                # 找到当前分组数对应的复杂度
                current_layer_complexity = 0.0
                for g, c in layer["possible_groups"]:
                    if g == current_group:
                        current_layer_complexity = c
                        break

                # 尝试增加分组数（减少复杂度），但限制每次调整的幅度
                # 按分组数从大到小排序
                possible_groups_sorted = sorted(layer["possible_groups"], key=lambda x: x[0], reverse=True)

                # 最大调整幅度：不超过当前分组数的2倍
                max_adjusted_group = min(current_group * 2, max(g for g, _ in layer["possible_groups"]))

                for new_group, new_layer_complexity in possible_groups_sorted:
                    if new_group <= current_group or new_group > max_adjusted_group:
                        continue  # 只考虑增加分组数，且不超过最大调整幅度

                    # 计算调整后的总复杂度
                    new_total_complexity = total_current_complexity - current_layer_complexity + new_layer_complexity

                    if new_total_complexity <= target_complexity:
                        # 这个分组数可以满足约束，更新配置
                        current_groups[layer["name"]] = new_group
                        total_current_complexity = new_total_complexity
                        # print(f"调整层 {layer['name']} 的分组数从 {current_group} 到 {new_group}，总复杂度: {total_current_complexity:.6f}")
                        break
                    else:
                        # 这个分组数仍然不满足约束，尝试更大的分组数
                        current_groups[layer["name"]] = new_group
                        total_current_complexity = new_total_complexity
                        # print(f"调整层 {layer['name']} 的分组数从 {current_group} 到 {new_group}，总复杂度: {total_current_complexity:.6f}")
                        current_group = new_group
                        current_layer_complexity = new_layer_complexity
        else:  # 当前复杂度低于下限，需要增加复杂度（减少分组数）
            # 按重要性从高到低调整分组数（优先调整重要的层）
            for layer in layer_info:
                if total_current_complexity >= complexity_lower_bound:
                    break

                current_group = current_groups[layer["name"]]

                # 找到当前分组数对应的复杂度
                current_layer_complexity = 0.0
                for g, c in layer["possible_groups"]:
                    if g == current_group:
                        current_layer_complexity = c
                        break

                # 尝试减少分组数（增加复杂度），但限制每次调整的幅度
                # 按分组数从小到大排序
                possible_groups_sorted = sorted(layer["possible_groups"], key=lambda x: x[0])

                # 最小分组数保护，确保分组数不会太小
                min_allowed_group = max(1, layer["grad_optimal_groups"] // 2)  # 不小于梯度最优分组数的一半

                # 最大调整幅度：不低于当前分组数的一半
                min_adjusted_group = max(min_allowed_group, current_group // 2)

                for new_group, new_layer_complexity in possible_groups_sorted:
                    if new_group >= current_group or new_group < min_adjusted_group:
                        continue  # 只考虑减少分组数，且不低于最小调整幅度

                    # 计算调整后的总复杂度
                    new_total_complexity = total_current_complexity - current_layer_complexity + new_layer_complexity

                    if new_total_complexity >= complexity_lower_bound:
                        # 这个分组数可以满足约束，更新配置
                        current_groups[layer["name"]] = new_group
                        total_current_complexity = new_total_complexity
                        # print(f"调整层 {layer['name']} 的分组数从 {current_group} 到 {new_group}，总复杂度: {total_current_complexity:.6f}")
                        break
                    else:
                        # 这个分组数仍然不满足约束，尝试更小的分组数
                        current_groups[layer["name"]] = new_group
                        total_current_complexity = new_total_complexity
                        # print(f"调整层 {layer['name']} 的分组数从 {current_group} 到 {new_group}，总复杂度: {total_current_complexity:.6f}")
                        current_group = new_group
                        current_layer_complexity = new_layer_complexity

        # 最终检查：确保所有分组数都在合理范围内
        for name, group in current_groups.items():
            min_groups = 1
            max_groups = 12
            if group < min_groups:
                current_groups[name] = min_groups
                # print(f"保护分组数下限: {name} 从 {group} 调整到 {min_groups}")
            elif group > max_groups:
                current_groups[name] = max_groups
                # print(f"保护分组数上限: {name} 从 {group} 调整到 {max_groups}")

        return current_groups
