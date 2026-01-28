import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from _quan_base import Qmodes, _ActQ, _Conv2dQ, _LinearQ

__all__ = ["Conv2dQ", "LinearQ", "ActQ", "print_linearq_groups"]


class FunQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, "alpha = {}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = (
            ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_weight * g)
            .sum()
            .unsqueeze(dim=0)
        )
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def print_linearq_groups(model):
    """打印所有LinearQ层的num_groups信息"""
    print("\n=== LinearQ Layers Group Information ===")
    total_params = 0
    total_flops = 0
    total_original_params = 0
    total_original_flops = 0

    for name, module in model.named_modules():
        if isinstance(module, LinearQ):
            if hasattr(module, "num_groups"):
                original_params = module.in_features * module.out_features
                original_flops = module.in_features * module.out_features
                current_params = module.in_features * module.out_features / module.num_groups
                current_flops = module.in_features * module.out_features / module.num_groups

                print(f"Layer: {name}")
                print(f"  in_features: {module.in_features}, out_features: {module.out_features}")
                print(f"  num_groups: {module.num_groups}")
                print(f"  Original Params: {original_params:,}, Current Params: {current_params:,.0f}")
                print(f"  Original FLOPs: {original_flops:,}, Current FLOPs: {current_flops:,.0f}")
                print(f"  Reduction Ratio: {module.num_groups}x")
                print("-" * 50)

                total_params += current_params
                total_flops += current_flops
                total_original_params += original_params
                total_original_flops += original_flops

    print("\n=== Summary ===")
    print(f"Total Original Params: {total_original_params:,}")
    print(f"Total Current Params: {total_params:,.0f}")
    print(f"Total Original FLOPs: {total_original_flops:,}")
    print(f"Total Current FLOPs: {total_flops:,.0f}")
    print(f"Overall Reduction Ratio: {total_original_params/total_params:.1f}x")
    print("=" * 50)


class Conv2dQ(_Conv2dQ):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nbits_w=8,
        mode=Qmodes.kernel_wise,
        **kwargs,
    ):
        super(Conv2dQ, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            nbits=nbits_w,
            mode=mode,
        )
        self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -(2 ** (self.nbits - 1))
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """
        Implementation according to paper.
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2),
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.

        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LinearQ(_LinearQ):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        nbits_w=4,
        linear_method="Normal",
        shift_step=0,
        num_groups=1,
        learnable_groups=False,
        **kwargs,
    ):
        super(LinearQ, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise
        )
        self.act = ActQ(in_features=in_features, nbits_a=nbits_w)
        self.linear_method = linear_method
        self.learnable_groups = learnable_groups

        # 支持可学习的num_groups
        if learnable_groups:
            # 将num_groups转换为可学习参数
            self.log_num_groups = torch.tensor(float(num_groups), requires_grad=False)
            self.num_groups = None  # 将在forward中动态计算
        else:
            self.num_groups = num_groups

        if self.linear_method == "shift" or self.linear_method == "block_diag":
            if learnable_groups:
                # 对于可学习的num_groups，我们使用最大可能的组数来初始化权重
                max_groups = min(in_features, out_features)
                self.shift_step = shift_step
                if self.shift_step < 0:
                    self.shift_step_shuffle = nn.Parameter(
                        torch.tensor(torch.randint(0, max_groups, (1,)).item(), dtype=torch.int),
                        requires_grad=False,
                    )
            else:
                self.shift_step = shift_step
                if self.shift_step < 0:
                    self.shift_step_shuffle = nn.Parameter(
                        torch.tensor(torch.randint(0, num_groups, (1,)).item(), dtype=torch.int),
                        requires_grad=False,
                    )
                self.num_groups = num_groups
                assert (
                    self.in_features % num_groups == 0
                ), f"in_features ({self.in_features}) must be divisible by num_groups ({num_groups})"
                assert (
                    self.out_features % num_groups == 0
                ), f"out_features ({self.out_features}) must be divisible by num_groups ({num_groups})"
                self.group_in_features = self.in_features // num_groups
                self.group_out_features = self.out_features // num_groups
                expected_total_size = self.num_groups * self.group_out_features * self.group_in_features
                current_total_size = self.weight.numel() if hasattr(self, "weight") and self.weight is not None else 0

                if current_total_size != expected_total_size and self.linear_method == "shift":
                    self.weight = torch.nn.Parameter(
                        torch.Tensor(self.group_out_features * self.num_groups, self.group_in_features)
                    )

    def get_effective_num_groups(self):
        """获取有效的num_groups，支持可学习和固定两种模式"""
        if self.learnable_groups:
            # 映射到[1, 12]区间
            num_groups = self.log_num_groups
            best_groups = num_groups
            min_diff = float("inf")

            # 搜索范围：从1到12
            for candidate in range(1, 13):
                if self.in_features % candidate == 0 and self.out_features % candidate == 0:
                    # 计算与初始分组数的差异
                    diff = abs(candidate - num_groups)
                    if diff < min_diff:
                        min_diff = diff
                        best_groups = candidate

            return best_groups
        else:
            return self.num_groups

    def create_block_diag_mask(self, num_groups):
        """创建分块对角矩阵的mask"""
        group_in = self.in_features // num_groups
        group_out = self.out_features // num_groups

        # 创建分块对角mask
        mask = torch.zeros(self.out_features, self.in_features)
        for i in range(num_groups):
            start_row = i * group_out
            end_row = (i + 1) * group_out
            start_col = i * group_in
            end_col = (i + 1) * group_in
            mask[start_row:end_row, start_col:end_col] = 1.0

        return mask

    def groups_shift(self, input_groups, shift_step=1):
        if shift_step > 0:
            if len(input_groups.shape) == 3:
                return torch.roll(input_groups, shifts=shift_step, dims=1)
            elif len(input_groups.shape) == 4:
                return torch.roll(input_groups, shifts=shift_step, dims=2)
        elif shift_step < 0:
            if len(input_groups.shape) == 3:
                return torch.roll(input_groups, shifts=self.shift_step_shuffle.item(), dims=1)
            elif len(input_groups.shape) == 4:
                return torch.roll(input_groups, shifts=self.shift_step_shuffle.item(), dims=2)
        else:
            return input_groups

    def forward(self, x):
        if self.alpha is None:
            # 对于未量化的权重，使用标准线性层
            out = F.linear(x, self.weight, self.bias)
            return out

        # 量化过程
        Qn = -(2 ** (self.nbits - 1))
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # 量化权重
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)

        # 根据linear_method选择计算方式
        if self.linear_method == "normal":
            out = F.linear(x, w_q, self.bias)
            return out

        elif self.linear_method == "shift" or self.linear_method == "block_diag":
            # 获取有效的num_groups
            effective_groups = self.get_effective_num_groups()

            if self.linear_method == "shift":
                weight_groups = w_q.view(
                    effective_groups, self.out_features // effective_groups, self.in_features // effective_groups
                )
            elif self.linear_method == "block_diag":
                mask = self.create_block_diag_mask(effective_groups).to(w_q.device)

            if len(x.shape) == 3:
                input_groups = x.view(x.shape[0], x.shape[1], effective_groups, self.in_features // effective_groups)
            elif len(x.shape) == 2:
                input_groups = x.view(x.shape[0], 1, effective_groups, self.in_features // effective_groups)
            else:
                raise ValueError(f"Unsupported x dimension: {len(x.shape)}, x shape: {x.shape}")

            input_groups = self.groups_shift(input_groups, shift_step=self.shift_step)

            if self.linear_method == "shift":
                output_groups = torch.einsum("bsgi,goi->bsgo", input_groups, weight_groups)
                if len(x.shape) == 3:
                    out = output_groups.reshape(x.shape[0], x.shape[1], self.out_features)
                elif len(x.shape) == 2:
                    out = output_groups.reshape(x.shape[0], self.out_features)
            elif self.linear_method == "block_diag":
                if len(x.shape) == 3:
                    shifted_input = input_groups.reshape(x.shape[0], x.shape[1], -1)
                elif len(x.shape) == 2:
                    shifted_input = input_groups.reshape(x.shape[0], -1)
                out = F.linear(shifted_input, w_q * mask, self.bias)

            return out

    def get_complexity_reduction(self):
        """计算计算量和参数量的减少比例"""
        effective_groups = self.get_effective_num_groups()
        original_complexity = self.in_features * self.out_features
        current_complexity = original_complexity / effective_groups
        return current_complexity / original_complexity, effective_groups


class ActQ(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        super(ActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -(2 ** (self.nbits - 1))
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2**self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn)
            )
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -(2 ** (self.nbits - 1))
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2**self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)
        # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        if len(x.shape) == 2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape) == 4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        return x
