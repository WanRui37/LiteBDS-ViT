# LiteBDS-ViT

![arch](./pic/arch.png)

Official implementation of **"Structural Optimization Framework for Efficient Low-Precision Vision Transformers"**.

## Environment Setup

| Library      | Version |
| ------------ | ------: |
| PyTorch      |   2.6.0 |
| TorchVision  |  0.15.0 |
| timm         |   1.0.9 |
| Transformers |  4.52.3 |

* See `environment.yml` for the remaining dependencies.

## Two-Stage Train

- Stage 1: Group Search Without Shift

```bash
CUDA_VISIBLE_DEVICES=<GPU_DEVICE_ID> python -m torch.distributed.launch --master_port=<FREE_PORT> --nproc_per_node=<NUM_GPU_PER_NODE> --use_env main.py --model fourbits_deit_small_patch16_224 --epochs 600 --warmup-epochs 0 --weight-decay 0.05 --batch-size 128 --data-path <DATASET_ROOT_PATH> --data-set CIFAR --lr 0.001 --output_dir <EXPERIMENT_OUTPUT_PATH> --distillation-type hard --teacher-model deit_small_patch16_224 --teacher-path <TEACHER_CHECKPOINT_PATH> --ffn_linear_method block_diag --ffn_fc1_shift_step 0 --ffn_fc2_shift_step 0 --ffn_group_num 1 --attn_linear_method block_diag --attn_qkv_shift_step 0 --attn_proj_shift_step 0 --attn_group_num 1 --head_linear_method normal --head_shift_step 0 --head_group_num 1 --distillation-type hard --learnable_groups --criterion_type block_diag --complexity-weight 1 --complexity-bias 0 --gradient-weight 100
```

- Stage 2: Fine-Tuning With Shift

```bash
CUDA_VISIBLE_DEVICES=<GPU_DEVICE_ID> python -m torch.distributed.launch --master_port=<DISTRIBUTED_COMM_PORT> --nproc_per_node=<NUM_GPU_PER_NODE> --use_env main.py --model fourbits_deit_small_patch16_224 --epochs 300 --warmup-epochs 0 --weight-decay 0.05 --batch-size 64 --data-path <CIFAR100_DATASET_ROOT> --data-set CIFAR --lr 0.001 --output_dir <EXPERIMENT_OUTPUT_DIR> --distillation-type hard --finetune <PRETRAINED_CHECKPOINT_PATH> --teacher-model deit_small_patch16_224 --teacher-path <TEACHER_CHECKPOINT_PATH> --ffn_linear_method block_diag --ffn_fc1_shift_step 1 --ffn_fc2_shift_step 1 --ffn_group_num 1 --attn_linear_method block_diag --attn_qkv_shift_step 1 --attn_proj_shift_step 1 --attn_group_num 1 --head_linear_method normal --head_shift_step 0 --head_group_num 1 --min-lr 5e-7 --criterion_type normal --layer_groups "{\"attn_qkv\":[2,8,2,2,2,2,8,8,2,8,8,12],\"attn_proj\":[2,8,2,2,2,2,12,12,3,12,12,12],\"mlp_fc1\":[1,12,12,12,12,12,12,12,12,12,8,1],\"mlp_fc2\":[1,8,8,12,12,12,12,12,12,12,8,2]}" --complexity-weight 1 --complexity-bias 0 --gradient-weight 100
```

## Evaluation

Use the following command to evaluate the fine-tuned model on CIFAR-100. The layer-wise group configuration and shift settings must be consistent with those used in stage 2.

```bash
CUDA_VISIBLE_DEVICES=<GPU_DEVICE_ID> python -m torch.distributed.launch --master_port=<FREE_PORT> --nproc_per_node=1 --use_env main.py --model fourbits_deit_small_patch16_224 --batch-size 128 --data-path <DATASET_ROOT_PATH> --data-set CIFAR --output_dir <EVALUATION_OUTPUT_PATH> --resume <FINE_TUNED_CHECKPOINT_PATH> --eval --distillation-type none --ffn_linear_method block_diag --ffn_fc1_shift_step 1 --ffn_fc2_shift_step 1 --ffn_group_num 1 --attn_linear_method block_diag --attn_qkv_shift_step 1 --attn_proj_shift_step 1 --attn_group_num 1 --head_linear_method normal --head_shift_step 0 --head_group_num 1 --criterion_type normal --layer_groups '{"attn_qkv":[2,8,2,2,2,2,8,8,2,8,8,12],"attn_proj":[2,8,2,2,2,2,12,12,3,12,12,12],"mlp_fc1":[1,12,12,12,12,12,12,12,12,12,8,1],"mlp_fc2":[1,8,8,12,12,12,12,12,12,12,8,2]}' 2>&1 | tee <EVALUATION_OUTPUT_PATH>/test.log
```

可以改成下面这样，结构更清楚，也说明了 CUTLASS 版本、各个程序的用途以及运行目录。

## CUDA/CUTLASS Kernel Benchmark

The fused low-precision CUDA kernels and benchmark programs are located in the `gpu_kernel/` directory.

### CUTLASS Version

The kernels are developed and evaluated using CUTLASS v4.3.0:

```text
Commit: e67e63c331d6e4b729047c95cf6b92c8454cba89
Tag: v4.3.0
Branch: release/4.3
```

### Build

Enter the kernel directory:

```bash
cd gpu_kernel
```

Set the target bit width in the source file:

```cpp
#define BIT_WIDTH 1
```

Change `BIT_WIDTH` to `1`, `4`, or `8`, and rebuild for each setting:

```bash
make clean
make bench_gemm.bin
```

Build the INT4 benchmarks:

```bash
make int4_mma_compute_bound.bin
make int4_mma_compute_bound_128x128x128.bin
```

### Run

Run the benchmark script and executables from the `gpu_kernel/` directory:

```bash
python run-gemm.py
./bench_gemm.bin
./int4_mma_compute_bound.bin
./int4_mma_compute_bound_128x128x128.bin
```


## Datasets and Teacher Models

- **CIFAR-100**: [Dataset](http://www.cs.toronto.edu/~kriz/cifar.html) | [Teacher Model](https://drive.google.com/file/d/1h_NWxG0-TcUU6-CbUvzclljfzxBlzCn4/view?usp=share_link)
- **Oxford Flowers-102**: [Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | [Teacher Model](https://drive.google.com/file/d/13N_uqWNCDwj5g6AgBXqqxzVBs1pBPnGp/view?usp=share_link)
- **Chaoyang**: [Dataset](https://bupt-ai-cz.github.io/HSA-NRL/) | [Teacher Model](https://drive.google.com/file/d/1jOBObIn3fzRQvRrT_RSbWVJNKeN_iAro/view?usp=share_link)


## Model Zoo

| Dataset | Method | W-A | OPs ($\times10^8$) | Params (MB) | Top-1 (%) | Link |
|---|---|---:|---:|---:|---:|---|
| CIFAR-100 | LiteBDS-ViT | 4-4 | 1.41 | 3.0 | 73.39 | [Download](https://drive.google.com/file/d/1ZH5UsSpM0_yli59qCjDYoqfIlP79bBgY/view?usp=sharing) |
| Oxford Flowers-102 | LiteBDS-ViT | 4-4 | 1.41 | 3.0 | 68.42 | - |
| Chaoyang | LiteBDS-ViT | 4-4 | 1.41 | 3.1 | 85.43 | - |

## Acknowledgement

Our code refers to Q-ViT(https://github.com/YanjingLi0202/Q-ViT) and GSB-ViT(https://github.com/IMRL/GSB-Vision-Transformer).