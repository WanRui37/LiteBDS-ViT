source set_mirror.sh

version="v1.3"
output_dir="result/cifar_w4a4_train_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
    --master_port=12350 \
    --nproc_per_node=2 \
    --use_env main.py  \
    --model fourbits_deit_small_patch16_224 \
    --epochs 900 \
    --warmup-epochs 0 \
    --weight-decay 0.05 \
    --batch-size 64 \
    --data-path /mnt/data/cifar-100/ \
    --data-set CIFAR \
    --lr 0.001 \
    --output_dir "$output_dir" \
    --distillation-type hard \
    --teacher-model deit_small_patch16_224 \
    --teacher-path /mnt/wr/3-LLM/9-ViT/GSB-Vision-Transformer/teacher/cifar/teacher.pth \
    --ffn_linear_method block_diag \
    --ffn_fc1_shift_step 0 \
    --ffn_fc2_shift_step 0 \
    --ffn_group_num 1 \
    --attn_linear_method block_diag \
    --attn_qkv_shift_step 0 \
    --attn_proj_shift_step 0 \
    --attn_group_num 1 \
    --head_linear_method normal \
    --head_shift_step 0 \
    --head_group_num 1 \
    --min-lr 5e-7 \
    --distillation-type hard \
    --learnable_groups \
    --criterion_type block_diag \
    --complexity-weight 1 \
    --complexity-bias 0 \
    --gradient-weight 100 \
    2>&1 | tee "$output_dir/train.log"

# --teacher-model vit_deit_small_distilled_patch16_224 \
# --opt fusedlamb \
# --repeated-aug \
