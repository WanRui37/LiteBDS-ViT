source set_mirror.sh

version="v1.9"
output_dir="result/chaoyang_w8a8_test_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
    --master_port=12358 \
    --nproc_per_node=1 \
    --use_env main.py  \
    --model eightbits_deit_small_patch16_224 \
    --epochs 600 \
    --weight-decay 5e-4 \
    --batch-size 32 \
    --data-path /mnt/data/small_dataset/chaoyang/ \
    --data-set chaoyang \
    --output_dir "$output_dir" \
    --resume result/chaoyang_w8a8_train_$version/best_checkpoint.pth \
    --ffn_linear_method shift \
    --ffn_fc1_shift_step 1 \
    --ffn_fc2_shift_step 1 \
    --ffn_group_num 4 \
    --attn_linear_method shift \
    --attn_qkv_shift_step 1 \
    --attn_proj_shift_step 1 \
    --attn_group_num 4 \
    --head_linear_method normal \
    --head_shift_step 0 \
    --head_group_num 1 \
    --min-lr 1e-6 \
    --eval \
2>&1 | tee "$output_dir/test.log"

# --teacher-model vit_deit_small_distilled_patch16_224 \
# --opt fusedlamb \