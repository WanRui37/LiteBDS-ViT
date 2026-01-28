source set_mirror.sh

version="v1.6"
output_dir="result/chaoyang_w4a4_train_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
    --master_port=12454 \
    --nproc_per_node=1 \
    --use_env main.py  \
    --model fourbits_deit_small_patch16_224 \
    --epochs 1200 \
    --warmup-epochs 0 \
    --weight-decay 1e-3 \
    --batch-size 32 \
    --data-path /mnt/data/small_dataset/chaoyang/ \
    --data-set chaoyang \
    --lr 4e-3 \
    --output_dir "$output_dir" \
    --distillation-type hard \
    --teacher-model deit_small_patch16_224 \
    --teacher-path /mnt/wr/3-LLM/9-ViT/GSB-Vision-Transformer/teacher/chaoyang/teacher.pth \
    --ffn_linear_method block_diag \
    --ffn_fc1_shift_step 1 \
    --ffn_fc2_shift_step 1 \
    --ffn_group_num 1 \
    --attn_linear_method block_diag \
    --attn_qkv_shift_step 1 \
    --attn_proj_shift_step 1 \
    --attn_group_num 1 \
    --head_linear_method normal \
    --head_shift_step 0 \
    --head_group_num 1 \
    --min-lr 1e-7 \
    --distillation-type hard \
    --dist-eval \
    --layer_groups "{\"attn_qkv\":[1,4,2,3,4,2,4,4,3,3,4,3],\"attn_proj\":[1,8,4,4,8,4,4,8,4,8,8,8],\"mlp_fc1\":[1,8,8,8,8,8,8,8,8,8,8,8],\"mlp_fc2\":[1,8,8,8,8,8,8,8,8,8,8,8]}" \
2>&1 | tee "$output_dir/train.log"

# --teacher-model vit_deit_small_distilled_patch16_224 \
# --opt fusedlamb \
# --repeated-aug \
# --criterion_type shift \
# --rank-reg-weight 0.00001 \
