source set_mirror.sh

version="v1.1"
output_dir="result/flower_w4a4_train_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
    --master_port=12356 \
    --nproc_per_node=1 \
    --use_env main.py  \
    --model fourbits_deit_small_patch16_224 \
    --epochs 1200 \
    --warmup-epochs 0 \
    --weight-decay 5e-2 \
    --batch-size 32 \
    --data-path /mnt/data/small_dataset/flower_kaggle/dataset/ \
    --data-set flower \
    --lr 4e-3 \
    --output_dir "$output_dir" \
    --distillation-type hard \
    --teacher-model deit_small_patch16_224 \
    --teacher-path /mnt/wr/3-LLM/9-ViT/GSB-Vision-Transformer/teacher/flower/teacher.pth \
    --ffn_linear_method normal \
    --ffn_fc1_shift_step 0 \
    --ffn_fc2_shift_step 0 \
    --ffn_group_num 1 \
    --attn_linear_method normal \
    --attn_qkv_shift_step 0 \
    --attn_proj_shift_step 0 \
    --attn_group_num 1 \
    --head_linear_method normal \
    --head_shift_step 0 \
    --head_group_num 1 \
    --min-lr 1e-7 \
    --distillation-type hard \
    --dist-eval \
2>&1 | tee "$output_dir/train.log"

# --teacher-model vit_deit_small_distilled_patch16_224 \
# --opt fusedlamb \
# --repeated-aug \
