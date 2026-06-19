source set_mirror.sh

version="stage2"
train_dir="result/cifar_w4a4_train_${version}"
output_dir="result/cifar_w4a4_test_${version}"

if [ -f "${train_dir}/best_checkpoint.pth" ]; then
    resume_ckpt="${train_dir}/best_checkpoint.pth"
elif [ -f "${train_dir}/checkpoint.pth" ]; then
    resume_ckpt="${train_dir}/checkpoint.pth"
else
    echo "Error: no checkpoint found in ${train_dir}"
    exit 1
fi

mkdir -p "$output_dir"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port=12351 \
    --nproc_per_node=1 \
    --use_env main.py \
    --model fourbits_deit_small_patch16_224 \
    --batch-size 128 \
    --data-path /root/autodl-tmp/data/cifar-100/ \
    --data-set CIFAR \
    --output_dir "$output_dir" \
    --resume "$resume_ckpt" \
    --eval \
    --distillation-type none \
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
    --criterion_type normal \
    --layer_groups "{\"attn_qkv\":[2,8,2,2,2,2,8,8,2,8,8,12],\"attn_proj\":[2,8,2,2,2,2,12,12,3,12,12,12],\"mlp_fc1\":[1,12,12,12,12,12,12,12,12,12,8,1],\"mlp_fc2\":[1,8,8,12,12,12,12,12,12,12,8,2]}" \
    2>&1 | tee "$output_dir/test.log"