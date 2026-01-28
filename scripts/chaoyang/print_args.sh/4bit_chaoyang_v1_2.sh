source set_mirror.sh

version="v1.2"
output_dir="result/chaoyang_w4a4_train_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=1 python utils/1-print_checkpoint_args.py \
    --checkpoint_path "$output_dir/best_checkpoint.pth" \
    --analyze_groups \
2>&1 | tee "$output_dir/print.log"
