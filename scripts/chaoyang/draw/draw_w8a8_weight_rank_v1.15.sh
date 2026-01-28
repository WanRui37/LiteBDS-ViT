source set_mirror.sh

# VERSION="v1_5"
VERSION="v1.15"

output_dir="utils/pic_weights_rank/chaoyang_w8a8_$VERSION"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=3 python ./utils/2-visualize_weights_rank.py \
    --checkpoint ./result/chaoyang_w8a8_train_$VERSION/best_checkpoint.pth \
    --output-dir "$output_dir" \
    --model-version "chaoyang_w8a8_$VERSION" \
    --num-groups-weight 4 \
2>&1 | tee "$output_dir/result.log"

# --finetune
# --resume
# --checkpoint ./result_stage1/cifar_w8a32_train/best_checkpoint.pth \
