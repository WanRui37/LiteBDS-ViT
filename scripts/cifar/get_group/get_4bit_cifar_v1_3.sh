source set_mirror.sh

version="v1.6"
output_dir="result/chaoyang_w4a4_train_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=2 python ./utils/3-get_layer_group.py  \
2>&1 | tee "$output_dir/get_group.log"
