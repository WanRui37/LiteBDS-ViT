# 运行

本目录仅保留 BDS 相关的内核与输出，用于快速验证不同尺寸下的 INT4 BDS 计算。

## 快速运行

```bash
# 大于等于256尺寸
make int4_mma_compute_bound.bin
./int4_mma_compute_bound.bin

# 小于256尺寸 (128x128x128)
make int4_mma_compute_bound_128x128x128.bin
./int4_mma_compute_bound_128x128x128.bin
```