PROFILE=0

path = /mnt/wr/3-LLM/9-ViT/LiteBDS-ViT-old/cutlass
INC = 	-I$(path)/include\
		-I$(path)/tools/util/include \
		-I$(path)/examples/common
		
FLAG= -std=c++17 -O3 -w -arch=compute_80 -code=sm_80 $(if $(PROFILE),--generate-line-info)

bench_gemm.bin: bench_gemm.cu
	nvcc $(INC) $(FLAG) $< -o $@

int4_mma_compute_bound.bin: int4_mma_compute_bound.cu
	nvcc $(INC) $(FLAG) $< -o $@

int4_mma_compute_bound_128x128x128.bin: int4_mma_compute_bound_128x128x128.cu
	nvcc $(INC) $(FLAG) $< -o $@

clean:
	rm -rf *.bin 
