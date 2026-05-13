PROFILE=0

INC = 	-I../cutlass/include\
		-I../cutlass/tools/util/include \
		-I../cutlass/examples/common
		
FLAG= -std=c++17 -O3 -w -arch=compute_80 -code=sm_80 $(if $(PROFILE),--generate-line-info)

int4_mma_compute_bound.bin: int4_mma_compute_bound.cu
	nvcc $(INC) $(FLAG) $< -o $@

clean:
	rm -rf *.bin 
