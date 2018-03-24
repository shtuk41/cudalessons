NVCC=nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-9.1/include

NVCC_OPTS=-O3 -Xcompiler -Wall -m64

reduce:	reduce.cu
	$(NVCC) -o  reduce reduce.cu $(NVCC_OPTS)

clean:
	rm -f *.o reduce

