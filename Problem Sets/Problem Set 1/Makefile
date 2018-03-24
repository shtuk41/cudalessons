
CC=nvcc
CFLAGS=-g
OPENCV_INCLUDE=/usr/local/include
CUDA_INCLUDE=/usr/local/cuda-9.1/include
CUDA_LIB=/usr/local/cuda-9.1/lib64
OPENCV_LIB=/usr/local/lib


all: 	main



main:	compare reference_calc
	$(CC) $(CFLAGS) -I$(OPENCV_INCLUDE) -I$(CUDA_INCLUDE) -L$(CUDA_LIB) \
	 -lcudart -lopencv_core -lopencv_highgui -lopencv_imgcodecs \
	 -lopencv_imgproc -o HW1 main.cpp compare.o reference_calc.o student_func.cu

compare:
	$(CC) $(CFLAGS) -I$(OPENCV_INCLUDE) -c -o compare.o compare.cpp

reference_calc:
	$(CC) $(CFLAGS) -I$(OPENCV_INCLUDE) -c -o reference_calc.o reference_calc.cpp

clean:
	rm -f main HW1 *.o  *.out
