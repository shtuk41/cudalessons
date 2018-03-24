#include <cuda_runtime.h>
#include <stdio.h>
#include <malloc.h>

__global__ void reduceMin(float *input, float *output, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (idx >= size)
	{
		return;
	}

	bool prev_stride_odd = (blockDim.x % 2 != 0);

	for (int stride = blockDim.x /2;	stride > 0;		stride >>=1)
	{
		if (tid < stride)
		{
			float temp = input[tid];
			input[tid] = min(input[tid], input[tid + stride]);

			if (prev_stride_odd && tid == 0)
			{
				printf("odd stride: %f\n", input[stride * 2]);
				input[tid] = min(input[tid], input[stride * 2]);
			}


			printf("tid %d, min of %f abd %f is: %f\n", tid, temp, input[tid + stride], input[tid]); 
		}

		prev_stride_odd = stride % 2 != 0;

		__syncthreads();
	}

	if (tid == 0)
	{
		output[0] = input[0];
	}
}

__global__ void reduceMax(float *input, float *output, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (idx >= size)
	{
		return;
	}

	bool prev_stride_odd = (blockDim.x % 2 != 0);

	for (int stride = blockDim.x /2;	stride > 0;		stride >>=1)
	{
		if (tid < stride)
		{
			float temp = input[tid];
			input[tid] = max(input[tid], input[tid + stride]);

			if (prev_stride_odd && tid == 0)
			{
				printf("odd stride: %f\n", input[stride * 2]);
				input[tid] = max(input[tid], input[stride * 2]);
			}


			printf("tid %d, max of %f abd %f is: %f\n", tid, temp, input[tid + stride], input[tid]); 
		}

		prev_stride_odd = stride % 2 != 0;

		__syncthreads();
	}

	if (tid == 0)
	{
		output[0] = input[0];
	}
}

__global__ void FindMin(float *input, float *output, int size)
{
  int start = blockIdx.x * blockDim.x;
  int idx   = start + threadIdx.x;
  int tid   = threadIdx.x;

  if (idx >= size)
  {
    return;
  }

  bool prev_stride_odd = (blockDim.x % 2 != 0);

  for (int stride = blockDim.x /2;  stride > 0;   stride >>=1)
  {
    if (tid < stride)
    {
      input[idx] = min(input[idx], input[idx + stride]);

      if (prev_stride_odd && tid == 0)
      {
        input[idx] = min(input[idx], input[idx + stride * 2]);
      }
    }

    prev_stride_odd = stride % 2 != 0;

    __syncthreads();
  }

  if (tid == 0)
  {
    output[blockIdx.x] = input[idx];
  }
}

__global__ void Blelock(unsigned int *input, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= size)
	{
		return;
	}

	for (int ii = 2; ii <= size; ii*=2)
	{
		if ((idx  + 1) % ii == 0)
		{
			input[idx] = input[idx] + input[idx - ii / 2];
		}

		__syncthreads();
	}

	if (idx == size - 1)
		input[idx] = 0;

	__syncthreads();	

	for (int ii = size; ii >= 2; ii>>=1)
	{
		if ((idx  + 1) % ii == 0)
		{
			int temp = input[idx];
			input[idx] = input[idx] + input[idx - ii / 2];
			input[idx - ii / 2] = temp;

		}

		__syncthreads();

	}
}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaSetDevice(dev);

	int numberBins = 16;
	int nBytesInput = numberBins * sizeof(unsigned int);


	unsigned int * h_input 	= (unsigned int *)malloc(nBytesInput);


	h_input[0] = 1;
	h_input[1] = 2;
	h_input[2] = 3;
	h_input[3] = 4;
	h_input[4] = 5;
	h_input[5] = 6;
	h_input[6] = 7;
	h_input[7] = 8;
	h_input[8] = 9;
	h_input[9] = 10;
	h_input[10] = 11;
	h_input[11] = 12;
	h_input[12] = 13;
	h_input[13] = 14;
	h_input[14] = 15;
	h_input[15] = 16;


	unsigned int *d_input;
	cudaMalloc((void **)&d_input, nBytesInput);

	cudaMemcpy(d_input, h_input, nBytesInput, cudaMemcpyHostToDevice);

	Blelock<<<1, numberBins>>>(d_input,numberBins);
	cudaDeviceSynchronize();

	cudaMemcpy(h_input, d_input, nBytesInput, cudaMemcpyDeviceToHost);

	for (int ii = 0; ii < numberBins; ii++)
	{
		printf("Bin %d, value: %u\n", ii, h_input[ii]);
	}

	cudaFree(d_input);
	free(h_input);


	return 0;
}