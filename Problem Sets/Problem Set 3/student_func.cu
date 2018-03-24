/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include "utils.h"

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

__global__ void FindMax(float *input, float *output, int size)
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
      input[idx] = max(input[idx], input[idx + stride]);

      if (prev_stride_odd && tid == 0)
      {
        input[idx] = max(input[idx], input[idx + stride * 2]);
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

__global__ void Histogram(float *input, unsigned int* output, int size, float lumMin, float lumRange, int numBins)
{
  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx >= size)
    {
      return;
    }


    int bin = (input[idx] - lumMin) / lumRange * numBins;

    if (bin >= numBins)
    {
      bin-=1;
    }

    atomicAdd(&output[bin], 1);


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


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum*/

       printf("Starting 1...\n");

       size_t numberOfPixels = numRows * numCols;
       size_t imageSize = (numRows * numCols) * sizeof(float);
       size_t outputSize = numRows * sizeof(float);

       float *d_input;
       float *d_output;
       cudaMalloc((void **)&d_input, imageSize);
       cudaMalloc((void **)&d_output, outputSize);

       cudaMemcpy(d_input,d_logLuminance,imageSize,cudaMemcpyDeviceToDevice);


       dim3 block = (numCols);
       dim3 grid = (numRows);

       FindMin<<< grid, block >>>(d_input, d_output, numberOfPixels);
       cudaDeviceSynchronize();

       FindMin<<< 1, numRows >>>(d_output, d_output, numRows);
       cudaDeviceSynchronize();


       cudaMemcpy(&min_logLum,d_output,sizeof(float),cudaMemcpyDeviceToHost);

       printf("Minimum is: %f\n", min_logLum);

       cudaMemcpy(d_input,d_logLuminance,imageSize,cudaMemcpyDeviceToDevice);

       cudaMemset((void*)&d_output, 0, outputSize);

       FindMax<<< grid, block >>>(d_input, d_output, numberOfPixels);
       cudaDeviceSynchronize();

       FindMax<<< 1, numRows >>>(d_output, d_output, numRows);
       cudaDeviceSynchronize();


       cudaMemcpy(&max_logLum,d_output,sizeof(float),cudaMemcpyDeviceToHost);

       printf("Maximum is: %f\n", max_logLum);

       
       cudaFree(d_output);

  /*  2) subtract them to find the range */

      float lumRange = max_logLum - min_logLum;

      printf("lumRange: %f\n", lumRange);

  /*  3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */

       unsigned int *d_histogram;
       cudaMalloc((void **)&d_histogram, numBins * sizeof(unsigned int));
       cudaMemset((void*)d_histogram, 0, numBins * sizeof(unsigned int));
       cudaMemcpy(d_input,d_logLuminance,imageSize,cudaMemcpyDeviceToDevice);
       Histogram<<< grid, block >>>(d_input, d_histogram, numberOfPixels, min_logLum, lumRange, numBins);


       cudaFree(d_input);
  /*  4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

       Blelock<<< grid , block >>>(d_histogram,numBins);

       cudaMemcpy(d_cdf,d_histogram,numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

       cudaFree(d_histogram);
}
