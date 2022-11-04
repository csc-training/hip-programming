#include <cstdio>
#include <hip/hip_runtime.h>

/* GPU kernel definition */
__global__ void hipKernel(int *d_value)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(d_value[idx] == 3*1e8)
    printf("Result is correct! (d_value = %d)\n", d_value[idx]);
  else
    printf("Result is incorrect! (d_value = %d)\n", d_value[idx]);
}

__global__ void hipKernel2(int *d_value, int multiplier)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 0; i <= 1e8; ++i)
    d_value[idx] = i * multiplier;
}

/* The main function */
int main(int argc, char* argv[])
{
  hipStream_t strm_copy;
  hipStreamCreate(&strm_copy);
  
  hipStream_t strm_kernel;
  hipStreamCreate(&strm_kernel);

  int *d_value;
  cudaMallocAsync((void**)&d_value, sizeof(int), strm_copy);
  hipKernel2<<<1, 1, 0, strm_copy>>>(d_value, 1);
  hipKernel2<<<1, 1, 0, strm_copy>>>(d_value, 2);
  hipKernel2<<<1, 1, 0, strm_copy>>>(d_value, 3);

  hipKernel<<<1, 1, 0, strm_kernel>>>(d_value);

  hipDeviceSynchronize();

  hipStreamDestroy(strm_copy);
  hipStreamDestroy(strm_kernel);
}
