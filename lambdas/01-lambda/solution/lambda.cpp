#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

/* Blocksize is small because we are printing from all threads */
#define BLOCKSIZE 4

/* CPU loop execution */
template <typename Lambda>
void cpuKernel(Lambda lambda, const int loop_size) {
  for(int i = 0; i < loop_size; i++){
    lambda(i);
  }
}

/* GPU loop execution */
template <typename Lambda> 
__global__ void gpuKernel(Lambda lambda, const int loop_size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < loop_size)
  {
    lambda(i);
  }
}

/* Check if this function is running on CPU or GPU */
__host__ __device__ void helloFromThread(const int i) {
  #ifdef __HIP_DEVICE_COMPILE__ // If running on GPU
    printf("Hello from GPU! I'm thread number %d\n", i);
  #else // If running on CPU
    printf("Hello from CPU! I'm thread number %d\n", i);
  #endif
}


/* The main function */
int main()
{
  // Set the problem dimensions
  const int loop_size = BLOCKSIZE;
  const int blocksize = BLOCKSIZE;
  const int gridsize = (loop_size - 1 + blocksize) / blocksize;

  // Define lambda1 function with 1 integer argument,
  // the lamba must call helloFromThread with that argument
  auto lambda1 = [] __host__ __device__ (const int i)
  {
    helloFromThread(i);
  };

  // Run lambda1 on the CPU device
  cpuKernel(lambda1, loop_size);

  // Run lambda1 on the GPU device
  gpuKernel<<<gridsize, blocksize>>>(lambda1, loop_size);
  hipStreamSynchronize(0);

  // Store value of pi in pi
  double pi = M_PI;

  // Define lambda2 that captures pi (use [=] to capture by value), 
  // and prints out the results for i * pi from each thread
  auto lambda2 = [=] __host__ __device__ (const int i)
  {
    printf("i * pi = %f \n", (double)i * pi);
  };

  // Run lambda2 on the GPU device
  gpuKernel<<<gridsize, blocksize>>>(lambda2, loop_size);
  hipStreamSynchronize(0);
}
