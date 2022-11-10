#include <cstdio>
#include <string>
#include <time.h>
#include <hip/hip_runtime.h>

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* GPU kernel definition */
__global__ void hipKernel(int* const A, const int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    A[idx] = idx;
}

/* Auxiliary function to check the results */
void checkTiming(const std::string strategy, const double timing)
{
  printf("%.3f ms - %s\n", timing * 1e3, strategy.c_str());
}

/* Run without timing */
void ignoreTiming(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  int *d_A;
  // Allocate pinned device memory
  hipMalloc((void**)&d_A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Synchronization
    hipStreamSynchronize(0);
  }
  // Free allocation
  hipFree(d_A);
}

/* Run without recurring allocation */
void noRecurringAlloc(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  int *d_A;
  // Allocate pinned device memory
  hipMalloc((void**)&d_A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
  }
  // Synchronization
  hipStreamSynchronize(0);
  // Check results and print timings
  checkTiming("noRecurringAlloc", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Free allocation
  hipFree(d_A);
}

/* Do recurring allocation without memory pooling */
void recurringAllocNoMemPools(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    hipMalloc((void**)&d_A, size);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    hipFree(d_A);
  }
  // Synchronization
  hipStreamSynchronize(0);
  // Check results and print timings
  checkTiming("recurringAllocNoMemPools", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Do recurring allocation with memory pooling */
void recurringAllocMemPool(int nSteps, int size)
{
  // Create HIP stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    cudaMallocAsync((void**)&d_A, size, stream);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, stream>>>(d_A, size);
    // Free allocation
    cudaFreeAsync(d_A, stream);
  }
  // Synchronization
  hipStreamSynchronize(stream);
  // Check results and print timings
  checkTiming("recurringAllocMemPoolNoSync", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Destroy the stream
  hipStreamDestroy(stream);
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 1D grid dimensions
  int nSteps = 1e6, size = 1e6;
  
  // Ignore first run, first kernel is slower
  ignoreTiming(nSteps, size);

  // Run with different memory allocatins strategies
  noRecurringAlloc(nSteps, size);
  recurringAllocNoMemPools(nSteps, size);
  recurringAllocMemPool(nSteps, size);
}
