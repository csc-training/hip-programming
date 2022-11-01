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
    // Synchronization
    hipStreamSynchronize(0);
  }
  // Check results and print timings
  checkTiming("noRecurringAlloc", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Free allocation
  hipFree(d_A);
}

/* Run without memory pooling */
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
    // Synchronization
    hipStreamSynchronize(0);
    // Free allocation
    hipFree(d_A);
  }
  // Check results and print timings
  checkTiming("recurringAllocNoMemPools", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Run using memory pooling but no recurring syncs */
void recurringAllocMemPoolNoSync(int nSteps, int size)
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
    cudaMallocAsync((void**)&d_A, size, 0);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    cudaFreeAsync(d_A, 0);
  }
  // Synchronization
  hipStreamSynchronize(0);
  // Check results and print timings
  checkTiming("recurringAllocMemPoolNoSync", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Run using memory pooling and recurring syncs */
void recurringAllocMemPoolSync(int nSteps, int size)
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
    cudaMallocAsync((void**)&d_A, size, 0);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    cudaFreeAsync(d_A, 0);
    // Synchronization
    hipStreamSynchronize(0);
  }
  // Check results and print timings
  checkTiming("recurringAllocMemPoolSync", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Run using memory pooling and recurring syncs with changed release threshold */
void recurringAllocMemPoolSyncWithThreshold(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  // Change the memory pool deallocations threshold
  int device;
  cudaGetDevice(&device);
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, device);
  uint64_t threshold = 0*UINT64_MAX;
  cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    cudaMallocAsync((void**)&d_A, size, 0);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    cudaFreeAsync(d_A, 0);
    // Synchronization
    hipStreamSynchronize(0);
  }
  // Check results and print timings
  checkTiming("recurringAllocMemPoolSyncWithThreshold", (double)(clock() - tStart) / CLOCKS_PER_SEC);
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
  recurringAllocMemPoolNoSync(nSteps, size);
  recurringAllocMemPoolSync(nSteps, size);
  recurringAllocMemPoolSyncWithThreshold(nSteps, size);
}
