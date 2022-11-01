#include <cstdio>
#include <cstring>
#include <time.h>
#include <hip/hip_runtime.h>

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* GPU kernel definition */
__global__ void hipKernel(int* const A, const int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    A[idx] = 1;
}

/* Auxiliary function to check the results */
void checkTiming(const std::string strategy, const double timing)
{
  printf("%.3fs - %s)\n", timing, strategy.c_str());
}

/* Run without memory pooling */
void noMemPools(int nSteps, int size)
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
  checkResults("noMemPools", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Run using memory pooling but no recurring syncs */
void memPoolNoSync(int nSteps, int size)
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
    hipMallocAsync((void**)&d_A, size, 0);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    hipFreeAsync(d_A, 0);
  }
  // Synchronization
  hipStreamSynchronize(0);
  // Check results and print timings
  checkResults("memPoolNoSync", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Run using memory pooling and recurring syncs */
void memPoolSync(int nSteps, int size)
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
    hipMallocAsync((void**)&d_A, size, 0);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    hipFreeAsync(d_A, 0);
    // Synchronization
    hipStreamSynchronize(0);
  }
  // Check results and print timings
  checkResults("memPoolSync", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Run using memory pooling and recurring syncs with changed release threshold */
void memPoolSyncWithThreshold(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  // Change the memory pool deallocations threshold
  int device;
  hipGetDevice(&device);
  hipMemPool_t mempool;
  hipDeviceGetDefaultMemPool(&mempool, device);
  uint64_t threshold = UINT64_MAX;
  hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &threshold);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    hipMallocAsync((void**)&d_A, size, 0);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    hipFreeAsync(d_A, 0);
    // Synchronization
    hipStreamSynchronize(0);
  }
  // Check results and print timings
  checkResults("memPoolSyncWithThreshold", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}



/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 1D grid dimensions
  int nSteps = 100, size = 1e6;

  // Run with different memory allocatins strategies
  noBemPools(nSteps, size);
  memPoolNoSync(nSteps, size);
  memPoolSync(nSteps, size);
  memPoolSyncWithThreshold(nSteps, size);
