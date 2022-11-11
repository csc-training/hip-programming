#include <cstdio>
#include <string>
#include <time.h>
#include <hip/hip_runtime.h>

#if defined(HAVE_UMPIRE)
  #include "umpire/interface/c_fortran/umpire.h"
#endif

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

// HIP error checking
#define HIP_ERR(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

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
  HIP_ERR(hipMalloc((void**)&d_A, sizeof(int) * size));

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Synchronization
    HIP_ERR(hipStreamSynchronize(0));
  }
  // Free allocation
  HIP_ERR(hipFree(d_A));
}

/* Run without recurring allocation */
void noRecurringAlloc(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  int *d_A;
  // Allocate pinned device memory
  HIP_ERR(hipMalloc((void**)&d_A, sizeof(int) * size));

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(0));
  // Check results and print timings
  checkTiming("noRecurringAlloc", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Free allocation
  HIP_ERR(hipFree(d_A));
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
    HIP_ERR(hipMalloc((void**)&d_A, sizeof(int) * size));
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    HIP_ERR(hipFree(d_A));
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(0));
  // Check results and print timings
  checkTiming("recurringAllocNoMemPools", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Do recurring allocation with memory pooling */
void recurringAllocMallocAsync(int nSteps, int size)
{
  // Create HIP stream
  hipStream_t stream;
  HIP_ERR(hipStreamCreate(&stream));

  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    cudaMallocAsync((void**)&d_A, sizeof(int) * size, stream);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, stream>>>(d_A, size);
    // Free allocation
    cudaFreeAsync(d_A, stream);
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(stream));
  // Check results and print timings
  checkTiming("recurringAllocMallocAsync", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Destroy the stream
  HIP_ERR(hipStreamDestroy(stream));
}

#if defined(HAVE_UMPIRE)
/* Do recurring allocation with Umpire memory pool */
void recurringAllocUmpire(int nSteps, int size)
{
  // Get Umpire pinned device memory pool
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);
  umpire_allocator allocator;
  umpire_resourcemanager_get_allocator_by_name(&rm, "DEVICE", &allocator);
  umpire_allocator pool;
  umpire_resourcemanager_make_allocator_quick_pool(&rm, "pool", allocator, 1024, 1024, &pool);

  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory with Umpire
    d_A = (int*) umpire_allocator_allocate(&pool, sizeof(int) * size);
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free Umpire allocation
    umpire_allocator_deallocate(&pool, d_A);
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(0));
  // Check results and print timings
  checkTiming("recurringAllocUmpire", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}
#endif

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 1D grid dimensions
  int nSteps = 1e4, size = 1e6;
  
  // Ignore first run, first kernel is slower
  ignoreTiming(nSteps, size);

  // Run with different memory allocatins strategies
  noRecurringAlloc(nSteps, size);
  recurringAllocNoMemPools(nSteps, size);
  recurringAllocMallocAsync(nSteps, size);
  #if defined(HAVE_UMPIRE)
    recurringAllocUmpire(nSteps, size);
  #endif
}
