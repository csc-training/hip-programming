/*
 * This code compares different GPU memory allocation strategies.
 *
 * Task is to:
 *   - allocate device memory using hipMalloc()
 *   - perform recurring allocations using hipMalloc()/hipFree()
 *   - perform recurring allocations using hipMallocAsync()/hipFreeAsync()
 *   - create and synchronize a HIP stream for async allocations
 *   - compare the timing between the approaches
 *
 * Observe how stream-ordered memory allocation can reduce
 * recurring allocation overhead through memory pooling.
 */
#include <cstdio>
#include <cstring>
#include <time.h>
#include <hip/hip_runtime.h>

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

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

/* Run without timing as a warmup */
void warmupRun(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);
  
  int *d_A;
  // Allocate device memory
  HIP_ERRCHK(hipMalloc((void**)&d_A, bytes));

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Synchronization
    HIP_ERRCHK(hipStreamSynchronize(0));
  }
  // Free allocation
  HIP_ERRCHK(hipFree(d_A));
}

/* Run using a single device allocation outside of the loop */
void noRecurringAlloc(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);

  int *d_A;
  // Allocate device memory
  #error allocate memory with hipMalloc for d_A of size `bytes`

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
  }
  // Synchronization
  #error synchronize the default stream here before stopping the timer
  // Check results and print timings
  checkTiming("noRecurringAlloc", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Free allocation
  #error free d_A allocation using hipFree
}

/* Do recurring allocation without memory pooling */
void recurringAllocNoMemPools(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate device memory
    #error allocate memory with hipMalloc for d_A of size `bytes`
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    HIP_ERRCHK(hipGetLastError());
    // Free allocation
    #error free d_A allocation using hipFree
  }
  // Synchronization
  // Ensure all queued allocations and kernels complete before stopping timing
  #error synchronize the default stream here before stopping the timer
  // Check results and print timings
  checkTiming("recurringAllocNoMemPools", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Do recurring allocation with memory pooling */
void recurringAllocMallocAsync(int nSteps, int size)
{
  // Create HIP stream
  hipStream_t stream;
  HIP_ERRCHK(hipStreamCreate(&stream));

  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate device memory
    #error allocate memory with hipMallocAsync for d_A of size `bytes` in stream
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, stream>>>(d_A, size);
    HIP_ERRCHK(hipGetLastError());
    // Free allocation
    #error free d_A allocation using hipFreeAsync in stream
  }
  // Synchronization
  #error synchronize stream here
  // Check results and print timings
  checkTiming("recurringAllocMallocAsync", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  // Destroy the stream
  HIP_ERRCHK(hipStreamDestroy(stream));
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 1D grid dimensions
  int nSteps = 1e4, size = 1e6;
  
  // Ignore first run, first kernel is slower (warmup)
  warmupRun(nSteps, size);

  // Run with different memory allocation strategies
  noRecurringAlloc(nSteps, size);
  recurringAllocNoMemPools(nSteps, size);
  recurringAllocMallocAsync(nSteps, size);
}
