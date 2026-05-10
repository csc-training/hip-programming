/*
 * Task is to optimize two functions in the code based on the previous exercise:
 * - avoid recurring host-to-device memory transfers
 * - keep data resident on the GPU during the iterative loop
 * - initialize memory directly on the device using hipMemset()
 * - compare explicit memory management against unified memory
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
__global__ void hipKernel(int* const A, const int nx, const int ny)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nx * ny)
  {
    const int i = idx % nx;
    const int j = idx / nx;
    A[j * nx + i] += idx;
  }
}

/* Auxiliary function to check the results */
void checkResults(int* const A, const int nx, const int ny, const std::string strategy, const double timing)
{
  // Check that the results are correct
  int errored = 0;
  for(unsigned int i = 0; i < nx * ny; i++)
    if(A[i] != i)
      errored = 1;

  // Indicate if the results are correct
  if(errored)
    printf("The results are incorrect!\n");
  else
    printf("The results are OK! (%.3fs - %s)\n", timing, strategy.c_str());
}

/* Run using explicit memory management without recurring host/device memcopies */
void explicitMemNoCopy(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A, *d_A;
  size_t size = nx * ny * sizeof(int);

  #error Allocate pageable host memory of size `size` for the pointer A

  #error Allocate device memory (d_A)

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all operations
     * are performed using device (i.e., recurring memcopy is avoided):
     * Initializing the array directly on the GPU and running a GPU kernel.
     */

    #error Initialize array A to zeros on the device using hipMemset

    // Launch GPU kernel
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(d_A, nx, ny);
    HIP_ERRCHK(hipGetLastError());
  }

  #error Copy data back to host (d_A to A)

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "ExplicitMemNoCopy", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  #error Free device array (d_A)

  #error Free host array (A)
}

/* Run using Unified Memory without recurring host/device memcopies */
void unifiedMemNoCopy(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int device;
  #error Get device id number for prefetching

  int *A;
  size_t size = nx * ny * sizeof(int);

  #error Allocate Unified Memory of size `size` for the pointer A

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all operations
     * are performed using device (i.e., recurring memcopy is avoided):
     * Initializing the array directly on the GPU and running a GPU kernel.
     */

    #error Initialize array A to zeros on the device using hipMemset

    // Launch GPU kernel
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(A, nx, ny);
    HIP_ERRCHK(hipGetLastError());
  }
  #error Prefetch data (A) from device to host

  #error Synchronization

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "UnifiedMemNoCopy", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  #error Free Unified Memory array (A)
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 2D grid dimensions
  int nSteps = 100, nx = 8000, ny = 2000;

  // Compare timing
  explicitMemNoCopy(nSteps, nx, ny);
  unifiedMemNoCopy(nSteps, nx, ny);
}
