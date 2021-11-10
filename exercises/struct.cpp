/* The purpose of this exercise is to 
 * run a loop accessing a struct from host and 
 * device using different memory management strategies.
 *
 * The function runHost() demonstrates the execution on
 * host. The task is to fill the functions runDeviceUnifiedMem()
 * and runDeviceExplicitMem() to do the same thing parallel
 * on the device. The latter function also requires further 
 * filling struct allocation and deallocation functions 
 * createDeviceExample() and freeDeviceExample().
 */

#include <cstdio>
#include <hip/hip_runtime.h>

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* Example struct to practise copying structs with pointers to device memory */
typedef struct 
{
	float *x;
  int *idx;
  int size;
} Example;

/* GPU kernel definition */
__global__ void hipKernel(Example* const d_ex)
{
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread < d_ex->size)
  {
    printf("x[%d]: %.2f, idx[%d]:%d/%d \n", thread, d_ex->x[thread], thread, d_ex->idx[thread], d_ex->size - 1);
  }
}

/* Run on host */
void runHost()
{
  // Allocate host struct
  Example *ex;
  ex = (Example*)malloc(sizeof(Example));
  ex->size = 10;

  // Allocate host struct members
  ex->x = (float*)malloc(ex->size * sizeof(float));
  ex->idx = (int*)malloc(ex->size * sizeof(int));

  // Initialize host struct members
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Print struct values from host
  printf("\nHost:\n");
  for(int i = 0; i < ex->size; i++)
  {
    printf("x[%d]: %.2f, idx[%d]:%d/%d \n", i, ex->x[i], i, ex->idx[i], ex->size - 1);
  }

  // Free host struct
  free(ex->x);
  free(ex->idx);
  free(ex);
}

/* Run on device using Unified Memory */
void runDeviceUnifiedMem()
{
  // Allocate struct using Unified Memory
  
  // Allocate struct members using Unified Memory

  // Initialize struct from host
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Print struct values from device by calling hipKernel()
  printf("\nDevice (UnifiedMem):\n");

  // Free struct
}

/* Create the device struct (needed for explicit memory management) */
Example* createDeviceExample(Example *ex)
{
  // Allocate device struct

  // Allocate device struct members

  // Copy arrays pointed by the struct members from host to device

  // Copy struct members from host to device

  // Return device struct
}

/* Free the device struct (needed for explicit memory management) */
void freeDeviceExample(Example *d_ex)
{
// Copy struct members (pointers) from device to host

  // Free device struct members

	// Free device struct
}

/* Run on device using Explicit memory management */
void runDeviceExplicitMem()
{
  // Allocate host struct

  // Allocate host struct members

  // Initialize host struct
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Allocate device struct and copy values from host to device
  Example *d_ex = createDeviceExample(ex);
  
  // Print struct values from device by calling hipKernel()
  printf("\nDevice (ExplicitMem):\n");

  // Free device struct
  freeDeviceExample(d_ex);

  // Free host struct
}

/* The main function */
int main(int argc, char* argv[]) 
{
  runHost();
  runDeviceUnifiedMem();
  runDeviceExplicitMem();
}