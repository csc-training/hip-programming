/*
 * This code in its current form uses the default stream
 * Task is to:
 *   - create a stream
 *   - copy memory to/from device with that stream
 *   - launch the readymade kernel using that stream
 *   - copy data back to the host using the stream
 *   - destroy the stream
 */
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <hip/hip_runtime.h>

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

// GPU kernel definition
__global__ void kernel(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n) {
    float x = (float)tid;
    float s = sinf(x);
    float c = cosf(x);
    a[tid] = a[tid] + sqrtf(s*s+c*c);
  }
}

float max_error(float *a, int n)
{
  float max_err = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > max_err) max_err = error;
  }
  return max_err;
}

int main() {
  const size_t N = 1<<9;

  constexpr int blocksize = 256;
  constexpr int gridsize =(N-1+blocksize)/blocksize;
  constexpr size_t N_bytes = N*sizeof(float);

  float *a;
  float *d_a;

  // Initialize a custom stream
  hipStream_t stream;
  HIP_ERRCHK(hipStreamCreate(&stream));

  a = (float*) malloc(N_bytes);
  HIP_ERRCHK(hipMallocAsync((void**)&d_a, N_bytes, stream));

  memset(a, 0, N_bytes);

  // Copy data to device
  HIP_ERRCHK(hipMemcpyAsync(d_a, a, N_bytes, hipMemcpyHostToDevice, stream));
  
  // Launch GPU kernel
  kernel<<<gridsize, blocksize,0,stream>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());

  // Copy data back to host
  HIP_ERRCHK(hipMemcpyAsync(a, d_a, N_bytes, hipMemcpyDeviceToHost, stream));

  // Synchronize before printing
  HIP_ERRCHK(hipStreamSynchronize(stream));

  for (int i = 0; i < 10; i++)
    printf("%f ", a[i]);

  printf("\n");

  printf("error: %f\n", max_error(a, N));

  HIP_ERRCHK(hipFreeAsync(d_a, stream));
  free(a);
  HIP_ERRCHK(hipStreamDestroy(stream));

}
