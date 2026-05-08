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

  #error create a new stream with hipStream_t

  a = (float*) malloc(N_bytes);
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));

  memset(a, 0, N_bytes);

  #error replace hipMemcpy with its Async counterpart, to copy data to the device using your stream
  HIP_ERRCHK(hipMemcpy(d_a, a, N_bytes, hipMemcpyHostToDevice));
  
  #error specify your stream at kernel launch
  kernel<<<gridsize, blocksize,0,0>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());
  
  #error replace hipMemcpy with its Async counterpart to copy data back to host using your stream
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes, hipMemcpyDeviceToHost));

  #error synchronize the host with your stream, before continuing
  
  for (int i = 0; i < 10; i++)
    printf("%f ", a[i]);

  printf("\n");

  printf("error: %f\n", max_error(a, N));

  HIP_ERRCHK(hipFree(d_a));
  free(a);
  #error Destroy the stream

}
