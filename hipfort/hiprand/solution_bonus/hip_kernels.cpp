#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void countInsideKernel(float *x, float *y, int *inside,  int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (x[idx] * x[idx] + y[idx] * y[idx] < 1.0f) {
            // Atomic increment to avoid race condition
            atomicAdd(inside, 1);
        }
    }
}

extern "C"
{
  void launch(float *x, float *y, int *inside_d,  int64_t N)
  {

     dim3 tBlock(256,1,1);
     dim3 grid(ceil((float)N/tBlock.x),1,1);
    
     countInsideKernel<<<grid, tBlock>>>( x, y, inside_d, N);
  }
}
