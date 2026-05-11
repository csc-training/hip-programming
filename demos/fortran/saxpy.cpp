#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <stdio.h>

extern "C" {

/* =========================
   SAXPY KERNEL
   y = a*x + y
   ========================= */

__global__
void saxpy_kernel(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

/* =========================
   FORTRAN-CALLABLE WRAPPER
   ========================= */

void saxpy_(int n,
            float a,
            const float *x_d,
            float *y_d)
{
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    saxpy_kernel<<<gridSize, blockSize>>>(n, a, x_d, y_d);

    hipDeviceSynchronize();
}

}
