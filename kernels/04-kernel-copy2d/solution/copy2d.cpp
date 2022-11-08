#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

/* copy all elements using threads in a 2D grid */
__global__ void copy2d_(int n, int m, double *src, double *tgt)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;  
  
  if (tidy < m && tidx < n) 
    tgt[tidy * n + tidx] = src[tidy * n + tidx];
}


int main(void)
{
    const int n = 600;
    const int m = 400;
    const int size = n * m;
    double x[size], y[size], y_ref[size];
    double *x_, *y_;

    // initialise data
    for (int i=0; i < size; i++) {
        x[i] = (double) i / 1000.0;
        y[i] = 0.0;
    }
    // copy reference values (C ordered)
    for (int j=0; j < m; j++) {
      for (int i=0; i < n; i++) {
        y_ref[j * n + i] = x[j * n + i];
      }
    }

    // allocate + copy initial values
    hipMalloc((void **) &x_, sizeof(double) * size);
    hipMalloc((void **) &y_, sizeof(double) * size);
    hipMemcpy(x_, x, sizeof(double) * size, hipMemcpyHostToDevice);
    hipMemcpy(y_, y, sizeof(double) * size, hipMemcpyHostToDevice);

    // define grid dimensions + launch the device kernel
    const int blocksize_x = 64;
    const int blocksize_y = 4;

    dim3 threads(blocksize_x, blocksize_y, 1);
    dim3 blocks(
        (n - 1 + blocksize_x) / blocksize_x, 
        (m - 1 + blocksize_y) / blocksize_y, 
        1);
    copy2d_<<<blocks, threads>>>(n, m, x_, y_);

    // copy results back to CPU
    hipMemcpy(y, y_, sizeof(double) * size, hipMemcpyDeviceToHost);

    // confirm that results are correct
    double error = 0.0;
    for (int i=0; i < size; i++) {
        error += abs(y_ref[i] - y[i]);
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", y_ref[42 * m + 42]);
    printf("     result: %f at (42,42)\n", y[42 * m + 42]);

    return 0;
}
