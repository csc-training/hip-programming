#include <stdio.h>
#include <hip/hip_runtime.h>

// GPU kernel
__global__ void fill_kernel(int n, double *x, double a)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  
  if(tid < n)
    x[tid] = tid * a;
}

int main(void)
{
    // set problem size
    const int n = 1e4;

    // allocate device memory
    double *d_x;
    hipMalloc(&d_x, sizeof(double) * n);

    // launch kernel
    const int blocksize = 256;
    const int gridsize = (n - 1 + blocksize) / blocksize;
    fill_kernel<<<gridsize, blocksize>>>(n, d_x, 3.0);

    // copy data to the host and print
    double x[n];
    hipMemcpy(x, d_x, sizeof(double) * n, hipMemcpyDeviceToHost);
    printf("%f %f %f %f ... %f %f\n",
            x[0], x[1], x[2], x[3], x[n-2], x[n-1]);

    return 0;
}
