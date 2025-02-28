#include <stdio.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>

#define synchronize \
    HIP_CHECK(hipGetLastError()); \
    HIP_CHECK(hipDeviceSynchronize())

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}


#define starttime  { auto start = std::chrono::high_resolution_clock::now(); 

#define endtime \
  auto stop = std::chrono::high_resolution_clock::now(); \
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count(); \
    if (my_repeat_counter > 1) std::cout << duration; \
  }

#define repeat(X) for(int my_repeat_counter=1;my_repeat_counter <= (X); ++my_repeat_counter)

__device__ double f_1(double x, double a, int Nz)
{
  double R = x;

#pragma unroll 8
  for(int i = 0; i<Nz; ++i) {
    R += a*R+x;
  }
  return R;
}

__device__ double f_2(double x, double a, int Nz) 
{
  double R = 1;

#pragma unroll 8
  for(int i = 0; i<Nz; ++i) {
    R += x*R+a;
  }
  return R;
}

// GPU kernel
__global__ void fill_kernel_div(size_t n, double *x, double a, int Nz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  
  if(tid < n) {
    if(tid%2 == 0) {
      x[tid] = f_1(double(tid)/n, a, Nz);
    } else {
      x[tid] = f_2(double(tid)/n, a, Nz);
    }
  }
}

//
// GPU kernel
__global__ void fill_kernel_nodiv(size_t n, double *x, double a, int Nz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  
  if(tid <n) {
    if (((tid)/64)%2 == 0) {
      x[tid] = f_1(double(tid)/n, a, Nz);
    } else {
      x[tid] = f_2(double(tid)/n, a, Nz);
    }
  }
}

__global__ void fill_kernel_noif(size_t n, double *x, double a, int Nz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  
  if(tid < n) {
    x[tid] = f_2(double(tid)/n, a, Nz);
    /* x[tid] = f_2(double(tid)/n, x[tid], Nz); */
  }
}

int main(void)
{
    // set problem size
    const size_t n = pow(2,24);
    
    int Nz = 1;
    // allocate device memory
    double *d_x;
    double *d_y; 
    double a = 0.1;
    HIP_CHECK(hipMalloc(&d_x, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&d_y, sizeof(double) * n));

    // launch kernel
    const int blocksize = 256;
    const int gridsize = (n - 1 + blocksize) / blocksize;


    std::cout << "% noif, nodiv, div, Nz \n"; // TODO: count flops
    repeat(10) {
      Nz = Nz << 1;
      starttime
        fill_kernel_noif<<<gridsize, blocksize>>>(n, d_x, a, Nz);
      synchronize;
      endtime
        if (my_repeat_counter > 1) std::cout << ", ";

        starttime
        fill_kernel_nodiv<<<gridsize, blocksize>>>(n, d_x, a, Nz);
      synchronize;
      endtime
        if (my_repeat_counter > 1) std::cout << ", ";

      starttime
        fill_kernel_div<<<gridsize, blocksize>>>(n, d_x, a, Nz);
      synchronize;
      endtime
        if (my_repeat_counter > 1) std::cout << ", " << Nz << "\n";
    }

    // copy data to the host and print
    double x[n];
    HIP_CHECK(hipMemcpy(x, d_x, sizeof(double) * n, hipMemcpyDeviceToHost));
    /* printf("%f %f %f %f ... %f %f\n", */
            /* x[0], x[1], x[2], x[3], x[n-2], x[n-1]); */

    return 0;
}
