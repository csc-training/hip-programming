#include <hip/hip_runtime.h>

namespace devices
{
  /* Very simple addition kernel */
  __global__ void add_kernel(double *in, int N)
  {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < N)
          in[tid]++;
  }
  
  void call_kernel(double *data, int N)
  {
    const int blocksize = 64;
    const int gridsize = (N - 1 + blocksize) / blocksize;
    add_kernel<<<blocksize, gridsize>>> (data, N);
  }

  void getDeviceCount(int *devCount) {
    hipGetDeviceCount(devCount);
  }

  void setDevice(int id) {
    hipSetDevice(id);
  }

  void freeDevice(void* ptr) {
    hipFree(ptr);
  }

  void freeHost(void* ptr) {
    hipHostFree(ptr);
  }

  void* mallocDevice(size_t bytes) {
    void* ptr;
    hipMalloc(&ptr, bytes);
    return ptr;
  }

  void* mallocHost(size_t bytes) {
    void* ptr;
    hipHostMalloc(&ptr, bytes);
    return ptr;
  }

  void memcpy_d2h(void* dst, void* src, size_t bytes){
    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
  }

  void memcpy_h2d(void* dst, void* src, size_t bytes){
    hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
  }
}
