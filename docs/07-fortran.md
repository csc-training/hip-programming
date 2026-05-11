---
title:    GPU programming with Fortran
event:    Introduction to GPU programming
date:     May 2026
lang:     en
---

# GPU support for Fortran

<div class="column">
 - CUDA Fortran
    - Only NVIDIA, no HIP equivalent
- Hipfort
    - Fortran interfaces to HIP memory management routines
    - Kernels in C/C++, Fortran interface with `iso_c_binding`
</div>
<div class="column">
- OpenACC 
    - Good NVIDIA support
    - AMD support only in HPE/Cray
- OpenMP offloading
    - Wide compiler support still in progress
</div>
- C/C++ HIP/CUDA and own Fortran interfaces
    - Portable and performant but additional work from interfacing

# C/C++ HIP/CUDA and own Fortran interfaces

- All GPU code in C/C++
    - similar porting approaches as with plain C/C++
- Fortran standards since 2003 have support for C/C++ interoperability
    - Function bindings
    - C/C++ compatible data types  

```fortran
  interface
     function gpuMalloc_(ptr, size) bind(c, name=gpuMalloc)
       use iso_c_binding
       implicit none
       type(c_ptr), value :: ptr
       integer(c_size_t), value :: size
       integer :: gpuMalloc_
     end subroutine
  end interface
```

# C/C++ HIP/CUDA and own Fortran interfaces

<small>

<div class="column">
```fortran
  interface
     subroutine saxpy(dx, dy, a, n) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr), value :: dx, dy
       type(c_float) :: a
       integer(c_int), value :: n
     end subroutine
  end interface
...
  err = gpuMalloc(dx, n)
...
  call saxpy(dx, dy, a, n)
```
</div>

<div class="column">
```cpp
#include <hip/hip_runtime.h>

__global__ void saxpy_kernel(float *dy, float *dx, 
                      float a, int n) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < n) {
      dy[i] = dy[i] + a*dx[i];
    }
}

extern "C" {
void saxpy(float *dy, float *dx, float a, int N) {
     dim3 threads(256, 1, 1);
     dim3 blocks(ceil((float)N/threads), 1, 1);
     
     saxpy<<<blocks, threads>>>(dx, dy, a, N);
  }
}
```
</div>
</small>


# Summary

- No native GPU support in Fortran
- Currently, two portable approaches:
- OpenMP offloading
    - Compiler support still in progress
- C/C++ HIP/CUDA code and own Fortran interfaces
    - Additional work from interfacing boilerplate
        - AI tools can help a lot   
