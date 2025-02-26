---
title:    Fortran and HIP
subtitle: GPU programming with HIP
author:   CSC Training
date:     2022-11
lang:     en
---

# Fortran

* No native GPU support in Fortran:
    - HIP functions are callable from C, using `extern C; compiled with hipcc
    - interoperability with Fortran via `iso_c_binding`
    - linking with Fortran or `hipcc`
* Fortran + HIP:
    - needs wrappers and interfaces for all HIP calls
* Hipfort:
    - Fortran Interface For GPU Kernel Libraries
      - HIP: HIP runtime, hipBLAS, hipSPARSE, hipFFT, hipRAND, hipSOLVER
      - ROCm: rocBLAS, rocSPARSE, rocFFT, rocRAND, rocSOLVER
      - memory management: `hipMalloc`, `hipMemcpy`

# HIPFort for SAXPY (`Y=Y+a*X`). Fortran Code
<small>
<div class="column" width=45%>>
```cpp
program saxpy
  use iso_c_binding
  use hipfort
  use hipfort_check

  implicit none
  interface
     subroutine launch(y,x,b,N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr),value :: y,x
       integer, value :: N
       real, value :: b
     end subroutine
  end interface

  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: dy = c_null_ptr
  integer, parameter :: N = 400000000
  integer, parameter :: bytes_per_element = 4
  integer(c_size_t), parameter :: Nbytes = N*bytes_per_element
  real, allocatable,target,dimension(:) :: x, y
  real, parameter ::  a=2.0
```
</div>

<div class="column" width=53%>>
```cpp
  allocate(x(N));allocate(y(N))

  x = 1.0;y = 2.0

  call hipCheck(hipMalloc(dx,Nbytes))
  call hipCheck(hipMalloc(dy,Nbytes))

  call hipCheck(hipMemcpy(dx, c_loc(x), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y), Nbytes, hipMemcpyHostToDevice))

  call launch(dy, dx, a, N)

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(y), dy, Nbytes, hipMemcpyDeviceToHost))

  write(*,*) "Max error: ", maxval(abs(y-4.0))

  call hipCheck(hipFree(dx));call hipCheck(hipFree(dy))

  deallocate(x);deallocate(y)

end program testSaxpy
```
</div>
</small>

# HIPFort for SAXPY (`Y=Y+a*X`). HIP code
<div class="column">
```cpp
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void saxpy(float *y, float *x, 
                      float a, int n)
{
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < n) {
      y[i] = y[i] + a*x[i];
    }
}
``` 

</div>

<div class="column">
``` cpp
extern "C"{
void launch(float *dout, float *da, 
            float db, int N)
  {
     dim3 tBlock(256,1,1);
     dim3 grid(ceil((float)N/tBlock.x),1,1);
     
     saxpy<<<grid, tBlock>>>(dout, da, db, N);
  }
}
```
</div>

# Compilation

**NVIDIA**
```
gfortran -cpp -I$HIPFORT_HOME/include/hipfort/nvptx "-DHIPFORT_ARCH=\"nvptx\""  -c <fortran_code>.f90  
hipcc "--gpu-architecture=sm_80" --x cu -c <hip_kernels>.cpp
hipcc -lgfortran  $LIB_FLAGS  "--gpu-architecture=sm_80" -I$HIPFORT_HOME/include/hipfort/nvptx -L$HIPFORT_HOME/lib/ -lhipfort-nvptx \
        <fortran_code>.o <hip_kernels>.o  -o main
```
**LUMI**
```
ftn -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -c <fortran_code>.f90
hipcc -c <hip_kernels>.cpp
ftn  -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS \
        <fortran_code>.o <hip_kernels>.o -o main 
```

# Summary

* No native GPU support in Fortran
* HIP functions are callable from C, using `extern C`
  - `iso_c_binding` 
  - GPU objects are of type `c_ptr` in Fortran
* Hipfort provides Fortran interfaces for GPU libraries
