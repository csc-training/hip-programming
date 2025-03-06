# In this repository we include the files for the course of porting CUDA codes to HIP



# Instructions about porting the CUDA codes to HIP on Puhti

Basic converting instructions based on the https://gitlab.ci.csc.fi/compen/hpc-support/hpcs-team/-/wikis/gpu-programming/HIP

First load the `hip` module:

`module load hip/3.10.0`

### Using the hipexamine-perl.sh

This tool scans all the code and for files with CUDA prints a summary of related commands it found. Need to have 0 errors to be sure that it can handle most of the code, how many calls were converted from CUDA to HIP etc. This tool does not change the source code, it investigates. 

`hipexamine-perl.sh`

### Convert a file
Convert individual files with:

`hipify-perl --inplace filename`

### Convert a directory
Convert the whole directory using:

`hipconvertinplace-perl.sh Directory`

   * Before the conversion

`ls src/`

`Makefile.am  matMulAB.c  matMulAB.h matMul.c`

   * After the conversion

`ls src/`

`Makefile.am  matMulAB.c  matMulAB.c.prehip  matMulAB.h	matMulAB.h.prehip  matMul.c  matMul.c.prehip`

The old files are named *.prehip and the new HIP files named *.c *.h etc.

### Compilation

   1. Define the new compiler (wrapper), if required for your application

    export CC=hipcc

   2. If you do not modify more the code and try to compile while there is a dependency, then you can get an error like:

`matMulAB.c:21:10: fatal error: hipblas.h: No such file or directory
   21 | #include "hipblas.h"`

   3. Modify your Makefile/script etc. and add the corresponding paths for hipBLAS to appropriate flags (CFLAGS, LDFLAGS etc.)

   For the header:

`-I/appl/opt/rocm/rocm-3.10.0/hipblas/include/`

   For the library:

`-L/appl/opt/rocm/rocm-3.10.0/hipblas/lib -lhipblas`

### Then you can compile and execute.

### Available libraries (mentioning only the libraries that are supported by NVIDIA hardware):

`hipblas
hipcub
hiprand
hipsparse
`
**The hiprand library seems to be missing on Puhti.**

## Use hipfort for Fortran codes

Porting Fortran codes to HIP require more effort than C/C++ when the main code is CUDA Fortran. The procedure has been tested with a custom HIP installation. For codes constituted by Fortran + CUDA in C/C++, where there is no CUDA call in the Fortran file, the procedure is easier, hipify the CUDA code only and then compile with hipcc and link the files with hipcc.

```
module load hip/4.0.0
export HIPFORT_ARCHGPU=nvptx-sm_70
module load cuda
export ROCM=$HIP_PATH  //this is because the Makefile.hipfort is looking for the hipcc in this path
```

* If you have a Fortran with CUDA application, you need to put the kernel in .cu file for NVIDIA GPUs or .cpp for AMD GPUs. the file will be supporting iso_c_binding so we will call the kernel from Fortran.

As an example, we have this Fortran code, example.cuf

```
module mathOps
contains
  attributes(global) subroutine saxpy(x, y, a)
    implicit none
    real :: x(:), y(:)
    real, value :: a
    integer :: i, n
    n = size(x)
    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i <= n) y(i) = y(i) + a*x(i)
  end subroutine saxpy
end module mathOps

program testSaxpy
  use mathOps
  use cudafor
  implicit none
  integer, parameter :: N = 40000
  real :: x(N), y(N), a
  real, device :: x_d(N), y_d(N)
  type(dim3) :: grid, tBlock

  tBlock = dim3(256,1,1)
  grid = dim3(ceiling(real(N)/tBlock%x),1,1)

  x = 1.0; y = 2.0; a = 2.0
  x_d = x
  y_d = y
  call saxpy<<<grid, tBlock>>>(x_d, y_d, a)
  y = y_d
  write(*,*) 'Max error: ', maxval(abs(y-4.0))
end program testSaxpy
```

We split the above code to a Fortran file and an cu file. The .cu file will provide the kernel and the iso_c_binding call, for example hipsaxpy.cu:

```
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void saxpy(float *y, float *x, float a, int n)
{
    size_t i = blockDim.x * blockIdx.x  + threadIdx.x;
    if (i < n) y[i] = y[i] + a*x[i];
}

extern "C"
{
  void launch(float **dout, float **da, float db, int N)
  {
     dim3 tBlock(256,1,1);
     dim3 grid(ceil((float)N/tBlock.x),1,1);
    hipLaunchKernelGGL((saxpy), grid, tBlock, 0, 0, *dout, *da, db, N);
  }
}
```

and the Fortran file will be, main.f03

```
program testSaxpy
  use iso_c_binding
  use hipfort
  use hipfort_check

  implicit none
  interface
     subroutine launch(y,x,b,N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: y,  x
       integer, value :: N
       real, value :: b
     end subroutine
  end interface

  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: dy = c_null_ptr
  integer, parameter :: N = 40000
  integer, parameter :: bytes_per_element = 8 
  integer(c_size_t), parameter :: Nbytes = N*bytes_per_element
  real,allocatable,target,dimension(:) :: x, y

  real, parameter ::  a=2.0

  call hipCheck(hipMalloc(dx,Nbytes))
  call hipCheck(hipMalloc(dy,Nbytes))

  allocate(x(N))
  allocate(y(N))
  x = 1.0
  y = 2.0

  call hipCheck(hipMemcpy(dx, c_loc(x), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y), Nbytes, hipMemcpyHostToDevice))

  call launch(dy, dx, a, N)

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(y), dy, Nbytes, hipMemcpyDeviceToHost))

  write(*,*) 'Max error: ', maxval(abs(y-4.0))

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  deallocate(x)
  deallocate(y)

end program testSaxpy
```

Summary:
* We have to transfer data to and from the GPU with hip calls
* We need to allocate necessary data on the GPU
* Need to write the C-style file for the kernel
* Need to be sure to understand which part of the code should be changed/moved
* Change the blockDim%x  to blockDim.x etc. 

The 29 lines of the original code, became two files with total 52 lines of code with more than 30 new lines or changed code.

To compile now these two files you need a Makefile like this:

```
export HIPFORT_HOME=/appl/opt/rocm/rocm-4.0.0/hipfort/
include ${HIPFORT_HOME}/bin/Makefile.hipfort

OUTPUT_DIR ?= $(PWD)
APP         = $(OUTPUT_DIR)/saxpy

.DEFAULT_GOAL := all

all: $(APP)

$(APP): $(OUTPUT_DIR)/main.o $(OUTPUT_DIR)/hipsaxpy.o
        $(FC) $^ $(LINKOPTS) -o $(APP)

$(OUTPUT_DIR)/main.o: main.f03
        $(FC)  -c $^ -o $(OUTPUT_DIR)/main.o

$(OUTPUT_DIR)/hipsaxpy.o: hipsaxpy.cu
        $(CXX) -c $^ -o $(OUTPUT_DIR)/hipsaxpy.o

run: $(APP)
        HIP_TRACE_API=1 $(APP)

clean:
        rm -f $(APP) *.o *.mod *~

```


## Debug

Add in your submission script before srun:

```
export AMD_LOG_LEVEL=4
```

For example, with no debug mode:

```
srun: error: r01g01: task 0: Segmentation fault
srun: Terminating job step 4339273.0
```

with debug mode:

```
:3:rocdevice.cpp            :458 : 2193024923864 us: Initializing HSA stack.
:1:rocdevice.cpp            :466 : 2193024923948 us: hsa_init failed.
:4:runtime.cpp              :82  : 2193024923950 us: init
srun: error: r01g01: task 0: Segmentation fault
srun: Terminating job step 4339273.0
```

The outcome is that the used library does require NVIDIA hardware and it crashes immediately. IN a reall execution you will observe a lot of output data.

## Known issues

- RocFFT library works only with AMD hardware, need to investigate the hipFFT (21/12/2020)

- If your CUDA kernel, includes the dim3() call, then hipify will convert wrongly. Bug: https://github.com/ROCm-Developer-Tools/HIPIFY/issues/246 (21/12/2020)

- In CUDA, `__CUDACC__` is defined by `nvcc`, but the HIP equivalent `__HIPCC__` is defined in `hip_runtime.h`. Thus, if code uses `__CUDACC__` without `#include <cuda_runtime_api.h>`, one needs to add manually `#include <hip_runtime.h>` to have the automatically converted `__HIPCC__` to get defined. [Issue in Github](https://github.com/ROCm-Developer-Tools/HIP/issues/29)
- `hiprand` library can not be compiled on Nvidia systems
