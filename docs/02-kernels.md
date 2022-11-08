---
title:    HIP and GPU kernels
subtitle: GPU programming with HIP
author:   CSC Training
date:     2022-11
lang:     en
---

# HIP

- Heterogeneous-computing Interface for Portability
    - AMD effort to offer a common programming interface that works on both
      CUDA and ROCm devices
- HIP is a C++ runtime API and kernel programming language
    - standard C++ syntax, uses nvcc/hcc compiler in the background
    - almost a one-on-one clone of CUDA from the user perspective
    - allows one to write portable GPU codes
- AMD offers also a wide set of optimised libraries and tools


# HIP kernel language

- Qualifiers: `__device__`, `__global__`, `__shared__`, ...
- Built-in variables: `threadIdx.x`, `blockIdx.y`, ...
- Vector types: `int3`, `float2`, `dim3`, ...
- Math functions: `sqrt`, `powf`, `sinh`, ...
- Arithmetic functions: `atomicAdd`, `atomicMin`, ...
- Intrinsic functions: `__syncthreads`, `__threadfence`, ...


# HIP API

- Device init and management
- Memory management
- Execution control
- Synchronisation: device, stream, events
- Error handling, context handling, ...


# HIP programming model

- GPU accelerator is often called a *device* and CPU a *host*
- Parallel code (kernel) is launched by the host and executed on
  a device by several threads
- Code is written from the point of view of a single thread
    - each thread has a unique ID


# Example: Hello world

```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void)
{
    int count, device;

    hipGetDeviceCount(&count);
    hipGetDevice(&device);

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    return 0;
}
```


# AMD GPU terminology

- Compute Unit
    - one of the parallel vector processors in a GPU
- Kernel
    - function launched to the GPU that is executed by multiple parallel
      workers


# AMD GPU terminology (continued)

- Thread
    - individual lane in a wavefront
- Wavefront (cf. CUDA warp)
    - collection of threads that execute in lockstep and execute the same
      instructions
    - each wavefront has fixed number of threads (AMD: 64, NVIDIA 32)
    - number of wavefronts per workgroup is chosen at kernel launch
- Workgroup (cf. CUDA thread block)
    - group of wavefronts (threads) that are on the GPU at the same time and
      are part of the same compute unit (CU)
    - can synchronise together and communicate through memory in the CU


# GPU programming considerations

- GPU model requires many small tasks executing a kernel
    - e.g. can replace iterations of loop with a GPU kernel call
- Need to adapt CPU code to run on the GPU
    - rethink algorithm to fit better into the execution model
    - keep reusing data on the GPU to reach high occupancy of the hardware
    - if necessary, manage data transfers between CPU and GPU memories
      carefully (can easily become a bottleneck!)


# Grid: thread hierarchy

<div class="column"  width=44%>
- Kernels are executed on a 3D *grid* of threads
    - threads are partitioned into equal-sized *blocks*
- Code is executed by the threads, the grid is just a way to organise the
  work
- Dimension of the grid are set at kernel launch
</div>

<div class="column"  width=54%>
![](img/ThreadExecution_new.jpg){.center width=56%}

- Built-in variables to be used within a kernel:
    - `threadIdx`, `blockIDx`, `blockDim`, `gridDim`
</div>


# Kernels

- Kernel is a function to be executed by the GPU
- A kernel definition should be of `void` type and must be declared with the
  `__global__` attribute
- Any function called from a kernel must be declared with `__device__` attribute
- All pointers passed to a kernel should point to memory accessible from
  the device
- Unique thread and block IDs can be used to distribute work


# Example: axpy

```cpp
__global__ void axpy_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        y[tid] += a * x[tid];
    }
}
```

- Global ID `tid` calculated based on the thread and block IDs
    - only threads with `tid` smaller than `n` calculate
    - works only if number of threads ≥ `n`


# Example: axpy (revisited)

```cpp
__global__ void axpy_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}
```

- Handles any vector size, but grid dimensions should be still "optimised"


# Launching kernels

- Kernels are launched with one of the two following options:
  - CUDA syntax (recommended, because it works on CUDA and HIP both):
  ```cpp
  somekernel<<<blocks, threads, shmem, stream>>>(args)
  ```
  
  - Native HIP syntax:
  ```cpp
  hipLaunchKernelGGL(somekernel, blocks, threads, shmem, stream, args)
  ```

- Grid dimensions are obligatory, `shmem`, and `stream` are optional arguments for CUDA syntax, and can be `0` for the native HIP syntax
  - Must have an integer type or vector type of `dim3`
- Kernel execution is asynchronous with the host




# Simple memory management

- In order to calculate something on the GPUs, we usually need to
  allocate device memory and pass a pointer to it when launching a kernel
- Similarly to `cudaMalloc` (or simple `malloc`), HIP provides a function to
  allocate device memory: `hipMalloc()`

```cpp
double *x_
hipMalloc((void **) &x_, sizeof(double) * n);
```

- To copy data to/from device, one can use `hipMemcpy()`:

```cpp
hipMemcpy(x_, x, sizeof(double) * n, hipMemcpyHostToDevice);
hipMemcpy(x, x_, sizeof(double) * n, hipMemcpyDeviceToHost);
```


# Error checking
<small>

* Always use HIP error checking with larger codebases!
  * It has low overhead, and can save a lot of debugging time!

```cpp
#define HIP_ERR(err) (hip_error(err, __FILE__, __LINE__))
inline static void hip_error(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(1);
  }
}
```
* Here we wrap a HIP call into the above defined HIP_ERR macro:
```cpp
inline static void* alloc(size_t bytes) {
  void* ptr;
  HIP_ERR(hipMallocManaged(&ptr, bytes));
  return ptr;
}

```

* For simplicity, most exercises of this course do not have error checking, but error checking is strongly recommended when building bigger projects.

</small>


# Example: fill (complete device code and launch)

<small>
<div class="column">
```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void fill_(int n, double *x, double a)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        x[tid] = a;
    }
}
```
</div>

<div class="column">
```cpp
int main(void)
{
    const int n = 10000;
    double a = 3.4;
    double x[n];
    double *x_;

    // allocate device memory
    hipMalloc(&x_, sizeof(double) * n);

    // launch kernel
    dim3 blocks(32);
    dim3 threads(256);
    hipLaunchKernelGGL(fill_, blocks, threads, 0, 0, n, x_, a);

    // copy data to the host and print
    hipMemcpy(x, x_, sizeof(double) * n, hipMemcpyDeviceToHost);
    printf("%f %f %f %f ... %f %f\n",
            x[0], x[1], x[2], x[3], x[n-2], x[n-1]);

    return 0;
}
```
</div>
</small>


# Summary

- HIP supports both AMD and NVIDIA GPUs
- HIP contains API functions for writing GPU code
  - Some functions are callable from host, some from device
- Kernels launch grid of blocks, each block consists of many threads
    - Each block is executed in wavefronts of 64 (AMD) or 32 (NVIDIA) threads
- Kernels need to be declared `__global__` and `void` and are launched with special syntaxes
