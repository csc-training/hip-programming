---
title:    Kernel optimisation
subtitle: GPU programming with HIP
author:   CSC Training
date:     2022-11
lang:     en
---

# Libraries (I)

| NVIDIA   | HIP       | ROCm       | Description                                                                         |
| -------- | --------- | ---------- | ----------------------------------------------------------------------------------- |
| cuBLAS   | hipBLAS   | rocBLAS    | Basic Linear Algebra Subroutines                                                    |
| cuFFT    | hipFFT    | rocfft     | Fast fourier Transform Library                                                      |
| cuSPARSE | hipSPARSE | rocSPARSE  | Sparse BLAS + SMV                                                                   |
| cuSOLVER | hipSOLVER | rocSOLVER  | Lapack library                                                                      |
| AMG-X    |           | rocALUTION | Sparse iterative solvers and preconditioners with Geometric and Algebraic MultiGrid |
| Thrust   | hipThrust | rocThrust  | C++ parallel algorithms library                                                     |


# Libraries (II)

| NVIDIA | HIP     | ROCm    | Description                                                                   |
| ------ | ------- | ------- | ----------------------------------------------------------------------------- |
| CUB    | hipCUB  | rocPRIM | Low level Optimized Parllel Primitives                                        |
| cuDNN  |         | MIOpen  | Deep learning solver library                                                  |
| cuRAND | hipRAND | rocRAND | Random number generator library                                               |
| EIGEN  | EIGEN   | EIGEN   | C++ template library for linear algebra: matrices, vectors, numerical solvers |
| NCCL   |         | RCCL    | Communications Primitives Library based on the MPI equivalents                |


# hipBLAS

![](./img/hipblas.png){width=1600px}


# Kernels

You can call a kernel with the command:

```cpp
hipLaunchKernelGGL(kernel_name, dim3(Blocks), dim3(Threads), 0, 0, arg1, arg2, ...);
```

or
```cpp
kernel_name<<<dim3(Blocks), dim3(Threads),0,0>>>(arg1,arg2,...);
```
* where blocks are for the 3D dimensions of the grid of blocks dimensions
* threads for the 3D dimentions of a block of threads
* 0 for bytes of dynamic LDS space
* 0 for stream
* kernel arguments


# Exercise Nbody


* 32768 number of small particles, 2000 time steps
* Executing the code on an NVIDIA V100 GPU, the execution time is 68.5
  seconds.
* Compile and execute the code Nbody on an AMD MI100 GPU
    * `hip-programming/nbody`


# AMD MI100 architecture

![](./img/mi100_arch.png){width=1200px}


# Compute Units (CU)

<div class="column" width=75%>
* Each CU is a 64-wide execution unit, so multiple of 64 as the thread limit.
    * The 64-wide execution is sub-divided into 4 SIMD units of 16 elements.
    * For a 16-wide SIMD instruction, the best possible latency is 4 cycles.
    * So, you need at least 4 SIMD instructions in flight to saturate the
      SIMD units.

</div>

<div class="column" width=23%>
![](img/CUgray.png){width=140%}
</div>
* Using 256 threads per block would give the best performance in many
cases, though in general more tuning is required
* Similarly on Nvidia cards


# Global memory access in device code

- Global memory access from the device has high latency
- Threads are executed in warps, memory operations are grouped in a similar
  fashion
- Memory access is optimized for coalesced access where threads read from and write to successive memory locations
- Exact alignment rules and performance issues depend on the architecture

# Coalesced memory access

<div class="column">
- The global memory loads and stores consist of transactions of a certain size (eg. 32 bytes)
- If the threads within a warp access data within such a block of 32 bytes,
  only one global memory transaction is needed
</div>

<div class="column">
- Now, 32 threads within a warp can each read a different 4-byte integer value with just 4 transactions
- When the stride between each 4-byte integer is increased, more transactions are required (up to 32 for the worst case)!
</div>

# Coalesced memory access example

<div class="column">
```
__global__ void memAccess(float *out, float *in)
{
 int tid = blockIdx.x*blockDim.x + threadIdx.x;
 if(tid != 12) out[tid + 16] = in[tid + 16];
}
``` 
![](img/coalesced_access_4.png){width=80%}
</div>

<div class="column">
```
__global__ void memAccess(float *out, float *in)
{
 int tid = blockIdx.x*blockDim.x + threadIdx.x;
 out[tid + 1] = in[tid + 1];
}
```
![](img/coalesced_access_3.png){width=80%}
</div>

# Shared Memory I
- Fast memory on the CU
- Shared memory is divided into banks
- Each bank can service one address per cycle
- Conflicting accesses are serialized
- Conflicts solved by padding

# Shared Memory II
<div class="column">
![](img/NoBankConflicts.jpeg){width=100%}
</div>
<div class="column">
![](img/BankConflicts.jpeg){width=100%}
</div>

### Example

# Shared Memory III
- Can be used as user controled cache
- Useful to reduce the amount of global memory operations
- Can be used as buffer to transform uncoalesced operations in coalesced 
- Limited resources per CU

# Low level optimizations
- Avoid branching
  - All threads in  wavefront should execute the sme instruction
    - `if(tid%2==0)` would result in 32 branches
- Sometimes recomputing can be faster than reading from the memory
- Depeding on the problem, consider using lower precision instead of `double` 

# Optimizing matrix operations. `B(i,j)=A(j,i)` 
![](img/transpose_img.png){.center width=70%}


# Optimizing matrix operations. Copy matrix

* Simple copy operation as base
```cpp
__global__ void copy_kernel(float *in, float *out, int width, int height) {
  int x_index = blockIdx.x * tile_dim + threadIdx.x;
  int y_index = blockIdx.y * tile_dim + threadIdx.y;

  int index = y_index * width + x_index;

  out[index] = in[index];
}
```

```cpp
  int block_x = width / tile_dim;
  int block_y = height / tile_dim;
   hipLaunchKernelGGL(copy_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);
   hipDeviceSynchronize();
```


# Profile the code

```shell
rocprof --stats ./copy
```

```shell
cat results.stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"copy_kernel(float*, float*, int, int) [clone .kd]",1,165920,165920,100.0
```

So the duration of the Copy kernel is 165920 ns

```shell
rocprof -i metrics_copy_kernel.txt -o metrics_copy.csv ./copy
```

```shell
cat metrics_copy.csv
GPUBusy,Wavefronts,VALUInsts,SALUInsts,SFetchInsts,MemUnitStalled,
VALUUtilization,VALUBusy,SALUBusy,L2CacheHit,WriteUnitStalled,LDSBankConflict
100,262144,11,1,2,13,100,13,1,0,6,0
```


# Matrix transpose naive

```cpp
__global__ void transpose_kernel(float *in, float *out, int width, int height) {
  int x_index = blockIdx.x * tile_dim + threadIdx.x;
  int y_index = blockIdx.y * tile_dim + threadIdx.y;

  int in_index = y_index * width + x_index;
  int out_index = x_index * height + y_index;

  out[out_index] = in[in_index];
}
```

```shell
rocprof --stats ./matrix_transpose_naive
```

```shell
cat results.stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"transpose_kernel(float*, float*, int, int) [clone .kd]",1,418083,418083,100.0
```

The duration is 418083 ns, 2.5 times slower


# Profile counters

```shell
rocprof -i metrics_matrix_transpose_naive_kernel.txt -o metrics_naive.csv ./matrix_transpose_naive
```

```shell
cat metrics_naive.csv
GPUBusy,Wavefronts,VALUInsts,SALUInsts,SFetchInsts,MemUnitStalled,
VALUUtilization,VALUBusy,SALUBusy,L2CacheHit,WriteUnitStalled,LDSBankConflict
100,262144,16,0,2,83,100,6,0,77,0,0
```


# Matrix transpose LDS

```cpp
__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index =
      (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  int out_index =
      (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

  tile[threadIdx.y][threadIdx.x] = in[in_index];

  __syncthreads();

  out[out_index] = tile[threadIdx.x][threadIdx.y];
}
```


# Profile statistics

```shell
rocprof --stats ./matrix_transpose_lds
```

```shell
cat results.stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"transpose_lds_kernel(float*, float*, int, int) [clone .kd]",1,227522,227522,100.0
```

The duration is 227522, which is 1.37 time slower instead of 2.5 times


# Profile Metrics

```shell
rocprof -i metrics_matrix_transpose_lds_kernel.txt -o metrics_lds.csv ./matrix_transpose_lds
```

```shell
cat metrics_lds.csv
GPUBusy,Wavefronts,VALUInsts,SALUInsts,SFetchInsts,MemUnitStalled,
VALUUtilization,VALUBusy,SALUBusy,L2CacheHit,WriteUnitStalled,LDSBankConflict
100,262144,20,2,2,26,100,26,2,0,0,67
```


# Explanation

* There is need for more optimization
* There are LDS bank conflicts


# Exercise

* Do the hipfort exercise
* Do the nbody exercise, also profile and visualize 
* Execute the presented exercise
* Copy the streams exercise, profile it and visualize the json file.

# Summary

- Uses existing provided libraries: `hipBLAS`, `hipFFT`, ...
- Coalesced memory access in kernels results in better
  performance
