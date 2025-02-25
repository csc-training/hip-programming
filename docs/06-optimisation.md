---
title:    Kernel optimisation
subtitle: GPU programming with HIP
author:   CSC Training
date:     2022-11
lang:     en
---

# Kernel optimisation strategies

1. Use existing libraries
2. Minimise host-device data transfers
3. Minimise device memory-compute unit data transfers
4. Optimise for coalesced memory access
5. Avoid branching within warp
6. Minimise number of active local variables

# 1. Libraries (I)

::: notes

- Before you optimize, use libraries

:::

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
| CUB    | hipCUB  | rocPRIM | Low level Optimized Parallel Primitives                                        |
| cuDNN  |         | MIOpen  | Deep learning solver library                                                  |
| cuRAND | hipRAND | rocRAND | Random number generator library                                               |
| EIGEN  | EIGEN   | EIGEN   | C++ template library for linear algebra: matrices, vectors, numerical solvers |
| NCCL   |         | RCCL    | Communications Primitives Library based on the MPI equivalents                |

# 2. Host-device data transfers

### Peak theoretical bandwidth

| Link | Host-device | Device memory | 
|------|------------:|--------------:|
| LUMI-G MI250x | 36 GB/s$^{\star}$ | 1600 GB/s|
| PCIE4.0 x16 | $\sim$ 32 GB/s |  |
| A100 (Mahti) |  | 2000 GB/s |

$^\star$ per GCD, MI250x has 2 GCDs

::: notes

- Dont be afraid of host-device memory copies
- But be aware of the 2-order of magnitude BW difference

:::

# 3. Device global memory access

- Matrix multiplication: temporary variable avoids K-1 global memory accesses
- Fuse kernels if applicable

::::::{.columns}
:::{.column width=49%}
```cpp
  if (x < M && y < N) {
    for(int i = 0; i < K; ++i) {
      C[y+x*M] += A[x + i*M]*B[i + y*K];
    }
  }



```
:::
:::{.column width=49%}
```cpp
  if (x < M && y < N) {
    float tmp(0); 
    for(int i = 0; i < K; ++i) {
      tmp += A[x + i*M]*B[i + y*K];
    }
    C[y+x*M] = tmp;
  }
```
:::
::::::

# 3. Device global memory access

## Device memory hierarchy

<div class="column">
- Registers (per-thread-access)
- Local memory (per-thread-access)
- Shared memory (per-block-access)
- Global memory (global access)
</div>

<div class="column">
![](img/memlayout.png){width=80%}
</div>


# 3. Device global memory access

## Device memory hierarchy

<div class="column">
- Registers (per-thread-access)
    - Used automatically
    - Size on the order of kilobytes
    - Very fast access
- Local memory (per-thread-access)
    - Used automatically if all registers are reserved
    - Local memory resides in global memory
    - Very slow access
</div>

<div class="column">
- Shared memory (per-block-access)
    - Usage must be explicitly programmed
    - Size on the order of kilobytes
    - Fast access
- Global memory (per-device-access)
    - Managed by the host through HIP API
    - Size on the order of gigabytes
    - Very slow access
</div>

---

## MI250x Compute Units (CU)
::::::{.columns}

:::{.column width=40%}

- 64 kiB of **local data share** (LDS) memory
- 800 *scalar registers* (**SGPR**, 12.6 kiB)
  - up to 102 per warp
- 16 kiB of L1 cache
:::

:::{.column width=59%}
![](img/coarse_CU.svg){.center width=120%}
:::
::::::

- 4$\times$SIMD units, 16 threads per SIMD unit
  - 128 kiB *vector register* (**VGPR**) storage per SIMD unit $\Rightarrow$ 512 kiB register storage on CU
  - 512 4-byte registers per thread (2 kiB). 
- MI250x CU has matrix units, but they are tricky to program


::: notes
- Simplification
- Ballpark sizes for registers, LDS and cache
- MI250x GCD has 110 compute units
- Lot of register storage
- MI250x has also matrix units but they are tricker. Need intrinsics.
:::


# 4. Optimise for coalesced memory access

Memory is fetched in cache lines of 64/128 bytes from device memory

- warp requests non-consecutive elements, cannot coalesce memory operations

  ```cpp
   int tid = gridDim.x*blockIdx.y + (threadIdx.y + blockIdx.x)*blockDim.x + 8*threadIdx.x;
   if(tid < L) out[tid] = in[tid];
   
  ```

- consecutive elements, memory operations are coalesced

  ```cpp
   int tid = gridDim.x*blockIdx.y + (threadIdx.y + blockIdx.x)*blockDim.x + threadIdx.x;
   if(tid < K) out[tid] = in[tid];
  ```
- call: 
  ```cpp
     kernel<<<dim3(M,N,1), dim3(M_b, N_b, 1)>>>(float* in, float* out, size_t L);
  ```

---

## Uncoalesced memory access

::::::{.columns}
:::{.column width="76%"}
![](img/uncoalesced.svg){width="100%"}
:::
:::{.column width="23%"}
<br> <br> <br> <br> <br> <br> <br>
![](img/global-mem-arrow.svg){width="40%"}
:::
::::::

9 read OPs for 9 elements

---

## Coalesced memory access

::::::{.columns}
:::{.column width="76%"}
![](./img/coalesced.svg){width="100%"}
:::
:::{.column width="23%"}
<br> <br> <br> <br> <br> <br> <br>
![](img/global-mem-arrow.svg){width="40%"}
:::

::::::

4 read operations for 32 elements

---

## Local data share

- Variable defined as `__shared__` is shared within block 
- Divided into 32 banks of 512 Dwords (MI250x), each serve one address per cycle
- Use cases:
  - reduce ovelapping global memory operations <br>$\Rightarrow$ Remember to `__syncthreads()`!
  - User controlled cache
  - Transform uncoalesced memory OPs to coalesced
- Usage:
  ```cpp
  __shared__ float buf[256];
  ```

---

## Local data share

### **Advanced optimization**: avoid bank conflicts

<br>

<div class="column">
![](img/NoBankConflicts.jpeg){width=100%}
</div>
<div class="column">
![](img/BankConflicts.jpeg){width=100%}
</div>


# 5. Avoid branching within warp


::::::{.columns}
:::{.column width=60%}
- Both branches are executed sequentially but if statement value is used as a mask
- *However*
  - Compiler is pretty good at optimizing
  - Even if both branches are executed, code on right is memory-bound
:::
:::{.column width=39%}
```cpp
if ( (tid%2) == 0) {
  c[tid] = b[tid]-a[tid];
} else {
  c[tid] = a[tid]-b[tid];
}
```
```cpp
bool mask = (tid%2) == 0;
c[tid] = mask*(b[tid]-a[tid]);
c[tid] = mask*(a[tid]-b[tid]);
```
"Solution"
```cpp
if ( ((tid/16)%2) == 0) {
  c[tid] = b[tid]-a[tid];
} else {
  c[tid] = a[tid]-b[tid];
}
```
:::
::::::

---

## Branching across SIMD units

::::::{.columns}
:::{.column width=59%}
- $N=2^{29}$ doubles $\sim$ 512 MiB
 
Table: *Cost of branching at different optimization levels*

  | branching | `-O0` (ms) | `-O1` (ms)  |
  |-----------|-----:|-----:|
  | *Good* | 181 | 9 |
  | *Bad* | 320 | 9 | 
  | *single branch* | 108 | 6 |


:::
:::{.column width=39%}
*Good*
```cpp
if (((tid)/16)%2 == 0) {
  x[tid] = sin(M_PI*double(tid)/N);
} else {
  x[tid] = cos(M_PI*double(tid)/N);
}
```
*Bad*
```cpp
if(tid%2 == 0) {
  x[tid] = sin(M_PI*double(tid)/N);
} else {
  x[tid] = cos(M_PI*double(tid)/N);
}
```
*Single branch*
```cpp
x[tid] = sin(M_PI*double(tid)/N);
```
:::
::::::

# non-renewed material follows {.section}

# Low level optimizations 
- Avoid branching
  - All threads in  wavefront should execute the same instruction
    - `if(tid%2==0)` would result in 2 branches
    -  better use `if(tid<N/2)` - Sometimes recomputing can be faster than reading from the memory - Depending on the problem, consider using lower precision instead of `double` (math functions are available for `single` and `half` precision )

# Optimizing matrix operations. `B(i,j)=A(j,i)`  
![](img/transpose_img.png){.center width=60%}


# Copy operation as base

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
The duration is `0.174 ms`  and the effective bandwidth `717 GB/s`

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


The duration is `0.401 ms`  and the effective bandwidth `311 GB/s`




# Matrix transpose with shared memory

<small>
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
</small>

The duration is `0.185 ms`  and the effective bandwidth `674 GB/s`


# Matrix transpose with shared memory without bank conflicts

<small>
```cpp
__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim+1];

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
</small>

The duration is `0.179 ms`  and the effective bandwidth `697 GB/s`



# Other examples where shared memory is critical 
- Matrix-matrix/vector multiplication
- N-body problem
- reductions

# Summary

- Uses existing provided libraries: `hipBLAS`, `hipFFT`, ...
- Coalesced memory access in kernels results in better
  performance
- Use shared memory to reduce duplicate global memory accesses or make the acceses coalesced, but watch out for bank conflicts
- Try to avoid branching of the threads inside a wavefront