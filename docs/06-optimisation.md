---
title:  Kernel optimisation and profiling
author: CSC Training
date:   2021-11
lang:   en
---

# Libraries (I)

 NVIDIA  | HIP  | ROCm  | Description  |
|---|---|---|---|
| cuBLAS  | hipBLAS  |  rocBLAS | Basic Linear Algebra Subroutines  |
| cuFFT   | hipFFT |  rocfft |  Fast fourier Transform Library |
| cuSPARSE | hipSPARSE  | rocSPARSE  | Sparse BLAS + SMV  |
| cuSOLVER  | hipSOLVER  | rocSOLVER  | Lapack library  |
| AMG-X  |   | rocALUTION  | Sparse iterative solvers and preconditioners with Geometric and Algebraic MultiGrid  |
| Thrust  | hipThrust  | rocThrust  | C++ parallel algorithms library  |


---

# Libraries (II) 

 NVIDIA  | HIP  | ROCm  | Description  |
|---|---|---|---|
| CUB  | hipCUB  | rocPRIM  | Low level Optimized Parllel Primitives  |
| cuDNN  |   | MIOpen  | Deep learning solver library  |
| cuRAND  | hipRAND  |  rocRAND | Random number generator library  |
| EIGEN   | EIGEN |  EIGEN |  C++ template library for linear algebra: matrices, vectors, numerical solvers |
| NCCL  |   |  RCCL | Communications Primitives Library based on the MPI equivalents   |

---

# hipBLAS

![width:1000px height:13cm](./img/hipblas.png)

---

# Kernels

You can call a kernel with the command:

```
hipLaunchKernelGGL(kernel_name, dim3(Blocks), dim3(Threads), 0, 0, arg1, arg2, ...);
```

or
```
kernel_name<<<dim3(Blocks), dim3(Threads),0,0>>>(arg1,arg2,...);
```
* where blocks are for the 3D dimensions of the grid of blocks dimensions
* threads for the 3D dimentions of a block of threads
* 0 for bytes of dynamic LDS space
* 0 for stream
* kernel arguments 

---

# Exercise Nbody


* 32768 number of small particles, 2000 time steps
* Executing the code on an NVIDIA V100 GPU, the execution time is 68.5 seconds.
* Compile and execute the code Nbody on an AMD MI100 GPU
    * `hip-programming/nbody`

---


# AMD MI100 architecture 

![width:1000px height:13cm](./img/mi100_arch.png)

---

# Compute Units (CU)

* Each CU is a 64-wide execution unit, so multiple of 64 as the thread limit.  
* The 64-wide execution is sub-divided into 4 SIMD units of 16 elements.  
* For a 16-wide SIMD instruction, the best possible latency is 4 cycles.  
* So, you need at least 4 SIMD instructions in flight to saturate the SIMD units.

Finally, using 256 threads per block would give the best performance in most cases.

* Change the threads per block to 256 for the all the calls to launch kernel at Nbody exercise and discuss what is the performance improvement.
* Is the workload enough to stress the AMD MI100 GPUs?

---

# Tips

* Get familiar with the GPU hardware
* Compute units, memory etc.
* Using 240 blocks (1 x 120 CUs) and 256 threads per block provides good performance, maybe additional tuning required

---

# Copy matrix

## Example

```
__global__ void copy_kernel(float *in, float *out, int width, int height) {
  int x_index = blockIdx.x * tile_dim + threadIdx.x;
  int y_index = blockIdx.y * tile_dim + threadIdx.y;

  int index = y_index * width + x_index;

  out[index] = in[index];
}
```
```
  int block_x = width / tile_dim;
  int block_y = height / tile_dim;
   hipLaunchKernelGGL(copy_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);
   hipDeviceSynchronize();
```





