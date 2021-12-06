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

--

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

* Executing the code on an NVIDIA V100 GPU, the execution time is 68.5 seconds.
* Compile and execute the code Nbody on an AMD MI100 GPU
    * `hip-programming/nbody`

