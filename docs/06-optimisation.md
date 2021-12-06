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


