<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# HIPBLAS Matrix Multiplication Exercise

In this exercise you will optimize a naive GPU matrix-matrix multiplication
by replacing the manual HIP kernel implementation with a call to HIPBLAS.
General Matrix Matrix Multiplication (GEMM) is one of the most important kernels in HPC and AI:

You will:

1. Understand a simple HIP GEMM kernel
2. Benchmark naive matrix multiplication
3. Replace the implementation using `hipblasSgemm`
4. Compare performance between:
   - naive kernel
   - hipBLAS implementation


## Tasks

### 1. Build and run the naive implementation

The code [gemm.cpp](gemm.cpp) contains a naive implementation that launches one 
thread per output element. 

Compile:
```bash
hipcc -o gemm gemm.cpp -O3 --offload-arch=gfx90a
```

To run the code, one provides the matrix dimension and number of repeates as 
command line arguments:
```bash
srun ... gemm 2048 10
```

You may try to run the code also under `rocprofv3` with the summary mode:
```bash
srun ... rocprofv3 -r -T -S gemm 2048 10
```
You can check the meaning of `rocprofv3` input parameters and how to print summary
output to a file with `rocprofv3 --help`


### 2. Use hipBLAS for the GEMM operation

hipBLAS documentation is available at https://rocm.docs.amd.com/projects/hipBLAS, however, note
that cuBLAS (which has practically the same API) contains more detailed documentation on the usage.
cuBLAS documentation can be found at https://docs.nvidia.com/cuda/cublas/

In short the steps needed for using hip/cuBLAS are:
1. include the `<hipblas/hipblas.h>` header
2. create a hipBLAS handle with `hipblasCreate` function
3. call `hipblasSgemm`
   - as operation (whether matrices are transposed or not) we use here `HIPBLAS_OP_N` for both input matrices
   - hipBLAS performs the general operation `C = alpha * A * B + beta * C`, for our basic matrix multiplication
     `alpha = 1.0` and `beta = 0.0`
   - the leading dimensions `lda`, `ldb` and `ldc` are here the same as the first matrix dimensions
4. link the code agains `-lhipblas`
   ```bash
   hipcc -o gemm gemm.cpp -O3 --offload-arch=gfx90a -lhipblas
   ```
5. it is good practive to destroy all hipBLAS handles with `hipblasDestroy` once they are no longer needed

Compare the performance of hipBLAS to naive implementation with various matrix sizes

### (Bonus) Type of input data and performance

As a fun bonus task, you can check how the hipBLAS performance can depend a lot on the type of input data.
By default, the input matrices are made of random `float`s, but by building the code with `INTEGER_INIT` 
preprocessor define integers (converted to floats) are used:
```
hipcc -o gemm -DINTEGER_INIT gemm.cpp -O3 --offload-arch=gfx90a -lhipblas
```
You should notice that with integers the performance is improved a lot. The reason for this seems to be in the power
management of AMD MI250x, when using integers GPU needs to "flip fewer" bits and it can run on higher clock frequency.

The feature does not probably provide any optimization prospects, but it is good to be aware that it may bias results 
when comparing performance of microbenchmarks to that of real applications if initialization is not done properly.


