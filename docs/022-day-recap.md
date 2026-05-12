---
title:  Monday recap
subtitle: Introduction to GPU programming
author:   CSC Training
date:     2026-05
lang:     en
---

# Monday recap

# Monday recap

* GPU is a massively parallel processor
* HIP/CUDA is an API for programming AMD/NVidia GPUs
  * C++ extension kernel language
* Threads are partitioned in blocks
  * Blocks ⟷  Compute unit / streaming multiprocessor
* Blocks are partitioned in grid
  * Grid ⟷  GPU

# Kernels

# Kernels

```c++
__global__ void mykernel(size_t n_x, size_t n_y, float* x) {
        const size_t tid_x = threadIdx.x + blockIdx.x*blockDim.x;
        const size_t tid_y = threadIdx.y + blockIdx.y*blockDim.y;

        if((tid_x < n_x) && (tid_y < n_y)) {
                // do something
        }
}
```

# Calling kernels: typical workflow

1. allocate memory
2. copy input data to gpu
3. launch kernel
4. copy data back to host

# Calling kernels

# Calling kernels
```c++
...
float* d_x;
float* x;
x = (float*) malloc(M*N*sizeof(float));

HIP_ERRCHK(hipMalloc(&d_x, M*N*sizeof(float)));
HIP_ERRCHK(hipMemcpy(d_x, x, M*N*sizeof(float), hipMemcpyDefault));

// 2D grid
dim3 blocks(num_blocks_x, num_blocks_y, 1);
dim3 threads(num_threads_x, num_threads_x, 1),

mykernel<<<blocks,threads>>>(M, N, d_x);  //CUDA Syntax
LAUNCH_KERNEL(mykernel, blocks, threads, 0, 0, M, N, d_x)); // Error checking macro

HIP_ERRCHK(hipMemcpy(x, d_x, M*N*sizeof(float), hipMemcpyDefault));

...
```

# Using LUMI

- Look at [exercise instructions](https://github.com/csc-training/hip-programming/blob/main/exercise-instructions.md).
- Load modules
- compile: `CC -xhip ...`
- runnig: `sbatch job.sh` or `srun ... <binary>`
- or `source setup_env_lumi` and `run_tue  <binary>`