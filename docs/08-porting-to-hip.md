---
title:  Porting Applications to HIP 
subtitle: GPU programming with HIP
author:   CSC Training
date:     2025-03
lang:     en
---

# Heterogeneous-Compute Interface for Portability

- code to run on both **AMD ROCm** and **NVIDIA CUDA** platforms with minimal changes
- the `hipcc` compiler driver calls different compilers depending on the architecture: on **NVIDIA** platforms calls `nvcc` 
- (very) similar to **CUDA**, but designed for cross-platform compatibility
- supports a strong subset of the **CUDA** runtime functionality
- enables fast translation of **CUDA API** calls: most calls can be converted in place by simply replacing `cuda` with `hip`

::: notes
HIP (Heterogeneous-Compute Interface for Portability) is a C++ runtime API and programming model designed by AMD to enable seamless portability between CUDA and ROCm-based GPU architectures. It provides an interface similar to CUDA, allowing developers to write GPU-accelerated code that can run on both NVIDIA and AMD GPUs with minimal changes. The HIP API includes equivalents for CUDA runtime functions, memory management, and kernel launches, as well as a HIPified version of libraries like cuBLAS (hipBLAS) and cuDNN (hipDNN). Developers can use hipify tools to automatically translate CUDA code to HIP, making it easier to migrate applications across different hardware platforms while maintaining high performance.
:::
# CUDA vs. HIP

<div class="column" width=45%>>
```cpp
// CUDA
```
</div>

<div class="column" width=45%>>
```cpp
// HIP
```
</div>

<small>
 <div class="column" width=45%>>
```cpp
cudaMalloc(&d_x,N*sizeof(double));
  
cudaMemcpy(d_x,x,N*sizeof(double),
              cudaMemcpyHostToDevice);
              
cudaDeviceSynchronize();

cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                &alpha, d_A, N, d_B, N, &beta, d_C, N);

kernel_name<<<gridsize, blocksize, 
              shared_mem_size, 
              stream>>>
              (arg0,arg1, ...);





              
``` 
</div>

<div class="column" width=45%>
```cpp
hipMalloc(&d_x,N*sizeof(double));

hipMemcpy(d_x,x,N*sizeof(double),
              hipMemcpyHostToDevice);

hipDeviceSynchronize();

hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, 
                 &alpha, d_A, N, d_B, N, &beta, d_C, N);

kernel_name<<<gridsize, blocksize, 
              shared_mem_size, 
              stream>>>
              (arg0,arg1, ...);

hipLaunchKernelGGL(kernel_name,
                    gridsize, 
                    blocksize, 
                    shared_mem_size, 
                    stream,arg0,arg1, ...);
```
</div>

</small>


# Porting a CUDA Project

- **Migrating Workflow**:
    * start on a CUDA platform
    * get a fully working HIP version
    * compile the HIP code on an AMD machine
    * handle platform-specific features through conditional compilation (or by adding them to the open-source HIP infrastructure)
- **Conversion Methods**:
    * **Manual Code Conversion** (search/replace)
    * **HIPIFY Tools** (automated translation tools)
    * **Header Porting** (on the fly translation)

# HIPIFY Tools. Automated Translation Tools
- collection of tools that automatically translate CUDA to HIP code
- **hipify-perl**
   * translates to HIP using pattern matching
   * does not require a working CUDA installation
   * can also convert CUDA code, that is not syntactically correct
- **hipify-clang**
   * uses the Clang compiler CUDA source into an Abstract Syntax Tree (AST)
   * generates the HIP source from the AST
   * needs to be able to compile the code
   * requires a working CUDA installation
   * CUDA code needs to be correct

::: notes
hipify-perl is the simplest tool for converting CUDA code to HIP. It works by scanning a directory and performing basic string replacements, such as converting cudaMemcpy to hipMemcpy. However, since it relies on straightforward text substitution (sed -e 's/cuda/hip/g'), some manual post-processing may be required. It is best suited for quick scans of projects, but it does not handle unrecognized CUDA calls and will report them instead of translating them.

hipify-clang, on the other hand, provides a more robust and accurate translation. It processes the code at a deeper level, generating warnings and offering assistance for further analysis. This tool is particularly useful for high-quality translations, especially when working with projects that involve complex build systems like Make.
::: 

# HIPIFY-perl 

- `hipify-perl –examin <file>.cu` or `hipexamine-perl.sh <file>.cu`
     * basic statistics and number of replacements
     * no replacements
- `hipify-perl <file>.cu`
     * translation a file to standard output
- `hipify-perl -inplace <file>.cu` or `hipconvertinplace-perl.sh <file>.cu`
     * modifies the input file inplace, saves the input file in .prehip file 
     * works with folders:recursively do folders
- `--print-stats` return a report for each file


# Hipify-perl (cont.)
![](img/cublas_cuda_hip.png){ .center width=100% }

# Hipify-perl (cont.)
![](img/kernel_cuda_hip.png){ .center width=100% }


# Hipify-clang

* Build from source
*  Some times needs to include manually the headers -I/...
```bash
$ hipify-clang --print-stats -o matMul.o matMul.c
[HIPIFY] info: file 'matMul.c' statistics:
CONVERTED refs count: 0
UNCONVERTED refs count: 0
CONVERSION %: 0
REPLACED bytes: 0
TOTAL bytes: 4662
CHANGED lines of code: 1
TOTAL lines of code: 155
CODE CHANGED (in bytes) %: 0
CODE CHANGED (in lines) %: 1
20 TIME ELAPSED s: 22.94
```


# Hipify-tools - translating CUDA to HIP

* Hipify-tools can translate CUDA source code into portable HIP C++ automatically
* Although most CUDA expressions are supported, manual intervention may be required
  * For example, a CUDA macro ```__CUDA_ARCH__``` is not translated
    * If the purpose of ```__CUDA_ARCH__``` is to distinguish between host and device code path, it can be replaced with ```__HIP_DEVICE_COMPILE__```
    * If ```__CUDA_ARCH__``` is used to determine architectural feature support, another solution is required, eg, ```__HIP_ARCH_HAS_DOUBLES__```


# Hipify-tools - translating CUDA to HIP

<small>

* To access Hipify-tools on Puhti, do:
  ```
   ml purge; ml gcc/11.3.0 hipify-clang/5.1.0
  ```

* hipify-clang: CUDA -> HIP translator based on LLVM clang
  * Only syntactically correct CUDA code is translated
  * Good support even for somewhat complicated constructs
  * Requires third party dependencies: 
    * 3.8.0 <= clang <= 13.0.1
    * 7.0 <= CUDA <= 11.5.1
  * Usage (-print-stats is optional, but on Puhti, --cuda-path must be specified):
  ```
    hipify-clang -print-stats -o src.cu.hip src.cu --cuda-path=/appl/spack/v018/install-tree/gcc-9.4.0/cuda-11.1.1-lfaa3j
  ```

* hipify-perl: a perl script for CUDA -> HIP translation that mostly uses regular expressions
  * Does not check the input CUDA code for correctness
  * No third party dependencies like clang or CUDA
  * Not as reliable as hipify-clang
  * Usage (-print-stats is optional):
  ```
    hipify-perl -print-stats -o src.cu.hip src.cu
  ```

</small>

# Summary

