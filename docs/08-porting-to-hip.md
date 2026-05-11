---
title:    GPU performance portability
event:    Introduction to GPU programming
date:     May 2026
lang:     en
---

# (Performance) portability in the GPU age

- CPUs are relatively easy, GPUs quite a bit harder
- How to support different GPU architectures?
<div style="border:3px solid var(--csc-blue); border-radius: 15px; margin-top:1ex">
  1. use accelerated GPU libraries: cublas, rocsolver, ...
  2. use a high-level abstraction layer
     - directive based methods: OpenMP, OpenACC
     - programming models: SYCL, Kokkos, Raja, ...
  3. use native GPU programming
     - CUDA, HIP
</div>
  - most approaches require support for multiple backends in a code
    (CUDA+HIP, cublas+hipblas, ...)

# Porting a CUDA code to HIP

- Manual code conversion (search/replace or incremental)
- HIPIFY tools (automated translation tools)
    - `hipify-perl` : Perl script for `cuda`->`hip` search-and-replace for API calls
    - `hipify-clang` : Generation of HIP code via clang 
        - requires working CUDA installation
- Might require maintaining two code bases for NVIDIA and AMD

# HOP: Header Only Porting

- Header Only Porting (HOP) is a light-weight, header-only library that
  enables <span class=text-blue>*automatic, bidirectional translation between
  CUDA and HIP*</span> at compile
  time
  - for C and C++ codes (also Fortran with ISO C bindings)
  - no code modifications needed
  - just add a few extra flags at compile time to hop from CUDA to HIP or
    back

- Leverages the almost one-to-one mapping between CUDA and HIP
  - catches include statements
  - redefines identifiers
- Generic IDs (`gpuMalloc` etc.) offer a way to write vendor-agnostic code
  that is trivially translated to CUDA or HIP at compile time (cf. GPAW)


# HOP: how does it work?

- Redefines identifiers using preprocessor directives<br>
  <span style="padding-left:1.5em">`cudaMalloc ⇔ hipMalloc`</span>
  <span style="font-size:0.8em; padding-left:1.5em">etc.</span>
- Catches include statements by providing alternative header files that
  take precedence over the original ones
  - source identifiers are redefined to target identifiers
  - target GPU backend needs to be defined (CUDA or HIP)
  - e.g. `"#include <hip/hip_runtime.h>"` will actually load a HOP header file
    that does a translation from HIP identifiers to CUDA identifiers


# Example: redefine identifiers

<div class="column" style="width:55%">
<div style="font-size:0.9em">
<span style="font-size:0.9em">translate from source (HIP):</span>
```c
#define hipMalloc                 gpuMalloc
#define hipMallocAsync            gpuMallocAsync
#define hipHostMalloc             gpuHostMalloc
#define hipHostMallocPortable     gpuHostMallocPortable
#define hipMemcpy                 gpuMemcpy
```

<span style="font-size:0.9em">translate to target (CUDA):</span>
```c
#include <cuda_runtime_api.h>

#define gpuMalloc                 cudaMalloc
#define gpuMallocAsync            cudaMallocAsync
#define gpuHostMalloc             cudaHostAlloc
#define gpuHostMallocPortable     cudaHostAllocPortable
#define gpuMemcpy                 cudaMemcpy
```
</div>
</div>

<div class="column" style="width:40%; text-align:center">
<br>
<br>
<br>
<code>hipMalloc</code>
<br>
⇓
<br>
<code>cudaMalloc</code>
</div>


# HOP: compile flags

`-I$HOP_ROOT`
  : include HOP headers

`-I$HOP_ROOT/source/cuda &nbsp;&nbsp;OR&nbsp;&nbsp; -I$HOP_ROOT/source/hip`
  : catch source code header includes

`-DHOP_TARGET_HIP` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="font-weight:normal">&nbsp;&nbsp;OR&nbsp;&nbsp;</span> `-DHOP_TARGET_CUDA`<span style="font-weight:normal">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(optional)</span>
  : define target for translation

<br>
where **`$HOP_ROOT`** points to the installation path of HOP:<br>
&nbsp;&nbsp;`export HOP_ROOT=/path/to/hop`


# Example: compile and run

## CUDA ⇒ HIP

```
export HOP_ROOT=/path/to/hop
export HOP_FLAGS=-I$HOP_ROOT -I$HOP_ROOT/source/cuda -DHOP_TARGET_HIP
CC -x hip $HOP_FLAGS hello.cu -o hello
./hello
```
<br>

## HIP ⇒ CUDA

```
export HOP_ROOT=/path/to/hop
export HOP_FLAGS=-I$HOP_ROOT -I$HOP_ROOT/source/hip -DHOP_TARGET_CUDA
CC -x cu $HOP_FLAGS hello.cpp -o hello
./hello
```


# HOP in code development

- HOP uses generic identifiers as intermediates in the translation
  - `gpuMalloc`, `gpuMemcpyHostToDevice`, ...
- One can use these generic identifiers directly in code
  - no CUDA/HIP identifiers, just generic identifiers that are then mapped to
    the correct target identifiers
- HOP headers are named and organised similar to HIP headers
  - if code uses only generic identifiers and includes the appropriate HOP
    headers, no need for `-I$HOP_ROOT/source/..`
- HOP headers may also be embedded in end-user code
  - MIT license


# Header Only Porting as a general approach

- Use generic identifiers (`gpuMalloc` ...)
  - easy to swap between GPU backends (single header change)
  - allows one to also implement more complex wrapper functions if and when
    needed
- Strong preference for features that are supported by both CUDA and HIP
  - if needed, wrapper functions can be used to write backend-specific
    implementations
- Use standard compliant C/C++
  - avoid implicit header includes (`nvcc`, we are looking at you!)
  - kernel launch with `<<<...>>>()` works, but better to use
    `gpuLaunchKernel()` that can be mapped to whatever is needed
    by the target GPU backend


# HOP: benefits and drawbacks

<div class="column" style="width:60%">
## Pros:

- Easy porting between CUDA and HIP
  - no code modifications
  - works also from HIP to CUDA!
- No code duplication
  - one can use generic identifiers, HIP, or CUDA
- Flexible and simple
  - transparent one-to-one mappings
  - trivial to add hardware specific implementations if and when needed
</div>

<div class="column" style="width:38%">
## Cons:
- Mapping limited to features supported by both HIP and CUDA
- Not aimed at other GPU backends
</div>


# HOP: how to get started?

- Code available at: &nbsp; [https://github.com/cschpc/hop](https://github.com/cschpc/hop)
  - working proof of concept implementation
  - most runtime identifiers included
  - rudimentary support for BLAS, FFT, RAND, and SPARSE libraries
  - no automatic testing at the moment, only commonly used IDs tested
- Future outlook:
  - fix open issues (IDs with mismatching arguments, C++ overloading, ...)
  - add testing
  - add support for other libraries
  - better documentation :)


# Summary

- Various ways to port code from CUDA to HIP
- HIPIFY tools can automatically convert code to HIP
- Header Only Porting (HOP)  bidirectional translation between CUDA and HIP at compile time
