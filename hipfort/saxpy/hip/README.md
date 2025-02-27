# SAXPY using FORTRAN & HIPFORT 

Inspect `saxpy`  code in the present folder. The Fortran code folows the same logic as the HIP C code. 
First the data is created on the cpu. Then the memory is allocated on the GPU and the data is transfered from CPU to GPU. When the transfer is completed a kernel is executed to perform the work.  In the end the results of the computation is copied to the CPU and processed further. 

**Note** Fortran does can not compile HIP  C code.  The GPU code is located in a separate file, [hipsaxpy.cpp](hipsaxpy.cpp). The HIP kernel is launched via C function which acts as a wrapper. Fortran calls this C wrapper using  `iso_c_binding` module.

In this code all calls to HIP API are done via HIPFORT. The exercise is to check and familiarize with how the memory management (allocations and transfers) is done and how Fortran is calling C functions using `iso_c_binding` module. 
If you have previous experience with CUDA Fortran you can compare it to the equivalent code in the [cuda](../cuda) folder.

In addition to the memory management, HIPFORT provides also  bindings for the mathematical libraries running on GPUs. You can find examples of how various `hipxxx` & `rocxxx` libraries are called in `Fortran` programs in the [HIPFORT repository](https://github.com/ROCm/hipfort/tree/develop/test).
