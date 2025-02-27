# SAXPY using FORTRAN & HIPFORT 

Inspect `saxpy`  code in the present folder. Check how the memory management (allocations and transfers) are done and how the fortran is calling C functions using ìso_c_binding` module. 
If you have previous experience with CUDA Fortran you can compare it to the equivalent code in the [cuda](../cuda) folder.

In addition to the memory management, HIPFORT provides also  bindings for the mathematical libraries running on GPUs. You can find examples of how various `hipxxx` & `rocxxx` libraries are called in `Fortran` programs in the [HIPFORT repository](https://github.com/ROCm/hipfort/tree/develop/test).
