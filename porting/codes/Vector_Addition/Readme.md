# Vector addition

This is simple vector addition for exemplify the [CUDA to HIP conversion]. The code executes `C[i]=A[i]+B[i]`, for `i=1,...,N`. 

Compile CUDA code: nvcc -arch=sm_70 vecadd.cu -o vecadd
