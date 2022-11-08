# Monte Carlo simulation with hipRAND library

## Exercise description

The HIP header file [devices_hip.h](src/devices_hip.h) has disappeared from the [src](src/) folder. Fortunately, the respective CUDA header, [devices_cuda.h](src/devices_cuda.h), is still present. The task is to use hipify tools to translate [devices_cuda.h](src/devices_cuda.h) to [devices_hip.h](src/devices_hip.h). What does the hipify tool translate? Is there anything that is not translated properly? You may compare the result with the original HIP header named [solution.h](src/solution.h). Instructions to compile the code with HIP at the bottom. 

IMPORTANT NOTE on hipify-clang module usage on Puhti! Load hipify-clang to hipify CUDA code by 
```
ml hipify-clang
```
and after loading and using hipify-clang, you must do the following before trying to compile any HIP code
```
ml purge
ml hip
```
Otherwise the compilation fails (you cannot compile HIP while having hipify-clang module loaded).
## Code description

This example uses the Monte Carlo method to simulate the value of Bessel's correction that minimizes the root mean squared error in the calculation of the sample standard deviation and variance for the chosen sample and population sizes. The sample standard deviation is typically calculated as $$s = \sqrt{\frac{1}{N - \beta}\sum_{i=1}^{N}(x_i - \bar{x})^2}$$ where $$\beta = 1.$$ The simulation calculates the root mean squared error for different values of $\beta$.

The implementation uses a special construct for the parallel loops in [bessel.cpp](src/bessel.cpp) which is based on a lambda function, an approach similar to some accelerator frameworks such as SYCL, Kokkos, RAJA, etc. The approach allows conditional compilation of the loops for multiple architectures while keeping the source code clean and readable. An example of the usage of cuRAND and hipRAND random number generation libraries inside a GPU kernel are given in [devices_cuda.h](src/devices_cuda.h) and [devices_hip.h](src/devices_hip.h).

The code can be conditionally compiled for either CUDA, HIP, or HOST execution with or without MPI. The correct definitions for each accelerator backend option are selected in [comms.h](src/comms.h) by choosing the respective header file. The compilation instructions are shown below:

```
// Compile to run sequentially on CPU
make

// Compile to run parallel on CPUs with MPI
make MPI=1

// Compile to run parallel on GPU with CUDA
make CUDA=1

// Compile to run parallel on GPU with HIP
make HIP=CUDA

// Compile to run parallel on many GPUs with HIP and MPI
make HIP=CUDA MPI=1

```
