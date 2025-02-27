# Hipfort

## Usage 

Test hipfort by compiling and running a simple Fortran code that uses a HIP kernel to calculate saxpy on the GPU.

The following modules are required:
```bash


module load LUMI/24.03
module load partition/G 
module load rocm
```
We use a custom installation  prepared in the training project. This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of hipfort with the Fortran Cray Compiler wrapper (ftn). 

We will use the Cray 'ftn' compiler wrapper as you would do to compile any fortran code plus some additional flags:
```bash
export HIPFORT_HOME=/projappl/<project_number>/apps/HIPFORT
ftn -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -c <fortran_code>.f90 
CC -xhip -c <hip_kernels>.cpp
ftn  -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -o main <fortran_code>.o hip_kernels.o
```
This option gives enough flexibility for calling HIP libraries from Fortran or for a mix of OpenMP/OpenACC offloading to GPUs and HIP kernels/libraries.

## Computing `pi` 


Hipfort provides interfaces for various highly optimized library. 

The folder  [hiprand_example](hiprand_example/) shows how to call the `hiprand` for generating single precision uniform random distributed nubmbers for calculation the value of `pi`.

The exercise is to analyse and run the programs. For more examples of hipfort check also the [official repository](https://github.com/ROCmSoftwarePlatform/hipfort/tree/develop/test).
