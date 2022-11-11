# HIPFORT installation
HIPFORT package at the moment is not installed on Puhti. It is straightforward to compile and use it:
- load `hip` module:
```
module load gcc hip
```

- make a directory in your working directory for installing it and get the absolute path, <hipfort_install_folder>.
- clone & compile the package:
```
git clone https://github.com/ROCmSoftwarePlatform/hipfort.git
cd hipfort; mkdir build ; cd build
cmake -DHIPFORT_INSTALL_DIR=<hipfort_install_folder> ..
make install
```
For this training try  replacing `<hipfort_install_folder>` with  `/scratch/project_2000745/training160/hip-programming/hipfort/hipfort/build/hipfort_install` 
# Compilation
The `rocm` repository folder `hipfort` contains a set of example (test) codes `.../hipfort/test/f2003`. One can start with the `vecadd` example:

```
hipcc "--gpu-architecture=sm_70" --x cu -c hip_implementation.cpp -o hip_implementation.o
gfortran -cpp -I<hipfort_install_folder>/include/hipfort/nvptx "-DHIPFORT_ARCH=\"nvptx\""  -c main.f03 -o main.o 
hipcc -lgfortran main.o hip_implementation.o  "--gpu-architecture=sm_70" -I<hipfort_install_folder>/include/hipfort/nvptx -L<hipfort_install_folder>/lib/ -lhipfort-nvptx
```
Now the executable `a.out` can be executed as a normal gpu program. 

# Exercise
Apply the above procedure to the `saxpy`  code in the present folder  and inspect the codes in the `rocm` repository folder `hipfort` containing the example (test) codes `.../hipfort/test/f2003`. See how the memory management (allocations and transfers) are done and how  various `hipxxx` libraries are called in `Fortran` programs.
