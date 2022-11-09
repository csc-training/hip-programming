# HIPFORT installation
HIPFORT package at the moment is nto installed on Puhti. It is straightforward to compile and use it:
- load `hip` module:
```
module load gcc hip
```

- make a directory in your working directory for installing it and get the absolute path, <hipfort_install_folder>.
- clone the package:
```
git clone https://github.com/ROCmSoftwarePlatform/hipfort.git
cd hipfort; mkdir build ; cd build
cmake -DHIPFORT_INSTALL_DIR=<install_dir> ..
make install
```
# Compilation
```
hipcc "--gpu-architecture=sm_70" --x cu -c hip_implementation.cpp -o hip_implementation.o
gfortran -cpp -I<hipfort_install_folder>/include/hipfort/nvptx "-DHIPFORT_ARCH=\"nvptx\""  -c main.f03 -o main.o 
hipcc -lgfortran main.o hip_implementation.o  "--gpu-architecture=sm_70" -I<hipfort_install_folder>/include/hipfort/nvptx -L<hipfort_install_folder>/lib/ -lhipfort-nvptx
```
Now the executable `a.out` can be executed as a normal gpu program. 


# Compilation

## AMD

If the Makefile.hipfort had no issues:

```
export HIPFORT_ARCHGPU=amdgcn-gfx908
make
```

However, execute:

```
export PATH=/opt/rocm-4.5.0/hipfort/bin:$PATH
hipfc --offload-arch=gfx908 hipsaxpy.cpp main.f03
```


## NVIDIA

If the Makefile.hipfort had no issues:

```
export HIPFORT_ARCHGPU=nvptx-sm_70
make
```

Use:

```
export PATH=/opt/rocm/hipfort/bin:$PATH
hipfc -x cu --gpu-architecture=sm_70 hipsaxpy.cpp main.f03
```
