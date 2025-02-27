# Generic instructions for the exercises

For most of the exercises, skeleton codes are provided to serve as a starting
point. Some may have sections marked with `// TODO` or `#error` to indicate a place in the code where something is missing or needs to be changed.

In addition, most exercises have example solutions in a `solution`
subdirectory. Note that these are seldom the only or even the best way to
solve the problem.

All of the exercise materials can be downloaded with the command

```shell
git clone https://github.com/csc-training/hip-programming.git
```

If you have a GitHub account you can also **Fork** this repository and clone
then your fork.

### Puhti

We provide you with access to CSC's Puhti system that has NVIDIA's V100 GPUs, but has a working HIP installation to support code porting activities.

To get started with Puhti, you should log in to Puhti and load the appropriate modules to get working with HIP:
```shell
ssh -Y trainingXXX@puhti.csc.fi
module load gcc cuda hip
```

For the November 2022 the `xxx` is `141-164`. Password will be provided on-site. 
For more detailed instructions, please refer to the system documentation at
[Docs CSC](https://docs.csc.fi/).

#### Compiling

In order to compile code with the `hipcc` on Puhti, one needs to add a the target architecture with `--gpu-architecture=sm_70`:
```shell
hipcc hello.cpp -o hello --gpu-architecture=sm_70
```

#### Running

Puhti uses SLURM for batch jobs. Please see [Docs CSC](https://docs.csc.fi/)
for more details. If you are using CSC training accounts, you should use the
following project as your account: `--account=project_2000745`.

We have also reserved some GPU nodes for the course. In order to use these
dedicated nodes, you need to run your job with the option
`--reservation=HIPtraining`, such as

```shell
srun --reservation=HIPtraining -n1 -p gpu --gres=gpu:v100:1 --account=project_2000745 ./my_program
```
For a multi-gpu application more cards can be requested. For example if 3 cards are needed one would use `--gres=gpu:v100:3`.
The number of mpi processes can be as well controled by changing by the `-n` parameter. In order to assign cores for OpenMP theading the parameter `--cpus-per-task`must be set as well.

Please note that the normal GPU partition (`-p gpu`) needs to be used with
the reservation. Otherwise you may use the `gputest` partition for rapid fire
testing.

### HIPFORT on Puhti and Mahti
CSC national systems, Puhti and Mahti, have Nvidia GPUs which use CUDA. However HIP supports also Nvidia architecture. HIPFORT can be installed as well on Nvidia architectures is HIP is setup. 

#### HIPFORT installation
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

#### Compilation
The `rocm` repository folder `hipfort` contains a set of example (test) codes `.../hipfort/test/f2003`. One can start with the `vecadd` example:

```
hipcc "--gpu-architecture=sm_70" --x cu -c hip_implementation.cpp -o hip_implementation.o
gfortran -cpp -I<hipfort_install_folder>/include/hipfort/nvptx "-DHIPFORT_ARCH=\"nvptx\""  -c main.f03 -o main.o 
hipcc -lgfortran main.o hip_implementation.o  "--gpu-architecture=sm_70" -I<hipfort_install_folder>/include/hipfort/nvptx -L<hipfort_install_folder>/lib/ -lhipfort-nvptx
```
Now the executable `a.out` can be executed as a normal gpu program. 

### HIPFORT on LUMI

The following modules are required:
```bash
module load LUMI/24.0.3
module load partition/G
module load rocm
```
Because the default `HIPFORT` installation only supports gfortran,  we use a custom installation  prepared in the training project. This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of HIPFFORT with the Fortran Cray Compiler wrapper (ftn).

The package was installed via:
```bash
# In some temporary folder
wget https://github.com/ROCm/hipfort/archive/refs/tags/rocm-6.1.0.tar.gz # one can try various realeases
tar -xvzf rocm-6.1.0.tar.gz;
cd hipfort-rocm-6.1.0;
mkdir build;
cd build;
cmake -DHIPFORT_INSTALL_DIR=<path-to>/HIPFORT -DHIPFORT_COMPILER_FLAGS="-ffree -eZ" -DHIPFORT_COMPILER=<path-to>/ftn -DHIPFORT_AR=${CRAY_BINUTILS_BIN_X86_64}/ar -DHIPFORT_RANLIB=${CRAY_BINUTILS_BIN_X86_64}/ranlib  ..
make -j 64
make install
```
Where `<path-to>/ftn` can be obtain by running `which ftn`.

We will use the Cray 'ftn' compiler wrapper as you would do to compile any fortran code plus some additional flags:
```bash
export HIPFORT_HOME=/projappl/<project_number>/apps/HIPFORT
ftn -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -c <fortran_code>.f90
CC -xhip -c <hip_kernels>.cpp
ftn  -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -o main <fortran_code>.o hip_kernels.o
```
This option gives enough flexibility for calling HIP libraries from Fortran or for a mix of OpenMP/OpenACC offloading to GPUs and HIP kernels/libraries.
