
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

## LUMI

We provide you with access to LUMI system with AMD's MI250X GPUs.

### Login to LUMI

To get started, log in to LUMI:
```shell
ssh -i identity_file username@lumi.csc.fi
```

The username is the CSC account you created before the training.
For more information, refer to the [LUMI documentation](https://docs.lumi-supercomputer.eu/firststeps/).

### Compiling

```shell
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

CC -xhip -o <yourapp> <hip_source.cpp>
```

or with `hipcc`

```shell
module load PrgEnv-amd

export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

hipcc -o <yourapp> <hip_source.cpp>
```

More information on compiling can be found in the [LUMI documentation](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/#compile-hip-code).

### Running

LUMI uses SLURM for batch jobs. Please see [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/)
for more details. If you are using CSC training accounts, you should use the following project as your account: `--account=project_462000877`.

We have also reserved some GPU nodes for the course.
In order to use these dedicated nodes, you need to run your job with the option `--reservation=HIPcourse`, such as

```shell
srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./executable
```

For a multi-gpu application more cards can be requested.

#### Examples

The common part for all of these examples includes: `srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00`

- 1 MPI process(es), 1 GPU(s) per process, 1 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1`
- 1 MPI process(es), 3 GPU(s) per process, 1 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=3`
- 3 MPI process(es), 1 GPU(s) per process, 1 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=3 --cpus-per-task=1 --gpus-per-task=1`
- 3 MPI process(es), 1 GPU(s) per process, 7 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=3 --cpus-per-task=7 --gpus-per-task=1`
- 2 MPI process(es), 3 GPU(s) per process, 7 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=2 --cpus-per-task=7 --gpus-per-task=3`

### HIPFORT on LUMI

The following modules are required:
```bash
module load LUMI/24.0.3
module load partition/G
module load rocm
```
Because the default `HIPFORT` installation only supports gfortran,  we use a custom installation  prepared in the training project.
This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of HIPFFORT with the Fortran Cray Compiler wrapper (ftn).

The package was installed via:
```bash
# In some temporary folder
wget https://github.com/ROCm/hipfort/archive/refs/tags/rocm-6.1.0.tar.gz # one can try various realeases
tar -xvzf rocm-6.1.0.tar.gz;
cd hipfort-rocm-6.1.0;
mkdir build;
cd build;
cmake -DHIPFORT_INSTALL_DIR=/projappl/<project_number>/apps/HIPFORT -DHIPFORT_COMPILER_FLAGS="-ffree -eZ" -DHIPFORT_COMPILER=<path-to>/ftn -DHIPFORT_AR=${CRAY_BINUTILS_BIN_X86_64}/ar -DHIPFORT_RANLIB=${CRAY_BINUTILS_BIN_X86_64}/ranlib  ..
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

## Mahti
In case LUMI is inaccessible during the training, we can use Mahti as a backup.

### Login to Mahti

To get started, log in to Mahti:
```shell
ssh -i identity_file username@mahti.csc.fi
```

The username is the CSC account you created before the training.
For more information, refer to the [CSC documentation](https://docs.csc.fi/computing/#accessing-puhti-and-mahti).

### Compiling

```shell
module purge
module use /appl/spack/v021/summerschool/modules/linux-rhel8-x86_64/Core
module load gcc
module load hip
module load cuda
hipcc "--gpu-architecture=sm_80" --x cu -o <yourapp> <hip_source.cpp>
```

More information on compiling can be found in the [CSC documentation](https://docs.csc.fi/computing/compiling-mahti/#general-instructions).

### Running

Mahti uses SLURM for batch jobs. Please see [CSC documentation](https://docs.csc.fi/computing/running/getting-started/)
for more details. If you are using CSC training accounts, you should use the following project as your account: `--account=project_2013645`.

We have also reserved some GPU nodes for the course.
In order to use these dedicated nodes, you need to run your job with the option `--reservation=HIPcourse`, such as

```shell
srun --reservation=HIPcourse --account=project_2013645 --partition=gpusmall --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 ./executable
```

For a multi-gpu application more cards can be requested.

#### Examples

The common part for all of these examples includes: `srun --reservation=HIPcourse --account=project_2013645 --time=00:05:00`

- 1 MPI process(es), 1 GPU(s) per process, 1 OpenMP thread(s) per process: `--partition=gpusmall --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1`
- 1 MPI process(es), 3 GPU(s) per process, 1 OpenMP thread(s) per process: `--partition=gpusmall --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:3`
- 3 MPI process(es), 1 GPU(s) per process, 1 OpenMP thread(s) per process: `--partition=gpusmall --nodes=1 --ntasks-per-node=3 --cpus-per-task=1 --gres=gpu:a100:3`
- 3 MPI process(es), 1 GPU(s) per process, 7 OpenMP thread(s) per process: `--partition=gpusmall --nodes=1 --ntasks-per-node=3 --cpus-per-task=7 --gres=gpu:a100:3`
- 2 MPI process(es), 4 GPU(s) per process, 7 OpenMP thread(s) per process: `--partition=gpumedium --nodes=2 --ntasks-per-node=1 --cpus-per-task=7 --gres=gpu:a100:8`

More information about the number of GPUs and reserving them can be found in the [CSC documentation](https://docs.csc.fi/computing/running/creating-job-scripts-mahti/#gpu-batch-jobs).

### HIPFORT on Mahti
CSC national systems, Puhti and Mahti, have Nvidia GPUs which use CUDA. However HIP supports also Nvidia architecture. HIPFORT can be installed as well on Nvidia architectures if HIP is setup. 

#### HIPFORT installation
HIPFORT package at the moment is not installed on Mahti. It is straightforward to compile and use it:
- load `hip` module:
```
module load gcc hip
```

- make a directory in your working directory for installing it and get the absolute path, <hipfort_install_folder>.
- clone & compile the package:
```
# In some temporary folder
wget https://github.com/ROCm/hipfort/archive/refs/tags/rocm-6.1.0.tar.gz # one can try various realeases
tar -xvzf rocm-6.1.0.tar.gz;
cd hipfort-rocm-6.1.0;
mkdir build;
cd build;
cmake -DHIPFORT_INSTALL_DIR=/projappl/<project_number>/apps/HIPFORT ..
make -j 32
make install
```

#### Compilation
The `rocm` repository folder `hipfort` contains a set of example (test) codes `.../hipfort/test/f2003`. One can start with the `vecadd` example:

```
export HIPFORT_HOME=/projappl/<project_number>/apps/HIPFORT
hipcc "--gpu-architecture=sm_80" --x cu -c hip_implementation.cpp -o hip_implementation.o
gfortran -cpp -I$HIPFORT_HOME/include/hipfort/nvptx "-DHIPFORT_ARCH=\"nvptx\""  -c main.f03 -o main.o 
hipcc -lgfortran main.o hip_implementation.o  "--gpu-architecture=sm_80" -I$HIPFORT_HOME/include/hipfort/nvptx -L$HIPFORT_HOME/lib/ -lhipfort-nvptx
```
Now the executable `a.out` can be executed as a normal gpu program. 
**Note** HIPFORT provides as well the `hipfc` script which can be used to compilations, though by using this script the linking is always done using fortran and it is more difficult to instegrate with `make` when working with big projects.
