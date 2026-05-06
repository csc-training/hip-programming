
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

If you have a GitHub account you can also **Fork** this repository and clone your fork, to keep your work in version control throughout the course.

## LUMI

For the duration of the course, we provide you with access to the LUMI system with AMD's MI250X GPUs.

### Login to LUMI

To get started, log in to LUMI:
```shell
ssh -i identity_file username@lumi.csc.fi
```

The username is the CSC account you created before the training.
For more information, refer to the [LUMI documentation](https://docs.lumi-supercomputer.eu/firststeps/).

### Compiling

```shell
module load LUMI/25.03
module load partition/G
module load rocm/6.3.4

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
for more details. If you are using CSC training accounts, you should use the following project as your account: `--account=project_462001376`.

We have also reserved some GPU nodes for the course.
In order to use these dedicated nodes, you need to run your job with the option `--reservation=gpu-course-day1`, such as

```shell
srun --reservation=gpu-course-day1 --account=project_462001376 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./executable
```

On Monday, and using the reservation `gpu-course-day2` on Tuesday and `gpu-course-day3` on Wednesday.

For a multi-gpu application more cards can be requested.

#### Examples

The common part for all of these examples includes: `srun --account=project_462001376 --partition=small-g --time=00:05:00`

- 1 MPI process(es), 1 GPU(s) per process, 1 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1`
- 1 MPI process(es), 3 GPU(s) per process, 1 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=3`
- 3 MPI process(es), 1 GPU(s) per process, 1 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=3 --cpus-per-task=1 --gpus-per-task=1`
- 3 MPI process(es), 1 GPU(s) per process, 7 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=3 --cpus-per-task=7 --gpus-per-task=1`
- 2 MPI process(es), 3 GPU(s) per process, 7 OpenMP thread(s) per process: `--nodes=1 --ntasks-per-node=2 --cpus-per-task=7 --gpus-per-task=3`

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
In order to use these dedicated nodes, you need to run your job with the option `--reservation=gpu-course-day1`, such as

```shell
srun --reservation=gpu-course-day1 --account=project_2018588 --partition=gpusmall --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 ./executable
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
