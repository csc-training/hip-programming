<!--
SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

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
ssh username@lumi.csc.fi
```

The username is the CSC account you created before the training.
For more information, refer to the [LUMI documentation](https://docs.lumi-supercomputer.eu/firststeps/).

### Disk areas

All the exercises should be carried out in the scratch disk area.
This scratch area is shared between all the project members, so create a personal working directory there:

    mkdir -p /scratch/project_462001376/$USER
    cd /scratch/project_462001376/$USER

and clone the course git repository there:

    git clone https://github.com/csc-training/hip-programming.git /scratch/project_462001376/$USER/hip-programming

Now, `/scratch/project_462001376/$USER/hip-programming` is your own clone of the course repository on LUMI
and you can modify files there without causing conflicts with other course participants.



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
srun --reservation=gpu-course-day1 --account=project_462001376 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 ./executable
```

On Monday, and using the reservation `gpu-course-day2` on Tuesday and `gpu-course-day3` on Wednesday.

For a multi-gpu application more cards can be requested.

#### Examples

The common part for all of these examples includes: `srun --account=project_462001376 --partition=small-g --time=00:05:00`

- 1 MPI process(es), 1 GPU(s) per node: `--nodes=1 --ntasks-per-node=1 --gpus-per-node=1`
- 2 MPI process(es), 2 GPU(s) per node: `--nodes=1 --ntasks-per-node=2 --gpus-per-nodes=2`

See the page for example Slurm scripts on LUMI: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/

## Mahti
In case LUMI is inaccessible during the training, we can use Mahti as a backup.

### Login to Mahti

To get started, log in to Mahti:
```shell
ssh username@mahti.csc.fi
```

The username is the CSC account you created before the training.
For more information, refer to the [CSC documentation](https://docs.csc.fi/computing/#accessing-puhti-and-mahti).

### Disk areas

All the exercises should be carried out in the scratch disk area.
This scratch area is shared between all the project members, so create a personal working directory there:

    mkdir -p /scratch/project_2018588/$USER
    cd /scratch/project_2018588/$USER

and clone the course git repository there:

    git clone https://github.com/csc-training/hip-programming.git /scratch/project_2018588/$USER/hip-programming

Now, `/scratch/project_2018588/$USER/hip-programming` is your own clone of the course repository on LUMI
and you can modify files there without causing conflicts with other course participants.

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

The common part for all of these examples includes: `srun --reservation=gpu-course-day1 --account=project_2018588 --time=00:05:00`

- 1 MPI process(es), 1 GPU(s) per node: `--partition=gpusmall --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:1`
- 2 MPI process(es), 2 GPU(s) per node: `--partition=gpusmall --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:2`

More information about the number of GPUs and reserving them can be found in the [CSC documentation](https://docs.csc.fi/computing/running/creating-job-scripts-mahti/#gpu-batch-jobs).
