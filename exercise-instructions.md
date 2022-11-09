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

Please note that the normal GPU partition (`-p gpu`) needs to be used with
the reservation. Otherwise you may use the `gputest` partition for rapid fire
testing.
