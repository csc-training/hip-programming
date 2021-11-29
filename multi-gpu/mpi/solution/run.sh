#!/bin/bash -x
#SBATCH --account=Project_2002078
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err

# salloc --account=Project_2002078 --nodes=1 --partition=gputest --gres=gpu:v100:4 --mem-per-cpu=16G --time=00:15:00

srun mpiexample
