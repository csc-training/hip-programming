#!/bin/bash
#SBATCH --job-name=async_serial
#SBATCH --time=00:05:00
#SBATCH --partition=MI100
#SBATCH --nodes=1

time srun -n 1 ./nbody
