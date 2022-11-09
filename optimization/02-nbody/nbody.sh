#!/bin/bash
#SBATCH --job-name=async_serial
#SBATCH --time=00:05:00
#SBATCH --partition=
#sbatch --reservation=


time srun -n 1 ./nbody
