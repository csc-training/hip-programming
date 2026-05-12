<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Using multiple GPUs with MPI

In this exercise you can practice how to assign different GPUs to different MPI ranks
with a simple "hello" world program.

## Tasks

The provided [hello.cpp](hello.cpp) code is simple pure MPI program that prints out
total number of MPI tasks and the rank of each task.

Expand the code as follows:

1. Create a node local communicator by utilizing `MPI_Comm_split_type` and `MPI_COMM_TYPE_SHARED`.
   Query also node local rank from this communicator
2. Query the number of GPUs in each node
3. Assign a GPU to each rank. In the most common situation single GPU is used by single MPI task,
   but in some cases (*e.g.* single MPI task does not have enough parallelism) it can be useful
   to assign multiple GPUs to same GPU? Can you figure out how to achieve that?

Run the program with various combinations of `--nodes`, `--ntasks-per-node` and `--gpus-per-node`, 
e.g. two nodes with two tasks and two GPUs on each node i.e.
```
...
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
...

srun ./hello
```

## Bonus: limiting GPU visibility with `ROCR_VISIBLE_DEVICES`

The GPU visibility can limited by setting the environment variable `ROCR_VISIBLE_DEVICES`. If
application does not do the MPI task to GPU binding, a wrapper script utilizing 
`ROCR_VISIBLE_DEVICES` together with SLURM environment variables can be used as follows:
```
...
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
...
 
cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

srun ./select_gpu myapp
```



