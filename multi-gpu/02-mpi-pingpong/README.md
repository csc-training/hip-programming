# Ping-pong with multiple GPUs and MPI

Implement a simple ping-pong test for GPU-to-GPU communication using:
a) indirect communication via the host, and b) direct communication with
HIP-aware MPI.

The ping-pong test consists of the following steps:
  1. Send a vector from one GPU to another
  2. The receiving GPU should increment all elements of the vector by one
  3. Send the vector back to the original GPU

For reference, there is also a CPU-to-CPU implementation in the skeleton
code ([ping-pong.cpp](ping-pong.cpp)). Timing of all tests is also included to
compare the execution times.

On **Lumi**, one can compile the MPI example simply using the Cray compiler with
```
CC -xhip ping-pong.cpp
```

On LUMI, enable gpu-aware MPI on runtime (and compiling) by eexecuting:
```
MPICH_GPU_SUPPORT_ENABLED=1
```
For running, one should use two GPUs and two MPI processes:

```
srun --account=XXXXXX --partition=small-g -N1 -tasks-per-node=2 --cpus-per-task=1 --gpus-per-node=2 --time=00:15:00 ./a.out #  # The reservation is for small-g partition
```


On **Mahti**, to compile, just load the required modules and type `make`. A gpu-aware MPI is
available with:
```
ml openmpi/4.1.4-cuda
```
For running, one should use two GPUs and two MPI processes:
```
srun --account=XXXXXX --partition=gputest -N1 -n2 --cpus-per-task=1 --gres=gpu:v100:2 --time=00:15:00 ./a.out
```
