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

To compile, just load the required modules and type `make`. On Mahti, a gpu-aware MPI is
available with:
```
ml openmpi/4.1.4-cuda
```
On LUMI, enable gpu-aware MPI on runtime (and compiling) by eexecuting:
```
MPICH_GPU_SUPPORT_ENABLED=1
```
For running, one should use two GPUs and two MPI processes.

