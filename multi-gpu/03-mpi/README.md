# Ping-pong with multiple GPUs and MPI

Implement a simple ping-pong test for GPU-to-GPU communication using:
a) indirect communication via the host, and b) direct communication with
HIP-aware MPI.

The ping-pong test constists of the following steps:
  1. Send a vector from one GPU to another
  2. The receiving GPU should increment all elements of the vector by one
  3. Send the vector back to the original GPU

For reference, there is also a CPU-to-CPU implementation in the skeleton
code ([ping-pong.cpp](ping-pong.cpp)). Timing of all tests is also included to
compare the execution times.

To compile, just load the required modules and type `make`. On Puhti, a HIP-aware MPI is
available with:
```
ml openmpi/4.1.4-cuda
```
For running, one should use two GPUs and two MPI processes.


# Notes from testing on Mahti

Compile and run:

```bash
module load cuda/11.5.0
module load openmpi/4.1.2-cuda

export SINGULARITY_BIND="/scratch,/projappl,/appl"
singularity exec -B /local_scratch,/lib64/libhwloc.so.15,/usr/lib64/libevent_core-2.1.so.6,/usr/lib64/libevent_pthreads-2.1.so.6 .../cuda_hip_0.1.0.sif hipcc -I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include -L/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -lmpi ping-pong.cpp

srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 0:15:00 ./a.out
```

Output:
```
MPI rank 0: Found 2 GPU devices, using GPU 0
CPU-CPU: time 5.300000e-06, errorsum 0.000000
GPU-GPU direct: time 3.096000e-05, errorsum 0.000000
GPU-GPU via host: time 4.849000e-05, errorsum 0.000000
MPI rank 1: Found 2 GPU devices, using GPU 1
```
