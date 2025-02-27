# GPU programming with HIP

Course material for the CSC course "GPU programming with HIP". The course is
part of the EuroCC training activities at CSC.

## Agenda (WIP)

### Day 1

| Time | Topic |
| ---- | ----- |
| 09:00–10:30 | Welcome (lumi/mahti access, slurm etc) & Introduction to GPU programming w/ break (JL) (timing: WIP) |
| 10:30–10:45 | Break & Snacks |
| 10:45-11:30 | HIP and GPU kernels (JL) |
| 11:30-11:45 | Break |
| 11:45-12:15 | Exercises |
| 12:15-13:00 | Lunch |
| 13:00-13:45 | Streams, events, and synchronization (JK) |
| 13:45-14:00 | Break |
| 14:00-14:30 | Exercises |
| 14:30-15:15 | Memory allocations, access and unified memory (JK) |
| 15:15-15:30 | Break |
| 15:30-16:00 | Exercises |
| 16:00-16:15 | Day summary |


### Day 2

| Time | Topic |
| ---- | ----- |
| 09:00–10:00 | Kernel optimizations (JK) |
| 10:00–10:15 | Break & Snacks |
| 10:15–10:45 | Exercises |
| 10:45-11:30 | Multi-GPU programming, HIP+MPI (CA) |
| 11:30-11:45 | Break |
| 11:45-12:15 | Exercises |
| 12:15-13:00 | Lunch |
| 13:00-13:30 | Fortran and HIP (CA) |
| 13:30-13:45 | Break |
| 13:45-14:15 | Exercises  |
| 13:30-14:15 | Applications portability (CA) |
| 14:15-14:30 | Break |
| 14:30-15:45 | Exercises |
| 15:45-16:00 | Close-up | 


Link to [slides](https://csc-training.github.io/hip-programming/)
## Exercises

[General instructions](exercise-instructions.md)

### Introduction and GPU kernels

- [Hello world](kernels/01-hello-world)
- [Error checking](kernels/02-error-checking)
- [Kernel saxpy](kernels/03-kernel-saxpy)
- [Kernel copy2d](kernels/04-kernel-copy2d)

### Streams, events, and synchronization

- [Understanding asynchronity using events](streams/01-event-record)
- [Investigating streams and events](streams/02-concurrency)

### Memory allocations, access, and unified memory

- [Memory management strategies](memory/01-prefetch)
- [The stream-ordered memory allocator and memory pools](memory/02-mempools)
- [Unified memory and structs](memory/03-struct)

### Fortran and HIP

- [SAXPY](hipfort/saxpy/hip/)
- [HIPRAND](hipfort/hiprand/hip/)

### Optimization

- [Matrix Transpose](optimization/01-matrix_transpose)

### Multi-GPU programming and HIP+MPI

- [Peer to peer device access](multi-gpu/01-p2pcopy)
- [Vector sum on two GPUs without MPI](multi-gpu/02-vector-sum)
- [Ping-pong with multiple GPUs and MPI](multi-gpu/03-mpi)

### Code design, conditional compilation, lambdas, hipify

- [Host-device lambda functions and general kernels](lambdas/01-lambda)
- [Reductions with host-device lambdas and hipCUB](lambdas/02-reduction)
- [Monte Carlo simulation with hipRAND library](lambdas/03-hipify)

### Bonus
- [Heat equation with HIP](bonus/heat-equation)
