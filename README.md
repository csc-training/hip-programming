# GPU programming with HIP

Course material for the CSC course "GPU programming with HIP". The course is
part of the PRACE Training Center (PTC) activities at CSC.

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
- [The stream-ordered memory allocator and memory pools](streams/02-mempools)
- [Unified memory and structs](memory/03-struct)

### Fortran and HIP

- [Hipfort exercise](hipfort)

### Optimization

- [Matrix Transpose](optimization/01-matrix_transpose)
- [Nbody](optimization/02-nbody)

### Multi-GPU programming and HIP+MPI

- [Vector sum on two GPUs without MPI](multi-gpu/02-vecotr-sum)
- [Peer to peer device access](multi-gpu/01-p2pcopy)
- [Ping-pong with multiple GPUs and MPI](multi-gpu/03-mpi)

### Bonus
- [Heat equation with HIP](bonus/heat-equation)
