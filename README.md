<!--
SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Introduction to GPU programming

Course material for the CSC course "Introduction to GPU programming". 

Siili document for additional info during the course: https://siili.rahtiapp.fi/9lXWbh-ETaiCuzMuWYmMRw

## Agenda

### Day 1

| Time | Topic |
| ---- | ----- |
| 09:00–09:15 | Welcome, LUMI access, slurm, git, web interface |
| 09:15-09:30 | What is CSC? |
| 09:30–09:45 | Break/debugging access |
| 09:45–10:30 | Introduction to GPU's and GPU programming |
| 10:30–10:45 | Break |
| 10:45-11:30 | Basic usage, memory allocation and kernels |
| 11:30-11:45 | Break |
| 11:45-12:15 | Exercises |
| 12:15-13:00 | Lunch |
| 13:00-13:45 | Basic usage cont. |
| 13:45-14:00 | Break |
| 14:00-14:30 | Exercises |
| 14:30-15:15 | Analyzing simple traces |
| 15:15-15:30 | Break |
| 15:30-16:00 | Exercises |
| 16:00-16:15 | Day summary |


### Day 2

| Time | Topic |
| ---- | ----- |
| 09:00–09:30 | Day 1 Recap |
| 09:30–10:00 | [Streams and events](https://csc-training.github.io/hip-programming/html/03-streams.html) |
| 10:00–10:15 | Break  ||
| 10:15–11:00 | [Exercises](streams/) |
| 11:00-11:30 | [Memory allocations, access, and unified memory](https://csc-training.github.io/hip-programming/html/04-memory.html) |
| 11:30-11:45 | Break |
| 11:45-12:15 | [Exercises](memory/) |
| 12:15-13:00 | Lunch |
| 13:00-13:45 | [Exercises cont'd](memory/) |
| 13:45-14:00 | Break |
| 14:00-14:30 | Inspecting exercise traces, rocprof |
| 14:30-15:00 | Bonus exercise |
| 15:00-16:00 | Close-up |


### Day 3

| Time | Topic |
| ---- | ----- |
| 09:00–10:00 | Performance optimization (I) |
| 10:00–10:15 | Break  |
| 10:15–10:45 | Exercises |
| 10:45-11:30 | Performance optimization (II) |
| 11:30-11:45 | Break |
| 11:45-12:15 | Exercises |
| 12:15-13:00 | Lunch |
| 13:00-13:30 | HOP and performance portability |
| 13:30-13:45 | Break |
| 13:45-14:15 | Exercises  |
| 14:15-14:45 | Multi-GPU programming |
| 14:45-15:45 | Break & Exercises |
| 15:45-16:00 | Close-up | 

## Slides

Link to [slides](https://csc-training.github.io/hip-programming/)

## First steps
- [Which technologies have you used?](https://strawpoll.com/w4nWWYReQnA)
- [First steps](first_steps.md)

## Exercises

[General instructions](exercise-instructions.md)

### Introduction and GPU kernels

- [Three questions at end of lecture](https://csc-training.github.io/hip-programming/html/01-introduction.html#/well-i-have-couple)
- [Hello world](kernels/01-hello-world)
- [Error checking](kernels/02-error-checking)
- [Kernel saxpy](kernels/03-kernel-saxpy)
- [Kernel copy2d](kernels/04-kernel-copy2d)

### Tracing kernels

- [Using rocprofv3](tracing/01-rocprofv3)
- [Tracing saxpy kernel](tracing/02-saxpy)

### Streams, events, and synchronization

- [Implementing and using a stream](streams/01-streams-basics/)
- [Running concurrent GPU kernels](streams/02-streams-asynckernel/)
- [Asynchronous data transfers between CPU and GPU](streams/03-streams-asyncmemcpy/)
- [Recording and timing GPU kernels with HIP events](streams/04-streams-timings/)
- [Overlapping CPU and GPU computation](streams/0X-bonus-cpu-gpu-overlap/)

### Memory allocations, access, and unified memory

- [Explicit and implicit memory management](memory/01-explicit-vs-implicit/)
- [Avoiding CPU-GPU data transfers](memory/02-singlecopy/)
- [The stream-ordered memory allocator and memory pools](memory/03-mempools/s)


### Optimization

- [Using BLAS library](optimization/01-using-blas)
- [Coalesced memory access](optimization/02-coalescing)
- [Using shared memory](optimization/03-matrix_transpose)
- Bonus: [Parallel histogram](optimization/04-histogram)
- Bonus: [Parallel reduction](optimization/05-reductions) 
- Bonus: [Optimizing calculation of pi](optimization/06-bbp-pi)

### Multi-GPU programming and HIP+MPI

- [Ping-pong with multiple GPUs and MPI](multi-gpu/03-mpi)

### Porting to HIP

- [Converting Tools & Portability](porting)

#### Bonus
- [Heat equation with HIP](bonus/heat-equation)
