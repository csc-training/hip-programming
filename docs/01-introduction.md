---
title:  Introduction to GPUs and GPU programming
author: CSC Training
date:   2021-11
lang:   en
---

# High-performance computing

<div class="column">
- High performance computing is fueled by ever increasing performance
- Increasing performance allows breakthroughs in many major challenges that
  humankind faces today
- Not only hardware performance, algorithmic improvements have also added
  ordered of magnitude of real performance
</div>

<div class="column">
![](img/top500-performance.png)
</div>


# HPC through the ages

<div class="column">
- Achieving performance has been based on various strategies throughout the
  years
- Frequency, vectorization, multinode, multicore ...
- Now performance is mostly limited by power consumption
- Accelerators provide compute resources based on a very high level of
  parallelism to reach high performance at low relative power consumption
</div>

<div class="column">
![](img/microprocessor-trend-data.png)
</div>


# Accelerators

- Specialized parallel hardware for floating point operations
    - Co-processors for traditional CPUs
    - Based on highly parallel architectures
    - Graphics processing units (GPU) have been the most common accelerators
      during the last few years
- Promises
    - Very high performance per node
- Usually major rewrites of programs required


# Accelerator model today

<div class="column">
- Connected to CPUs via PCIe
- Local memory
    - Smaller than main memory (32 GB in Puhti)
- Very high bandwidth (up to 900 GB/s)
- Latency high compared to compute performance
- Data must be copied over the PCIe bus
</div>

<div class="column">
![](img/gpu-cluster.png){}
![](img/gpu-bws.png){width=100%}
</div>


# GPU architecture

- Designed for running tens of thousands of threads simultaneously on
  thousands of cores
- Very small penalty for switching threads
- Running large amounts of threads hides memory access penalties
- Very expensive to synchronize all threads


# Challenges in using Accelerators

**Applicability**: Is your algorithm suitable for GPU?

**Programmability**: Is the programming effort acceptable?

**Portability**: Rapidly evolving ecosystem and incompatibilities between
vendors.

**Availability**: Can you access a (large scale) system with GPUs?

**Scalability**: Can you scale the GPU software efficiently to several nodes?


# Using GPUs

<div class="column">
1. Use existing GPU applications
2. Use accelerated libraries
3. Directive based methods
    - OpenMP
    - **OpenACC**
4. Use lower level language
    - CUDA
    - HIP
    - OpenCL
</div>

<div class="column">
Easier, but more limited

![](img/arrow.png){ width=15% }

More difficult, but more opportunities
</div>


# Directive-based accelerator languages

- Annotate code to mark accelerator-offloadable regions
- OpenACC
    - focus on optimizing productivity (reasonably good performance with
      minimal effort)
    - created in 2011, latest version is 3.1 (November 2020)
    - mostly Nvidia only
- OpenMP
    - de-facto standard for shared-memory parallelisation
    - initial support for accelerators in 4.0 (2013)
    - significant improvements/extensions in 4.5 (2015), 5.0 (2018),
      and 5.1 (2020)
