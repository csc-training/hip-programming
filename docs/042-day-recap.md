---
title:  Tuesday recap
subtitle: GPU programming with HIP
author:   CSC Training
date:     2026-05
lang:     en
---

# Key ideas from today

* GPU execution is asynchronous by default
* Streams allow independent operations to overlap
* Memory movement between host and device is often a major performance cost
* Pinned memory enables truly asynchronous memory copies
* Unified memory simplifies programming, with the cost of increased page faults
* Profiling traces help reveal synchronization, overlap, and hidden bottlenecks

# How to put this knowledge into use

- The exercises we have presented demonstrate how HIP can be integrated into existing applications
- When working on your own GPU application, consider today's topics:
  - Could or should I execute multiple kernels concurrently?
  - Could I overlap my data transfers with kernel execution?
  - What does my execution timeline look like in a profiler?
  - Could HIP events help me synchronize or time asynchronous operations?

# Tomorrow:

- Tomorrow you will learn about optimization techniques in GPU programming
  - Memory access patterns and data layout
  - Performance-oriented kernel design
  - GPU-accelerated computational libraries
- You will also learn how to develop multi-GPU programs with C++