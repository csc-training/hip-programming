---
title:  Memory allocations, access, and unified memory
subtitle: GPU programming with HIP
author:   CSC Training
date:     2022-11
lang:     en
---


# Outline

* Memory model and hierarchy
* Memory management strategies
* Page-locked memory
* The stream-ordered memory allocator and memory pools

# Memory model

* Host and device have separate physical memories
* It is generally not possible to call malloc() to allocate memory and access
  the data from the GPU
* Memory management can be
    * Explicit (user manages the movement of the data and makes sure CPU and
      GPU pointers are not mixed)
    * Automatic, using Unified Memory (data movement is managed in the
      background by the Unified Memory driver)


# Avoid moving data between CPU and GPU

* Data copies between host and device are relatively slow
* To achieve best performance, the host-device data traffic should be
  minimized regardless of the chosen memory management strategy
    * Initializing arrays on the GPU
    * Rather than just solving a linear equation on a GPU, also setting it up
      on the device
* Not copying data back and forth between CPU and GPU every step or iteration
  can have a large performance impact!


# Device memory hierarchy

<div class="column">
- Registers (per-thread-access)
- Local memory (per-thread-access)
- Shared memory (per-block-access)
- Global memory (global access)
</div>

<div class="column">
![](img/memlayout.png){width=80%}
</div>


# Device memory hierarchy

<div class="column">
- Registers (per-thread-access)
    - Used automatically
    - Size on the order of kilobytes
    - Very fast access
- Local memory (per-thread-access)
    - Used automatically if all registers are reserved
    - Local memory resides in global memory
    - Very slow access
</div>

<div class="column">
- Shared memory (per-block-access)
    - Usage must be explicitly programmed
    - Size on the order of kilobytes
    - Fast access
- Global memory (per-device-access)
    - Managed by the host through HIP API
    - Size on the order of gigabytes
    - Very slow access
</div>


# Device memory hierarchy (advanced)

- There are more details in the memory hierarchy, some of which are
  architecture-dependent, eg,
    - Texture memory
    - Constant memory
- Complicates implementation
- Should be considered only when a very high level of optimization is
  desirable


# Important memory operations

Allocate pinned device memory
```cpp
hipError_t hipMalloc(void **devPtr, size_t size)
```
Allocate Unified Memory; the data is moved automatically between host/device
```cpp
hipError_t hipMallocManaged(void **devPtr, size_t size)
```
Deallocate pinned device memory and Unified Memory
```cpp
hipError_t hipFree(void *devPtr)
```
Copy data (host-host, host-device, device-host, device-device)
```cpp
hipError_t hipMemcpy(void *dst, const void *src, size_t count, enum hipMemcpyKind kind)
```

# Memory management strategies
<small>
<div class="column">

* Example of explicit memory management
```cpp
int main() {
 int *A, *d_A;
 A = (int *) malloc(N*sizeof(int));
 hipMalloc((void**)&d_A, N*sizeof(int));
 ...
 /* Copy data to GPU and launch kernel */
 hipMemcpy(d_A, A, N*sizeof(int), hipMemcpyHostToDevice);
 kernel<<<...>>>(d_A);
 hipMemcpy(A, d_A, N*sizeof(int), hipMemcpyDeviceToHost);
 hipFree(d_A);
 ...
 printf("A[0]: %d\n", A[0]);
 free(A);
 return 0;
}
```
</div>

<div class="column">

* Example of Unified Memory
```cpp
int main() {
 int *A;
 hipMallocManaged((void**)&A, N*sizeof(int));
 ...
 /* Launch GPU kernel */
 kernel<<<...>>>(A);
 hipStreamSynchronize(0);
 ...
 printf("A[0]: %d\n", A[0]);
 hipFree(A);
 return 0;
}
```

</div>
</small>

# Unified Memory pros

- Allows incremental development
- Can increase developer productivity significantly
    - Especially large codebases with complex data structures
- Supported by the latest NVIDIA + AMD architectures
- Allows oversubscribing GPU memory on some architectures

# Unified Memory cons

- Data transfers between host and device are initially slower, but can be
  optimized once the code works
    - Through prefetches
    - Through hints
- Must still obey concurrency & coherency rules, not foolproof
- Although the performance on AMD cards is quite good, there may be issues with prefetching and hints (with AMD)


# Unified Memory workflow for GPU offloading

1. Allocate memory for the arrays accessed by the GPU with
   `hipMallocManaged()` instead of `malloc()`
    - It is a good idea to have a wrapper function and use conditional compilation
      for memory allocations
2. Offload compute kernels to GPUs
3. Check profiler backtrace for GPU->CPU Unified Memory page-faults (NVIDIA
   Visual Profiler, Nsight Systems, AMD profiler?)
    - This indicates where the data residing on the GPU is accessed by the CPU
      (very useful for large codebases, especially if the developer is new to
      the code)


# Unified Memory workflow for GPU offloading

4.  Move operations from CPU to GPU if possible, or use hints / prefetching
    (`hipMemAdvice()` / `hipMemPrefetchAsync()`)
    -  It is not necessary to eliminate all page faults, but eliminating the
       most frequently occurring ones can provide significant performance
       improvements
5.  Allocating GPU memory can have a much higher overhead than allocating
    standard host memory
    - If GPU memory is allocated and deallocated in a loop, consider using a
      GPU memory pool allocator for better performance (eg Umpire)


# Virtual Memory addressing

<div class="column">
- Modern operating systems utilize virtual memory
    - Memory is organized to memory pages
    - Memory pages can reside on swap area on the disk (or on the GPU with
      Unified Memory)
</div>

<div class="column">
![](img/virtual_memory_addressing.png){width=50%}
</div>

# Page-locked (or pinned) memory

- Normal `malloc()` allows swapping and page faults
- User can page-lock an allocated memory block to a particular physical memory
  location
- Enables Direct Memory Access (DMA)
- Higher transfer speeds between host and device
- Copying can be interleaved with kernel execution
- Page-locking too much memory can degrade system performance due to paging
  problems

# Allocating page-locked memory on host

- Allocated with `hipMallocHost()` or `hipHostAlloc()` functions instead of `malloc()`
- The allocation can be mapped to the device address space for device access
  (slow)
    - On some architectures, the host pointer to device-mapped allocation can
      be directly used in device code (ie, it works similarly to Unified
      Memory pointer, but the access from the device is slow)
- Deallocated using `hipFreeHost()`

# Asynchronous memcopies

- Normal `hipMemcpy()` calls are blocking (ie, synchronizing)
    - The execution of host code is blocked until copying is finished
- To overlap copying and program execution, asynchronous functions are required
    - Such functions have Async suffix, eg, `hipMemcpyAsync()`
- User has to synchronize the program execution
- Asynchronous memory copies require page-locked memory

# The stream-ordered memory allocator and memory pools
<small>

* Obtain unused memory already allocated from the device's current memory pool in the specified stream (if not enough memory is available, more memory is allocated for the pool)
```cpp
​hipError_t hipMallocAsync ( void** devPtr, size_t size, hipStream_t hStream )
```

* Return memory to the pool in the specific stream (does not deallocate memory)
```cpp
hipError_t hipFreeAsync ( void* devPtr, hipStream_t hStream ) 
```

* By default, the pool is deallocated completely when the stream is synchronized

* The `hipMemPoolAttrReleaseThreshold` represents the size down to which the pool is deallocated during a synchronization, and can be set by 

```cpp
hipMemPool_t mempool;
hipDeviceGetDefaultMemPool(&mempool, device);
uint64_t threshold = UINT64_MAX;
hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &threshold);
```

* Setting threshold to `UINT64_MAX` means, that the pool is practically never deallocated due to synchronization (because the threshold is a huge number)

</small>

# Memory pools - Example
<small>

<div class="column">
* Example 1 - slow
```cpp
for (int i = 0; i < 100; i++) {
  // Allocate memory here (slow)
  hipMalloc(&ptr, size); 
  // Run GPU kernel
  kernel<<<..., stream>>>(ptr);
  // Deallocate memory here
  hipFree(ptr); 
}
// Synchronize the default stream (no influence to memory allocations)
hipStreamSynchronize(0); 
```
* Allocating and deallocating memory in a loop is slow, and can have a significant impact on the performance
</div>
<div class="column">
* Example 2 - fast
```cpp
for (int i = 0; i < 100; i++) {
  // Obtain unused memory from the current memory pool, 
  // more memory is allocated for the pool if needed
  hipMallocAsync(&ptr, size, stream); 
  // Run GPU kernel
  kernel<<<..., stream>>>(ptr);
  // Return memory to the current memory pool
  hipFreeAsync(ptr, stream); 
}
// Synchronize and deallocate all memory from the 
// current memory pool (default behavior)
hipStreamSynchronize(stream); 
```
* Recurring memory allocation and deallocation does not occur anymore, because the memory is obtained from the memory pool and only deallocated during the synchronization (default behavior)

</div>
</small>

# Summary

- Host and device have separate physical memories
- Memory management can be explicit (managed by the user) or automatic (managed by the Unified Memory driver)
- Using Unified Memory can improve developer productivity and result in a
  cleaner implementation
- The number of data copies between CPU and GPU should be minimized
    - With Unified Memory, if data transfer cannot be avoided, using hints or
      prefetching to mitigate page faults is beneficial
- Recurring allocation and deallocation is slow, use memory pools instead 
  - Libraries provide pooled Unified Memory support as well (eg, Umpire)

