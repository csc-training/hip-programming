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

# Memory model and hierarchy

::::::{.columns}
:::{.column width="60%"}
::: {.fragment}
- Registers (*VGPR, SGPR*)
  - Compiler assigns automatically
:::
::: {.fragment}
- Shared memory (*Local data share, LDS*)
  - User controlled
  - Shared by threads in a block
:::
::: {.fragment}
- Local memory (*Scratch*)
  - Automatically used when registers run out
:::
::: {.fragment}
- Global device memory
:::
::: {.fragment}
- Host memory
:::
:::
::: {.column width="40%"}
![](img/memory-hierarchy.png){width=100%}
:::
::::::

::: {.notes}
Extra: 
- Not covered: texture memory, constant memory
- Could be considered only when lower hanging optimisation tricks are covered
- This lecture: host ⇄ global device memory
:::



# Memory management strategies

Memory management can be *Explicit* or *Implicit*.

:::{.incremental}
- *Explicit*: User manually manages data movement between host and device. Memory can be allocated with GPU-unaware allocators (`malloc`/`free` etc).
- *Implicit*: The runtime manages data movement between host and device. Memory needs to be allocated with special allocated offered by HIP api.
  - **Pinned** (page-locked) host allocations: Data moves to device with kernel invocations and is not stored there.
  - **Unified memory** (Managed memory): Page faults will initiate data movement.
:::

* [HIP API documentation on memory](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/doxygen/html/group___memory_m.html)

# Memory management strategies

::::::{.columns}
:::{.column}

:::{.fragment}
<small>
Explicit memory management
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
 // result is in A
 free(A);
}
```
</small>
:::
:::
:::{.column}

:::{.fragment}
<small>
Unified Memory management
```cpp
int main() {
 int *A;
 hipMallocManaged((void**)&A, N*sizeof(int));
 ...
 /* Launch GPU kernel */
 kernel<<<...>>>(A);
 hipStreamSynchronize(0);
 // result is in A
 hipFree(A);
}
```
:::

:::{.fragment}
Host-pinned, automatic copy
```cpp
int main() {
  int *A;
  hipHostMalloc((void**) &A, N*sizeof(int));
  kernel<<<...>>>(A)
  hipStreamSynchronize(0);
  // result is in A
  hipHostFree(A);
}
```
:::

</small>

:::
::::::


# Unified Memory pros & cons

::::::{.columns}
:::{.column}
**Pros**

- Incremental development
- Increased developer productivity
  - Especially large codebases with complex data structures
- Allows oversubscribing GPU memory on some architectures
- Data transfer can be optimized later
  - With prefetches and hints

:::
:::{.column}
**Cons**

- Data transfers between host and device are initially slower <br>⇒ Must be optimized with prefetches and hints
- Externalize memory management to library

:::
::::::

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
      GPU memory pool allocator for better performance (e.g. Umpire)


# Side-topic: Virtual Memory addressing

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
- Copying can be interleaved with kernel execution (??)
- Page-locking too much memory can degrade system performance due to paging
  problems

# Allocating page-locked memory on host

- Allocated with `hipHostMalloc()` functions instead of `malloc()`
- Maps the allocated host memory to address space of all available GPUs
- Memory can be accessed from GPU but the access is over host-device link (slow)
- Deallocated using `hipFreeHost()`

# Asynchronous memcopies

- Normal `hipMemcpy()` calls are blocking (ie, synchronizing)
    - The execution of host code is blocked until copying is finished
- To overlap copying and program execution, asynchronous functions are required
    - Such functions have Async suffix, eg, `hipMemcpyAsync()`
- User has to synchronize the program execution
- Asynchronous memory copies require page-locked memory

# Explicit memory API calls

- Allocate (pinned) device memory
  ```cpp
  hipError_t hipMalloc(void **devPtr, size_t size)
  ```

- Copy data
  ```cpp
  hipError_t hipMemcpy(void *dst, const void *src, size_t count, enum hipMemcpyKind kind)
  ```
  Where `kind`:
    - `hipMemcpyDeviceToHost`, `hipMemcpyHostToDevice`, <br>`hipMemcpyHostToHost`, `hipMemcpyDeviceToDevice`

- Deallocate device memory
  ```cpp
  hipError_t hipFree(void *devPtr)
  ```

# Explicit memory API calls

Pinned *host* memory

- Allocate/free pinned host memory
  ```cpp 
    hipHostMalloc(void **ptr, size_t size);
    hipHostFree(void *ptr);
  ```
- Lower operating system overhead: faster device-host copies
- Call kernels with host pointers: automatic copy to device and back
- Memory paging is disabled: no swapping, must be contiguous

# Unified memory API calls

Also known as [*Managed memory*](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/doxygen/html/group___memory_m.html)

- Allocate Unified Memory
  ```cpp
  hipError_t hipMallocManaged(void **devPtr, size_t size)
  ```
- Deallocate unified memory (same as explicitly managed memory)
  ```cpp
  hipError_t hipFree(void *devPtr)
  ```

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

