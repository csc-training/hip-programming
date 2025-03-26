---
title:  Memory allocations, access, and unified memory
subtitle: GPU programming with HIP
author:   CSC Training
date:     2025-03
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
- *Explicit*: User manually manages data movement between host and device. Host memory can be allocated with GPU-unaware allocators (`malloc`/`free` etc).
- *Implicit*: The runtime manages data movement between host and device. Host memory needs to be allocated with special allocators.
  - **Page-locked** (pinned) host allocations: Data moves to device with kernel invocations and is not stored there.
  - **Unified memory** (managed memory): Page faults will initiate data movement.
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

- Data access in device code is initially slower <br>⇒ Must be optimized with prefetches and hints
- Externalize memory management to library

:::
::::::

# Side-topic: Virtual Memory addressing

::::::{.columns}
:::{.column width=70%}
- Modern operating systems utilize virtual memory
    - Memory is organized in memory pages
    - Memory pages can reside on swap area on the disk 
- `malloc` returns an address in the virtual memory
:::

:::{.column}
![](img/virtual_memory_addressing.png){width=80%}
:::
::::::

# Page-locked (or pinned) memory

:::{.fragment}
- Normal `malloc` allows swapping, page migration and page faults
- `hipHostMalloc` page-locks the allocation to a physical memory location
  - Deallocate with `hipFreeHost()`
:::
:::{.fragment}
- **(A)** Direct Memory Access (DMA)
  - Higher transfer speeds between host and device
- **(B)** Access host memory from GPU without explicit `hipMemCpy`
  - Useful in sporadic access pattern
  - Implicit host-device access
  - *Very* slow
:::
:::{.fragment}
- Excessive page-locking can degrade system performance
:::

:::{.notes}
- having too much page-locked allocs is almost never a problem
:::


# Asynchronous memcopies

- Normal `hipMemcpy()` calls are blocking (ie, synchronizing)
    - The execution of host code is blocked until copying is finished
- To overlap copying and program execution, asynchronous functions are required
    - Such functions have Async suffix, eg, `hipMemcpyAsync()`
- User has to synchronize the program execution
- Concurrency with memory copy and computation requires page-locked host allocations

# Explicit memory API calls

- Allocate device memory
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

Page-locked host memory

- Allocate/free page-locked host memory
  ```cpp 
    hipHostMalloc(void **ptr, size_t size);
    hipHostFree(void *ptr);
  ```
- Lower operating system overhead: faster device-host copies
- ***SLOW***: Call kernels with host pointers: automatic copy to device and back
- Memory paging is disabled: no swapping, must be contiguous

:::{.notes}
- the speedup is not very big usually
:::

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

- Benefit of asynchronous memory management: allocate/free memory from/to a pool
  - Full allocate/free is slower
- Pool is deallocated after async free (`hipFreeAsync`) when it is synchronised (default behaviour)

<small>

| Description | API call |
|--|--|
| Allocate memory from pool. If pool is too small, assign more memory to it. | `hipMallocAsync(void** devPtr, size_t size, hipStream_t hStream)` |
| Free memory to the pool in the specific stream | `hipFreeAsync(void* devPtr, hipStream_t hStream)` |

</small>

:::{.notes}
**Extra:**
- Modify default behaviour: resize pool down to value when synchronizing (`hipDeviceGetDefaultMemPool`, `hipMemPoolSetAttribute`, `hipMemPoolAttrReleaseThreshold`)
- beta version
:::

# Memory pools - Example
<small>

<div class="column">
Example 1 - slow
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
Example 2 - fast
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

:::{.notes}
- ROCm has memory leak bug related to mallocasync and freeasync
:::

# Summary

- Host and device have separate physical memories
  - The data copies between CPU and GPU should be minimized
- Explicit or implicit memory management
- Page-locked host allocation: DMA and kernel access to host memory 
- Unified Memory: improve productivity and cleaner implementation
    - Optimize with hints and prefetches
- Recurring allocation and deallocation is slow, use memory pools instead 
  - Libraries provide pooled Unified Memory support as well (eg, Umpire)
