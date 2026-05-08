# Memory management strategies

The purpose of this exercise is to compare different GPU memory management
strategies and their computational overhead.

The following four functions are called at the end of the file by the `main()` function:

* The function `explicitMem()` represents a basic explicit memory management strategy
* The function `explicitMemPinned()` represents an explicit memory management strategy with pinned host memory
* The function `unifiedMem()` represents a basic unified memory management strategy
* The function `unifiedMemPrefetch()` represents a unified memory management strategy with prefetching

The task is to fill the missing function calls in the code indicated by lines beginning with `#error`, and followed by a descriptive instruction.

All functions and kernels should use the default stream.

## Instructions

In this exercise, you will:
1. Allocate host and device memory using three different strategies
    - Pageable (host) memory (`malloc()` / `hipMalloc()`)
    - Pinned (host) memory (`hipHostMalloc()` / `hipMalloc()`)
    - Unified memory (`hipMallocManaged()`)
2. Move data to the GPU (either explicitly or implicitly)
3. Implement prefetching in the `unifiedMemPrefetch()` function
4. Compare the timing between the different strategies

All four functions execute the same amount of steps (100) on the same amount of data
(2000 x 8000 elements).

Executed output will be a timing of the four functions:

```
The results are OK! (1.234s - ExplicitMemCopy)
The results are OK! (5.678s - ExplicitMemPinnedCopy)
The results are OK! (9.876s - UnifiedMemNoPrefetch)
The results are OK! (5.432s - UnifiedMemPrefetch)
```

## HIP functions used

The following HIP functions are needed in this exercise:

- `hipMalloc()`
- `hipFree()`
- `hipMemcpy()`
- `hipHostMalloc()`
- `hipHostFree()`
- `hipMallocManaged()`
- `hipMemPrefetchAsync()`
- `hipStreamSynchronize()`

Additionally, pageable host memory is allocated using the standard C allocation function:

- `malloc()`

## Hints

`int device;`
`hipGetDevice(&device);`

* prefetch:
`hipMemPrefetchAsync((const void*) ptr, size_t count, int device, hipStream_t stream)`

* prefetch to device on stream 0:
`hipMemPrefetchAsync(A, size, device, 0);`

* prefetch to host: use device `hipCpuDeviceId`
`hipMemPrefetchAsync(A, size, hipCpuDeviceId, 0);`

## Background

<details>
<summary>Memory management strategies</summary>

## Explicit vs. Implicit memory copies

With explicit memory management:
- Host and device allocations are separate
- Memory transfers are controlled manually using hipMemcpy()
- The programmer needs to implement all memory copies and transfers in the code

With implicit memory management:
- The programmer can allocate a single pointer
- The runtime automatically migrates memory pages between the CPU and GPU when page faults occur

## Unified memory prefetching

Unified memory pages migrate on-demand between the CPU and GPU.

Using:

hipMemPrefetchAsync(...)

the programmer can proactively move memory pages closer to the GPU before kernel execution.

This may:
- reduce page faults
- improve memory locality
- improve performance for predictable access patterns

But in this exercise it actually significantly hurts performance, as
it adds significant overhead to an already optimized transfer pattern.

</details>