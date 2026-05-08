# Memory management strategies

The purpose of this exercise is to compare different GPU memory management
strategies and their computational overhead on the same workload.

Complete the missing HIP API calls in the following functions in the program:

| Function | Strategy    |
| -------- | ----------- |
| `explicitMem()` | Explicit memory management |
| `explicitMemPinned()` | Explicit memory management with pinned memory |
| `unifiedMem()` | Unified memory |
| `unifiedMemPrefetch()` | Unified memory with prefetching |

Each code executes the same workload, but your task is to implement different memory management strategies for each of them.

Missing code sections are marked using: `#error`

All functions and kernels should use the default stream.

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