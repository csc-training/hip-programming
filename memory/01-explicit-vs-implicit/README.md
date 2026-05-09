# Memory management strategies

Compare explicit memory management and unified memory in HIP.

Complete the missing HIP API calls in the following functions in the program:

| Function | Strategy    |
| -------- | ----------- |
| `explicitMem()` | Explicit memory management |
| `explicitMemPinned()` | Explicit memory management with pinned memory |
| `unifiedMem()` | Unified memory |
| `unifiedMemPrefetch()` | Unified memory with prefetching |

Each code executes the same workload, but your task is to implement different memory management strategies for each of them.

Missing code sections are marked using: `#error`

All functions and kernels should use the default stream in this exercise.

## HIP functions used

| Explicit memory | Unified memory |
|---|---|
| `hipMalloc` | `hipMallocManaged` |
| `hipMemcpy` | `hipMemPrefetchAsync` *(optional)* |
| `hipFree`   | `hipFree` |

Additionally, pageable host memory is allocated/freed using the standard C allocation function:

- `malloc()` and `free()`

And pinned host memory with:

- `hipHostMalloc()` and `hipHostFree()`

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

### Explicit memory management

With explicit memory management:
- Host and device allocations are separate
- Memory transfers are controlled manually using hipMemcpy()
- The programmer needs to implement all memory copies and transfers in the code

```cpp
int *A, *d_A;

// Host memory allocation
A = (int*) malloc(N * sizeof(int));
// Device memory allocation
hipMalloc((void**)&d_A, N * sizeof(int));

hipMemcpy(d_A, A, N * sizeof(int), hipMemcpyHostToDevice);

kernel<<<...>>>(d_A);

hipMemcpy(A, d_A, N * sizeof(int), hipMemcpyDeviceToHost);

// Free memory on device and host
hipFree(d_A);
free(A);
```

Things to note:
- `A` and `d_A` point to different physical memories
- `hipMemcpy` is required before and after the kernel launch
- Forgetting a copy operation or mixing up the host/device pointers is a common source of bugs

### Implicit memory management through unified memory

With unified memory:
- The programmer can use a single memory allocation
- The runtime automatically moves memory pages between the CPU and GPU when page faults occur

```cpp
int *A; // <-- single pointer

hipMallocManaged((void**)&A, N * sizeof(int));

kernel<<<...>>>(A);

// No blocking hipMemcpy call with unified memory -> synchronize before accessing results on host
hipStreamSynchronize(0);

printf("%d\n", A[0]);

hipFree(A);
```

## Unified memory prefetching

Unified memory pages migrate on-demand between the CPU and GPU.

Prefetching allows the programmer to move pages to the GPU before kernel execution:

```cpp
hipMemPrefetchAsync(A, N * sizeof(int), deviceId);

kernel<<<...>>>(A);
```

This may:
- reduce page faults
- improve memory locality
- improve performance for predictable access patterns

But in this exercise it actually significantly hurts performance, as
it adds significant overhead to an already optimized transfer pattern.

</details>