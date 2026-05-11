# Avoiding recurring host-device memory transfers

Previously in [exercise 1](../01-explicit-vs-implicit/README.md), data was repeatedly:

- initialized on the CPU
- transferred to the GPU
- processed on the GPU

inside every iteration of a loop.

In this exercise, we optimize the workflow by keeping the data resident on the GPU during the iterative loop,
avoiding recurring host-to-device transfers.

The following two functions are implemented:

- `explicitMemNoCopy()`
    - Explicit memory management where data remains on the GPU during the iterative loop
- `unifiedMemNoCopy()`
    - Unified memory management where data remains resident on the GPU during the iterative loop

Both approaches need to be modified so that array initialization also happens directly on the GPU.

## Instructions

In this exercise, you will:

1. Allocate memory using:
    - Explicit device memory (`hipMalloc()`)
    - Unified memory (`hipMallocManaged()`)
2. Initialize memory directly on the GPU using:
    - `hipMemset()`
3. Avoid recurring host-to-device memory copies inside the iteration loop
4. Launch the provided kernel `hipKernel` on the device
5. Copy or prefetch data back to the CPU only after all GPU work has completed
6. Compare the timing between the two approaches

Modify each of the lines marked with `#error`.

<details>
<summary><strong>Bonus exercise: Prefetching this time</strong></summary>

- In the instructed workflow, the Unified Memory allocation is first accessed by GPU operations inside the loop
- Without prefetching, the first device-side access may cause managed-memory pages to migrate to the GPU during execution.
- Try modifying the Unified Memory version by prefetching the allocation to the GPU once before the loop:


```cpp
int device;
HIP_ERRCHK(hipGetDevice(&device));

HIP_ERRCHK(hipMemPrefetchAsync(A, size, device, 0));
HIP_ERRCHK(hipStreamSynchronize(0));
```

Then run the program again and compare the execution time of the Unified Memory version.

This case is different from the previous exercise: here the data is intended to stay resident on the GPU during the iterative loop, so prefetching once before the loop may improve performance.

</details>

## HIP functions used

- `hipMalloc()`
- `hipFree()`
- `hipMemcpy()`
- `hipMallocManaged()`
- `hipMemPrefetchAsync()`
- `hipMemset()`
- `hipStreamSynchronize()`

## Hints

* Get current GPU device id:
`int device;`
`hipGetDevice(&device);`

* prefetch:
`hipMemPrefetchAsync((const void*) ptr, size_t count, int device, hipStream_t stream)`

* prefetch to device on stream 0:
`hipMemPrefetchAsync(A, size, device, 0);`

* prefetch to host on stream 0: use device `hipCpuDeviceId`
    - Note that `hipCpuDeviceId` is a _predefined HIP constant_ representing CPU memory
`hipMemPrefetchAsync(A, size, hipCpuDeviceId, 0);`

* Device memset (setting all elements of array A to zeroes)
`hipMemset(A, 0, size);`

## Background

<details> <summary>Keeping data resident on the GPU</summary>

In many GPU applications, repeatedly transferring data between
the CPU and GPU becomes a major bottleneck.

A more efficient workflow is often:

- Allocating data once
- Keeping data resident on the GPU
- Performing multiple GPU operations
- Transferring the results back only when needed

In this exercise we realize that array initialization to zeros can also
happen on the device, avoiding the need to transfer data between the host
and device in each loop iteration.

The goal is to minimize data movement and maximize GPU residency.

</details>