# Asynchronous memory allocation and memory pools

The purpose of this exercise is to compare the overhead of different GPU memory allocation strategies and demonstrate how memory pools reduce recurring allocation costs.

The following three functions are called at the end of the file by the main() function, and need to be modified accordingly:

* The function `noRecurringAlloc()` allocates memory outside loop only once
* The function `recurringAllocNoMemPools()` allocates memory within a loop recurringly
* The function `recurringAllocMallocAsync()` obtains memory from a pool within a loop recurringly

The exercise launches the same kernel repeatedly (`10000` iterations) on an array containing `1e6` integers.  
The goal is to observe how recurring allocations affect performance, and how memory pools can reduce this overhead.

## Instructions

The task is to fill the missing function calls in the code indicated by lines beginning with `#error`, and followed by a descriptive instruction.

In this exercise you will:
- Allocate and free device memory using `hipMalloc()` and `hipFree()`
    - One function allocates this memory once
    - The other function allocates this memory in a loop
- Allocate device memory asynchronously in a HIP stream using `hipMallocAsync()` and `hipFreeAsync()`
    - Memory is allocated, freed, and reused repeatedly inside a loop
- Compare the execution times between the different allocation strategies
- Profile the execution with `rocprof` and inspect the behavior between the different approaches

Notice that we do not allocate any host memory, or do any data transfers between the host and device in this exercise.

The program will output timing information for all three cases.  

## HIP functions used

The following HIP functions are needed in this exercise:

- `hipMalloc()`
- `hipFree()`
- `hipMallocAsync()`
- `hipFreeAsync()`
- `hipStreamCreate()`
- `hipStreamSynchronize()`
- `hipStreamDestroy()`

## Profiling

<details> <summary>Profiling instructions</summary>

Run the program with ROCm profiling enabled:

```bash
run_tue rocprof --hip-trace ./<your-executable>
```

Copy the `results.json` to your local machine:

```bash
scp <your_username>@lumi.csc.fi:/scratch/project_462001376/<your_username>/hip-programming/memory/03-mempools/results.json .
```

Replace the `<your_username>` sections in the above.  
The `.` at the end means that the file will be copied to the current directory.

You can open the trace in either:
- `chrome://tracing` (in chromium)
- https://ui.perfetto.dev

In the timeline view, you should observe that the three kernels execute at overlapping times on the GPU.

</details>

## Background

<details> <summary>Asynchronous allocation and memory pools</summary>

### Recurring allocations are expensive

GPU memory allocations are significantly more expensive than ordinary CPU allocations.

Repeatedly calling:
```cpp
hipMalloc(...)
hipFree(...)
```
inside loops can introduce significant overhead.

In this exercise, the kernel itself is intentionally lightweight, which makes the allocation overhead visible in the execution time.

### Stream-ordered memory allocator

HIP provides asynchronous allocation functions:

```cpp
hipMallocAsync(...)
hipFreeAsync(...)
```

These functions use a memory pool internally.

Instead of repeatedly requesting fresh memory from the driver, freed memory is recycled and reused, which reduces allocation overhead significantly.

However, repeatedly allocating memory is still typically slower than allocating once and reusing the allocation.

</details>