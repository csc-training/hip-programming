# The stream-ordered memory allocator and memory pools

The purpose of this exercise is to compare different memory allocation
strategies within a loop and to understand the performance impact of using or not using a memory pool. The following timed functions are called at the end of the source file by the `main()` function:

* The function `noRecurringAlloc()` allocates memory outside loop only once
* The function `recurringAllocNoMemPools()` allocates memory within a loop recurringly
* The function `recurringAllocMemPoolNoSync()` obtains memory from a pool within a loop recurringly (no synchronization within loop)
* The function `recurringAllocMemPoolSync()` obtains memory from a pool within a loop recurringly and synchronizes (synchronization within loop)

The task is to fill the missing function calls in the code indicated by lines beginning with `#error`, and followed by a descriptive instruction.

IMPORTANT NOTE! Unfortunately, the support for memory pools was only recently added to HIP (version 5.2.0), but the available HIP version at Puhti is only 5.1.0. Therefore, please replace the following HIP terms by the CUDA equivalents to make the application compile (in the future, the corresponding HIP commands should be available for use): 

* `hipMallocAsync` -> `cudaMallocAsync`
* `hipFreeAsync` -> `cudaFreeAsync`
