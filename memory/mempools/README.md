# The stream-ordered memory allocator and memory pools

The purpose of this exercise is to compare 4 different memory allocation
strategies within a loop and to understand the performance impact of using or not using a memory pool. The following timed functions are called at the end of the source file by the `main()` function:

* `noRecurringAlloc();`
* `recurringAllocNoMemPools();`
* `recurringAllocMemPoolNoSync();`
* `recurringAllocMemPoolSync();`

The task is to fill the blanks indicated by: `#error`

IMPORTANT NOTE! Unfortunately, the support for memory pools was only recently added to HIP (version 5.2.0), but the available HIP version at Puhti is only 5.1.0. Therefore, please replace the following HIP terms by the CUDA equivalents to make the application compile (in the future, the corresponding HIP commands should be available for use): 

* `hipMallocAsync` -> `cudaMallocAsync`
* `hipFreeAsync` -> `cudaFreeAsync`
* `hipMemPool_t` -> `cudaMemPool_t`
* `hipDeviceGetDefaultMemPool` -> `cudaDeviceGetDefaultMemPool`
* `hipMemPoolSetAttribute` -> `cudaMemPoolSetAttribute`
* `hipMemPoolAttrReleaseThreshold` -> `cudaMemPoolAttrReleaseThreshold`
