# The stream-ordered memory allocator and memory pools

The purpose of this exercise is to compare different memory allocation strategies within a loop and to understand the performance impact of using or not using a memory pool. The following timed functions are called at the end of the source file by the `main()` function:

* The function `noRecurringAlloc()` allocates memory outside loop only once
* The function `recurringAllocNoMemPools()` allocates memory within a loop recurringly
* The function `recurringAllocMemPool()` obtains memory from a pool within a loop recurringly

The task is to fill the missing function calls in the code indicated by lines beginning with `#error`, and followed by a descriptive instruction.
