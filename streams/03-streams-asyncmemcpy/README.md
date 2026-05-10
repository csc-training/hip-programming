# Using asynchronous memory copies for multiple streams

Previously, kernels were launched concurrently in separate HIP streams, but memory copies back to the host were still blocking.

This exercise extends the previous example by performing device-to-host memory transfers asynchronously in each stream.

Start by **copying your last exercise result to this folder**:

```
cp ../02-streams-asynckernel/streams.cpp .
```

Expected output is still the same:

```
1.258728 1.258728 1.258728 ...
1.618034 1.618034 1.618034 ...
0.000000 0.000345 0.000345 ...
```

## Instructions

In this exercise, you will:

- Replace blocking `hipMemcpy()` D2H calls with `hipMemcpyAsync()`
- Associate each memory transfer with its corresponding stream
- Manually synchronize your streams with the host before accessing host memory
- Inspect kernel and memory transfer overlap using ROCm profiling tools

Make sure to synchronize **after** the device-to-host memory transfers.

## HIP functions used

The following HIP functions are needed in this exercise:

* `hipMemcpyAsync()`
* `hipStreamSynchronize()`

## Profiling kernel concurrency

After completing the exercise, inspect asynchronous memory copy behavior through
`rocprof` and Perfetto.

Run the program with ROCm profiling enabled:

```bash
run_tue rocprof --hip-trace ./<your-executable>
```

This generates a file called:

```
results.json
```

Copy the file to your local machine:

```bash
scp <your_username>@lumi.csc.fi:/scratch/project_462001376/<your_username>/hip-programming/streams/03-streams-asyncmemcpy/results.json .
```

Replace the `<your_username>` sections in the above.  
The `.` at the end means that the file will be copied to the current directory.

You can open the trace in either:
- `chrome://tracing` (in Chromium)
- https://ui.perfetto.dev

In the timeline view, you should observe that the three kernels execute at overlapping times on the GPU and memory copies to host
can happen in each stream while computation in other kernels is ongoing.

For this small example, performance differences may be minimal, but the execution timeline behavior becomes visible in the profiler.

# Extra: Using pinned host memory

<details>
<summary><strong>Optional: Using pinned host memory with hipHostMalloc()</strong></summary>

By default, this exercise uses ordinary pageable host memory allocated with:

```cpp
malloc()
```

To achieve true asynchronous memory transfers and overlap between transfers and computation,
pinned (page-locked) host memory is typically required.

Try replacing your host memory allocations (for `a`, `b` and `c`):

```cpp
a = (float*) malloc(N_bytes);
```

with: 

```cpp
HIP_ERRCHK(hipHostMalloc((void**)&a, N_bytes));
```

and also:

```cpp
free(a);
```

with:

```cpp
HIP_ERRCHK(hipHostFree(a));
```

After making these changes, compile and run your program again, and inspect its trace.

</details>