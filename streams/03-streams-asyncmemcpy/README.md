# Using asynchronous memory copies for multiple streams

This exercise builds upon the previous stream concurrency exercise,
by adding asynchronous memory copies from device to host.

Start by **copying your last exercise result to this folder**:

```
cp ../02-streams-asynckernel/streams.cpp .
```

Previously, kernels were launched concurrently in separate HIP streams, but memory copies back to the host were still blocking.

Expected output is still the same:

```
1.258728 1.258728 1.258728 ...
1.618034 1.618034 1.618034 ...
0.000000 0.000345 0.000345 ...
```

## Instructions

In this exercise, you will:

- replace blocking `hipMemcpy()` calls with `hipMemcpyAsync()`
- associate each memory transfer with its corresponding stream
- manually synchronize streams before accessing host memory
- validate asynchronous execution using ROCm profiling tools

Depending on where you placed your stream synchronization in the earlier exercise,
make sure to move the synchronization calls to be after the device-to-host memory transfers.

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
- `chrome://tracing` (in chromium)
- https://ui.perfetto.dev

In the timeline view, you should observe that the three kernels execute at overlapping times on the GPU.

# Extra: Using pinned host memory

<details>
<summary>Optional: Using pinned host memory with hipHostMalloc()</summary>

By default, this exercise uses ordinary pageable host memory allocated with:

```cpp
malloc()
```

To actually be able to do asynchronous memory transfers, it will require
pinned (page-locked) host memory.

Try replacing:

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

After making these changes, compare the profiling trace again.

</details>