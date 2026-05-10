# Executing multiple GPU kernels concurrently with HIP streams

This exercise demonstrates how multiple HIP streams can be used to execute independent GPU kernels concurrently. 

In the current code, all kernels are launched into the default stream.
This causes the kernels to execute sequentially.

If your program executes correctly, you should get the following output:

```
1.258728 1.258728 1.258728 ...
1.618034 1.618034 1.618034 ...
0.000000 0.000345 0.000345 ...
```

Printing the 10 first values in each kernel output array.

After running the code, you will use `rocprof` and Perfetto, to validate that your kernels are executing concurrently on the GPU.

Note that we do not execute any host to device memory copies in this
program, and all kernel data is generated directly on the GPU.

## Instructions

In this exercise, you will modify the `main()` function to:

- create three HIP streams
- launch one kernel into each stream
- synchronize the host with each stream
- destroy all streams

The GPU kernels are already implemented and do not need to be modified.

<details>
<summary><strong>Bonus exercise</strong></summary>

Modify the code so that `kernel_a` executes and copies the memory back to host, before `kernel_b` and `kernel_c` execute asynchronously.

</details>

## HIP functions used

The following HIP functions are needed in this exercise:

* `hipStreamCreate()`
* `hipStreamSynchronize()`
* `hipStreamDestroy()`

<details>
<summary><strong>Additional options</strong></summary>

In the exercise, you are instructed to use `hipStreamSynchronize()` to synchronize
your streams one by one. 

Another way to synchronize the GPU with the host is to wait for all GPU work at once using:

```
hipDeviceSynchronize()
```

Which synchronizes the entire device rather than a single stream.

</details>

## Profiling kernel concurrency

After completing the exercise, validate that the kernels execute concurrently.

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
scp <your_username>@lumi.csc.fi:/scratch/project_462001376/<your_username>/hip-programming/streams/02-streams-asynckernel/results.json .
```

Replace the `<your_username>` sections in the above.
The `.` at the end means that the file will be copied to the current directory.

You can open the trace in either:
- `chrome://tracing` (in chromium)
- https://ui.perfetto.dev

In the timeline view, you should observe that the three kernels execute at overlapping times on the GPU.

## Understanding the kernels

<details>
<summary><strong>Understanding the GPU kernels</strong></summary>

The kernels in this exercise are synthetic workloads only for teaching purposes.

Each GPU thread performs repeated floating-point computations using mathematical functions (e.g. `sin`, `cos`, `log`).

The workloads copy large amounts of data (~256MB) and are quite heavy computationally (although quite redundant)
so that concurrent execution becomes visible in the profiling tools, as well as data transfers.

</details>
