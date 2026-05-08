# HIP Memory Management

This folder contains exercises for the lecture on "Memory allocations, access, and unified memory".

Exercises should be completed in order.

## Exercises

1. Explicit memory management
2. Pinned host memory
3. Unified memory
4. Unified memory prefetching
5. Investigating memory behavior using traces

## Building

```
module load LUMI/25.03
module load partition/G
module load rocm/6.3.4

CC -xhip -o <yourapp> <hip_source.cpp>
```

## Running

```
run_tue ./<yourapp>
```

## Profiling

To inspect a trace, run the program with ROCm profiling enabled:

```bash
run_tue rocprof --hip-trace ./<yourapp>
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

You can open the `.json` trace file in either:
- `chrome://tracing` (in Chrome/Chromium)
- https://ui.perfetto.dev