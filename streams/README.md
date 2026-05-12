# HIP Streams

This folder contains exercises for the second-day lecture on "Streams, events, and synchronization."

Exercises should be completed in order.
Exercises 3 and 4 build on your solution to Exercise 2.

## Exercises

1. Creating and using a stream
2. Launching concurrent kernels
3. Asynchronous memory copies
4. Using events to time kernels

## Building

Source the [setup_env_script](../setup_env_lumi) with:

```bash
source setup_env_lumi
```

And compile with:

```
CC -xhip -o <yourapp> <hip_source.cpp>
```

## Running

Source the [setup_env_script](../setup_env_lumi) with:

```bash
source setup_env_lumi
```

And use (on Tuesday):

```
run_tue ./<yourapp>
```

## Profiling

To inspect a trace, run the program with ROCm profiling enabled:

```bash
run_tue rocprofv3 --hip-trace --runtime-trace --kernel-trace --output-format pftrace -- ./<yourapp>
```

This generates a file with a suffix: `.pftrace`

Copy the file to your local machine from e.g. www.lumi.csc.fi from the home directory file tree.

Another option is to copy it from your local (laptop) terminal:

```bash
scp <your_username>@lumi.csc.fi:/scratch/project_462001376/<your_username>/hip-programming/streams/<exercise>/<path-to-your-file>/<file>.pftrace .
```

Replace the `<your_username>`, `<exercise>` and `<path-to-your-file>` sections in the above. 
The `.` at the end means that the file will be copied to the current directory.

You can open the `.json` trace file in either:
- `chrome://tracing` (in Chrome/Chromium)
- https://ui.perfetto.dev