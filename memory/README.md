# HIP Memory Management

This folder contains exercises for the lecture on "Memory management and transfers with HIP".

Exercises should be completed in order.

## Exercises

1. Inspecting explicit vs. implicit memory management
2. Avoiding recurring host-device memory transfers
3. Memory pools and async memory copies

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
run_tue rocprof --hip-trace ./<yourapp>
```

This generates a file called:

```
results.json
```

Copy the file to your local machine:

```bash
scp <your_username>@lumi.csc.fi:/scratch/project_462001376/<your_username>/hip-programming/streams/<exercise>/results.json .
```

Replace the `<your_username>` and <exercise> sections in the above. 
The `.` at the end means that the file will be copied to the current directory.

You can open the `.json` trace file in either:
- `chrome://tracing` (in Chrome/Chromium)
- https://ui.perfetto.dev