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
