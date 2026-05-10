# Using HIP events for timing kernels

This exercise builds upon the previous asynchronous memory copy exercise by introducing HIP events.

Start by **copying your previous exercise solution into this folder**:

```
cp ../03-asyncmemcopy/streams.cpp .
```

In this exercise, we will measure our GPU kernel execution time using HIP events.

Expected output is still the same:

```
1.258728 1.258728 1.258728 ...
1.618034 1.618034 1.618034 ...
0.000000 0.000345 0.000345 ...
```

In addition, your program should print timing information for the kernels, for example:

```
kernel_a time: 1234.567 us
kernel_b time: 2345.678 us
kernel_c time: 3456.789 us
```

## Instructions

Starting from your previous solution:

1. Create HIP events for:
    - kernel start
    - kernel end
    for each stream (e.g. `start_a`, `start_b`, etc.)
2. Record start and end events around each kernel launch using:
    - `hipEventRecord()`
3. Measure the elapsed time (milliseconds) between two events with:
    - `hipEventElapsedTime()`
4. Print the measured kernel execution times
5. Destroy all created events before exiting the program

Ensure that you're placing your events in the correct stream.

Refer to your trace output in Perfetto to check that the timings make sense.

## HIP functions used

- `hipEventCreate()`
- `hipEventRecord()`
- `hipEventElapsedTime()`
    - Returns milliseconds by default
- `hipEventDestroy()`

## Bonus exercise:

<details>
<summary><strong>Profiling the execution</strong></summary>

Profile your program with rocprof. Do you see the same numbers
reported by the profiler as you do with events?
You might see some discrepancy, why?

Try to identify when the events are created on the host.

</details>

## Background

<details>
<summary><strong>About HIP events</strong></summary>

### HIP Events

HIP events are essentially markers inserted into a stream.

Events can be used to:
- measure GPU execution time
- track progress in a stream from the host
- synchronize e.g. across different streams

A typical timing workflow looks like:
```
hipEvent_t start
hipEvent_t end

hipEventCreate(&start);
hipEventCreate(&end);

hipEventRecord(start, stream);

kernel<<<..., stream>>>();

hipEventRecord(end, stream);

hipEventElapsedTime(&time_ms, start, end);

hipEventDestroy(*start);
hipEventDestroy(*end);
```

Synchronizing across streams:

```
hipEvent_t event_a_done;

hipEventCreate(&event_a_done);

// stream A
kernel_a<<<..., stream_a>>>();

hipEventRecord(event_a_done, stream_a);

// stream B waits for stream A
hipStreamWaitEvent(stream_b, event_a_done, 0);

kernel_b<<<..., stream_b>>>();

// independent work
kernel_c<<<..., stream_c>>>();
```

Here, `kernel_b` will only begin after execution in `stream_a`
has reached `event_a_done`.

</details>