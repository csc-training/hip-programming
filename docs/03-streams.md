---
title:  Synchronisation and streams
author: CSC Training
date:   2021-11
lang:   en
---

# Streams

* What is a stream?

    * A sequence of operations that execute in issue-order on the GPU
    * HIP operations in different streams could run concurrently
    * The ROCm 4.5.0 brings the  Direct Dispatch, the runtime directly queues a packet to the AQL queue in Dispatch and some of the synchronization. 
    * The previous ROCm uses queue per stream

---

## Concurrency 

![width:1000px height:13cm](./img/streams.png)


---

## Amount of concurrency 

![width:1000px height:13cm](./img/streams2.png)

---
## Default

* Only a single stream is used if not defined
* Commands are synchronized except the Async calls and Kernels
    
---


# Stream/Events API

* hipStreamCreate: Creates an asynchronous stream
* hipStreamDestroy: Destroy an asynchronous stream
* hipStreamCreateWithFlags: Creates an asynchronous stream with specified flags
* hipEventCreate: Create an event
* hipEventRecord: Record an event in a specified stream
* hipEventSynchronize: Wait for an event to complete
* hipEventElapsedTime: Return the elapsed time between two events
* hipEventDestroy: Destroy the specified event 
HIP API: https://raw.githubusercontent.com/RadeonOpenCompute/ROCm/master/AMD-HIP-API-4.5.pdf 
---

# Example - Data Transfer and Compute
* Serial
```
  hipCheck( hipEventRecord(startEvent,0) );
  
  hipCheck( hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice) );
  
  hipLaunchKernelGGL(kernel, n/blockSize, blockSize, 0, 0, d_a, 0);
  
  hipCheck( hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost) );
  
  hipCheck( hipEventRecord(stopEvent, 0) );
  hipCheck( hipEventSynchronize(stopEvent) );
  hipCheck( hipEventElapsedTime(&duration, startEvent, stopEvent) );
  printf("Duration of sequential transfer and execute (ms): %f\n", duration);
```

---
# How to improve the performance?

* Use streams to overlap computation with communication
```
hipStream_t stream[nStreams];
for (int i = 0; i < nStreams; ++i)
    hipStreamCreate(&stream[i])
```
* Use Asynchronous data transfer
```
hipMemcpyAsync(dst, src, bytes, hipMemcpy kind, stream)
```
* Execute kernels on different streams
```
hipLaunchKernelGGL((some_kernel, gridsize, blocksize, shared_mem_size, stream, arg0, arg1, ...);
```
---

# Synchronization (I)

* Synchronize everything

```
hipDeviceSynchronize()
```
* Synchronize a specific stream
Blocks host until all HIP calls are completed on this stream
```
hipStreamSynchronize(streamid)
```

---

# Synchronization (II)

* Synchronize using Events
    * Create event
    ```
    hipEvent_t stopEvent
    hipEventCreate(&stopEvent)
    ```
    * Record an event in a specific stream and wait until is recorded
    ```
    hipEventRecord(stopEvent,0)
    hipEventSynchronize(stopEvent)
    ```
    * Make a stream wait for a specific event
    ```
    hipStreamWaitEvent(stream[i], stopEvent, unsigned int flags) 	
    ```

---


