---
title:  Multi-GPU programming and HIP+MPI
subtitle: GPU programming with HIP
author:   CSC Training
date:     2025-03
lang:     en
---

# Running on HPC Systems

 <div class="column" width=60%>

* supercomputer are a collection of thousands of nodes
* currently there are  2 to 8 GPUs per node
* more GPU resources per node, better per-node-performance 
 
</div>

 <div class="column" width=40%>
    ![](img/lumi.png){.center width=200%}
    <small>Lumi - Pre-exascale system in Finland</small>
 </div>


# Three Levels of Parallelism

1. GPU - GPU threads on the CUs: HIP
2. Node - Multiple GPUs and CPUs: MPI, OpenMP
3. Supercomputer - Many nodes connected with interconnect: MPI 

![](img/parallel_regions.png){.center width=60% }

# Computing in Parallel

- parallel computing
    - a problem is split into smaller subtasks
    - multiple subtasks are processed simultaneously using multiple GPUs

<!-- Copyright CSC -->
 ![](img/compp.svg){.center width=40%}

# Parallel Programming Models

<!-- Copyright CSC -->
 ![](img/processes-threads.svg){.center width=80%}
<div class=column>
**MPI: Processes**

- independent execution units
- MPI launches N processes at application startup
- works over multiple nodes
- data exchange via messages
</div>
<div class=column>

**OpenMP: Threads**

- threads share memory space
- threads are created and destroyed  (parallel regions)
- limited to a single node
- directive based

</div>

# GPU Context

* a context is established implicitly on the current device when the first HIP function requiring an active context is evaluated 
* several processes can create contexts for a single device
    * the device resources are allocated per context
* by default, one context per device per process in HIP
    * (CPU) threads of the same process share the primary context (for each device)
* HIP supports explicit context management 

::: notes
A GPU context is an execution environment that manages resources such as memory allocations, streams, and kernel execution for a specific GPU. It acts as an interface between the application and the GPU, ensuring that operations like memory management and kernel launches are handled correctly.
:::

# Device Selection and Management

- Return the number of hip capable devices by `count`
```cpp
hipError_t hipGetDeviceCount(int *count)
```
- GPU device numbering starting from 0
- Set device as the current device for the calling host thread
```cpp
hipError_t hipSetDevice(int device)
```
- Return the current device for the calling host thread by `device`
```cpp
hipError_t hipGetDevice(int *device)
```
- Reset and destroy all current device resources
```cpp
hipError_t hipDeviceReset(void)
```

# Querying Device Properties

* One can query the properties of different devices in the system using
  `hipGetDeviceProperties()` function
    * No context needed
    * Provides e.g. name, amount of memory, warp size, support for unified
      virtual addressing, etc.
    * Useful for code portability

Return the properties of a HIP capable device by `prop`
```
hipError_t hipGetDeviceProperties(struct hipDeviceProp *prop, int device)
```


# Multi-GPU Programming Models

<div class="column">
* One GPU per process
    * Syncing is handled through message passing (e.g. MPI)
* Many GPUs per process
    * Process manages all context switching and syncing explicitly
* One GPU per thread
    * Syncing is handled through thread synchronization requirements
</div>

<div class="column">
![](img/single_proc_mpi_gpu2.png){width=50%}
![](img/single_proc_multi_gpu.png){width=50%}
![](img/single_proc_thread_gpu.png){width=50%}
</div>

# One GPU per Process

* recommended for multi-process applications using MPI
* message passing library takes care of all GPU-GPU communication
* each process interacts with only one GPU which makes the implementation
  easier and less invasive (if MPI is used anyway)
    * apart from each process selecting a different device, the implementation
      looks much like a single-GPU program
    * easy manage device selection using environment variables `ROC_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`


# Many GPUs per Process

* process switches the active GPU using `hipSetDevice()` function 
* after selecting the default device, operations such as the following are effective only
  on the selected GPU:
    * memory operations
    * kernel execution
    * streams and events
* asynchronous function calls are required to overlap work

# Many GPUs per Process: Code Example

```cpp
// Launch kernels
for(unsigned n = 0; n < num_devices; n++) {
  hipSetDevice(n);
  kernel<<<blocks[n],threads[n], 0, stream[n]>>>(arg1[n], arg2[n], size[n]);
}
//Synchronize all kernels with host
for(unsigned n = 0; n < num_devices; n++) {
  hipSetDevice(n);
  hipStreamSynchronize(stream[n]);
}
```

# Many GPUs per Process, One GPU per Thread

* One GPU per CPU thread
    * E.g. one OpenMP CPU thread per GPU being used
* HIP is threadsafe
    * Multiple threads can call the functions at the same time
* Each thread can create its own context on a different GPU
    * `hipSetDevice()` sets the device and create a context per thread
    * Easy device management with no changing of device
* From the point of view of a single thread, the implementation closer to a single-GPU case
* Communication between threads still not trivial


# Many GPUs per Process, One GPU per Thread: Code Example

```cpp
// Launch and synchronize kernels from parallel CPU threads using HIP
#pragma omp parallel num_threads(num_devices)
{
  unsigned n = omp_get_thread_num();
  hipSetDevice(n);
  kernel<<<blocks[n],threads[n], 0, stream[n]>>>(arg1[n], arg2[n], size[n]);
  hipStreamSynchronize(stream[n]);
}
```

# Direct Peer to Peer Access

* Access peer GPU memory directly from another GPU
    * Pass a pointer to data on GPU 1 to a kernel running on GPU 0
    * Transfer data between GPUs without going through host memory
    * Lower latency, higher bandwidth

```cpp
// Check peer accessibility
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)

// Enable peer access
hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags)

// Disable peer access
hipError_t hipDeviceDisablePeerAccess(int peerDevice)
```
* Between AMD GPUs, the peer access is always enabled (if supported)

# Peer to Peer Communication

* Devices have separate memories
* With devices supporting unified virtual addressing, `hipMemCpy()` with
  `kind=hipMemcpyDefault`, works:
```cpp
hipError_t hipMemcpy(void* dst, void* src, size_t count, hipMemcpyKind kind)
```
* Other option which does not require unified virtual addressing
```cpp
hipError_t hipMemcpyPeer(void* dst, int  dstDev, void* src, int srcDev, size_t count)
```
* If direct peer to peer access is not available or implemented, the functions should fall back to a normal copy through host memory



# Compiling MPI+HIP Code

* Trying to compile code with any HIP calls with other than the `hipcc`
  compiler can result in errors
* Either set MPI compiler to use `hipcc`, eg for OpenMPI:
```bash
OMPI_CXXFLAGS='' OMPI_CXX='hipcc'
```
* or separate HIP and MPI code in different compilation units compiled with
  `mpicxx` and `hipcc`
    * Link object files in a separate step using `mpicxx` or `hipcc`
* on LUMI, `cc` and `CC` wrappers know about both MPI and HIP

# Selecting the Correct GPU

* Typically all processes on the node can access all GPUs of that node
* The following implementation allows utilizing all GPUs using one process per GPU

```cpp
int deviceCount, nodeRank;
MPI_Comm commNode;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &commNode);
MPI_Comm_rank(commNode, &nodeRank);
hipGetDeviceCount(&deviceCount);
hipSetDevice(nodeRank % deviceCount);
```
::: notes
* Can be done from slurm using `ROCR_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`
:::

# GPU-GPU Communication through MPI

* CUDA/ROCm aware MPI libraries support direct GPU-GPU transfers
    * Can take a pointer to device buffer (avoids host/device data copies)
* Unfortunately, currently no GPU support for custom MPI datatypes (must use a
  datatype representing a contiguous block of memory)
    * Data packing/unpacking must be implemented application-side on GPU
* On LUMI, enable on runtime by `export MPICH_GPU_SUPPORT_ENABLED=1`
* Having a fallback for pinned host staging buffers is a good idea.

# Summary

- There are various options to write a multi-GPU program
- Use `hipSetDevice()` to select the device, and the subsequent HIP calls
  operate on that device
- If you have an MPI program, it is often best to use one GPU per process, and
  let MPI handle data transfers between GPUs 
- GPU-aware MPI is required when passing device pointers to MPI

     * Using host pointers does not require any GPU awareness

