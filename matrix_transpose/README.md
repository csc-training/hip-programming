# Matrix Transpose

## Copy

### Read the copy.cpp file, compile and execute it.

```
make copy
sbatch copy.sh
```

### Profile statistics in measurement mode

```
sbatch copy_profile.sh
```

The command is `rocprof --stats ./copy`

or

`rocprof --hip-trace ./copy`

Now, a file is created called results.csv or results.hip_stats.csv, depending on the case,  with information about the kernel


```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
hipMemcpy,2,583182462,291591231,99.78098732077954
hipLaunchKernel,1,1021780,1021780,0.1748238739467205
hipMalloc,2,238697,119348,0.04084042968100799
hipDeviceSynchronize,1,12150,12150,0.0020788330838856254
__hipPushCallConfiguration,1,6640,6640,0.0011360865577778232
__hipPopCallConfiguration,1,780,780,0.00013345595106426235
```

To calculate the kernel time, add the time of hipLaunchKernel and hipDeviceSynchronize 1033930 ns

### Profile in counter mode the hardware counters

The metrics are located in the file  `metrics_copy_kernel.txt`:

```
pmc: TCC_EA_WRREQ_sum, TCC_EA_RDREQ_sum
range: 0:1
gpu: 0
kernel: copy_kernel
```

where:
* pmc is the lines with the counters:
	* TCC_EA_WRREQ_sum: Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Sum over TCC instances
	* TCC_EA_RDREQ_sumi: Number of TCC/EA read requests (either 32-byte or 64-byte). Sum over TCC instances.
	* Find metrics information in: /opt/rocm/rocprofiler/lib/metrics.xml 
* range is the range of the kernels but her eis executed only once
* gpu is the GPU id //adjust for the GPU that you use
* kernel: is the kernel name, if you need to add a second one let an empty space


Execute: `rocprof -i metrics_copy_kernel.txt -o metrics_copy.csv ./copy `

```
sbatch copy_metrics.sh
```

There is a file called metrics_copy.csv:

```
Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,vgpr,sgpr,fbar,sig,obj,TCC_EA_WRREQ_sum,TCC_EA_RDREQ_sum
0,"copy_kernel(float*, float*, int, int) [clone .kd]",0,0,0,94670,94673,16777216,1024,0,0,4,24,0,0x0,0x7faee2c09880,1048576.0000000000,1057241.0000000000
```

Duration: 164160 (ns)
TCC_EA_WRREQ_sum: 1048576
TCC_EA_RDREQ_sum: 1057241

## Matrix Transpose
### Read the copy.cpp file, compile and execute it.

```
make matrix_transpose_naive
sbatch matrix_transpose_naive.sh
```

### Profile statistics in measurement mode

```
sbatch matrix_transpose_naive.sh_profile.sh
```

The command is `rocprof --stats ./matrix_transpose_naive`

Now, a file is created called results.csv with information about the kernel and the timing

### Profile in counter mode the hardware counters

The metrics are located in the file  `metrics_transpose_naive_kernel.txt`:

```
pmc: TCC_EA_WRREQ_sum, TCC_EA_RDREQ_sum
range: 0:1
gpu: 0
kernel: copy_kernel
```
