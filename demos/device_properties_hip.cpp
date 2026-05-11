#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

int main(int argc, char *argv[])
{
    int count, device;
    hipError_t err;

    err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("No HIP devices found\n");
        return 1;
    }

    err = hipGetDevice(&device);
    if (err != hipSuccess) {
        printf("Error in GetDevice\n");
        return 1;
    }

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, device);
    if (err != hipSuccess) {
        printf("Error in GetDeviceProperties\n");
        return 1;
    }

    // Note: name is empty string on LUMI, see https://github.com/ROCm/ROCm/issues/1625
    printf("Name: %s\n", prop.name);
    printf("Memory: %.2f GiB\n", prop.totalGlobalMem / pow(1024., 3));
    printf("Compute Units (CUs): %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per CU (approx): %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp/Wavefront Size: %d\n", prop.warpSize);
    printf("Clock Rate (MHz): %d\n", prop.clockRate / 1000);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Max Grid Size: [%d, %d, %d]\n", 
           prop.maxGridSize[0], prop.maxGridSize[1],  prop.maxGridSize[2]);
    printf("Max Threads Dim: [%d, %d, %d]\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    return 0;
}
