#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void)
{
    int count = 0;
    int device = 0;

    auto success = hipGetDeviceCount(&count);
    success = hipGetDevice(&device);

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    return 0;
}
