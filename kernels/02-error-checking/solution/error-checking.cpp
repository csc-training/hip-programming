#include <hip/hip_runtime.h>
#include <stdio.h>

/* HIP error handling macro */
#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int count = 0;
    HIP_ERRCHK(hipGetDeviceCount(&count));
    // When setting the device, the argument must be 0 <= arg < #devices
    // See
    // https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___device.html#ga43c1e7f15925eeb762195ccb5e063eae
    // for the API
    HIP_ERRCHK(hipSetDevice(count - 1));

    int device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    return 0;
}
