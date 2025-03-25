#include <hip/hip_runtime.h>
#include <stdio.h>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    static_assert(false, "TODO: remove me and implement the error checking. "
                         "(Hint: check the slides)");
}

int main() {
    // There's a bug in this program, find out what it is by implementing the
    // function above, and correct it
    int count = 0;
    HIP_ERRCHK(hipGetDeviceCount(&count));
    HIP_ERRCHK(hipSetDevice(count));

    int device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    return 0;
}
