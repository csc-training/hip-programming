#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

/*
TODO: add a device kernel that calculates y = a * x + y for vectors x, y and
constant a

Hints:

What attribute(s) do you need to add on a kernel declaration?
  - __device__?
  - __global__?
  - __shared__?
  - no attribute(s) needed?

What is the return type of a kernel?
  - int?
  - float?
  - void?
  - depends on the kernel?

What data do you need in the kernel to compute y = a * x + y, for vectors x, y,
and constant a?

What built-in variables can you use to calculate the (global) index for a
thread?
  - Is threadIdx enough or do you need blockIdx, blockDim, gridDim?
  - Is the problem one or multi-dimensional?
  - Remember the grid, block, thread hierarchy and the launch parameters
*/

int main() {
    // Use HIP_ERRCHK to help you find any errors you make with the API calls

    // Read the HIP Runtime API documentation to help you with the API calls:
    // Ctrl-click this to open it in a browser:
    // https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___memory.html

    static constexpr size_t n = 1000000;
    static constexpr size_t num_bytes = sizeof(float) * n;
    static constexpr float a = 3.4f;

    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> y_ref(n);

    // Initialise data and calculate reference values on CPU
    for (size_t i = 0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }

    // TODO: Allocate + copy initial values
    // - hipMalloc, hipMemcpy

    // TODO: Define grid dimensions + launch the device kernel
    // int/dim3 threads = ...
    // int/dim3 blocks = ...
    // kernelName<<<blocks, threads>>>(arguments);

    // TODO: Copy results back to CPU
    // - hipMemcpy

    // TODO: Free device memory
    // - hipFree

    // Check the result of the GPU computation
    printf("reference: %f %f %f %f ... %f %f\n", y_ref[0], y_ref[1], y_ref[2],
           y_ref[3], y_ref[n - 2], y_ref[n - 1]);
    printf("   result: %f %f %f %f ... %f %f\n", y[0], y[1], y[2], y[3],
           y[n - 2], y[n - 1]);

    float error = 0.0;
    static constexpr float tolerance = 1e-6f;
    for (size_t i = 0; i < n; i++) {
        const auto diff = abs(y_ref[i] - y[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42)\n", y_ref[42]);
    printf("     result: %f at (42)\n", y[42]);

    return 0;
}
