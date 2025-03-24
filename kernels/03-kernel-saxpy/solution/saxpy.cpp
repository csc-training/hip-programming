#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

__global__ void saxpy(int n, float a, float *x, float *y) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        y[i] += a * x[i];
    }
}

int main() {
    static constexpr size_t n = 100000;
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

    // Allocate + copy initial values
    void *d_x = nullptr;
    void *d_y = nullptr;
    HIP_ERRCHK(hipMalloc(&d_x, num_bytes));
    HIP_ERRCHK(hipMalloc(&d_y, num_bytes));
    HIP_ERRCHK(hipMemcpy(d_x, static_cast<void *>(x.data()), num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(d_y, static_cast<void *>(y.data()), num_bytes,
                         hipMemcpyDefault));

    // Define grid dimensions + launch the device kernel
    static constexpr int threads = 1024;
    static constexpr int blocks = 128;
    saxpy<<<blocks, threads>>>(n, a, static_cast<float *>(d_x),
                               static_cast<float *>(d_y));

    // Copy results back to CPU
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(y.data()), d_y, num_bytes,
                         hipMemcpyDefault));

    // Free device memory
    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    printf("reference: %f %f %f %f ... %f %f\n", y_ref[0], y_ref[1], y_ref[2],
           y_ref[3], y_ref[n - 2], y_ref[n - 1]);
    printf("   result: %f %f %f %f ... %f %f\n", y[0], y[1], y[2], y[3],
           y[n - 2], y[n - 1]);

    // Check result of computation on the GPU
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
