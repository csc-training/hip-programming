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

// Copy all elements using threads in a 2D grid
__global__ void copy2d(double *dst, double *src, size_t num_cols,
                       size_t num_rows) {
    const size_t row_start = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t col_start = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t row_stride = blockDim.x * gridDim.x;
    const size_t col_stride = blockDim.y * gridDim.y;

    for (size_t row = row_start; row < num_rows; row += row_stride) {
        for (size_t col = col_start; col < num_cols; col += col_stride) {
            const size_t index = row * num_cols + col;
            dst[index] = src[index];
        }
    }
}

int main() {
    static constexpr size_t num_cols = 600;
    static constexpr size_t num_rows = 400;
    static constexpr size_t num_values = num_cols * num_rows;
    static constexpr size_t num_bytes = sizeof(double) * num_values;
    std::vector<double> x(num_values);
    std::vector<double> y(num_values, 0.0);

    // Initialise data
    for (size_t i = 0; i < num_values; i++) {
        x[i] = static_cast<double>(i) / 1000.0;
    }

    void *d_x = nullptr;
    void *d_y = nullptr;
    // Allocate + copy initial values
    HIP_ERRCHK(hipMalloc(&d_x, num_bytes));
    HIP_ERRCHK(hipMalloc(&d_y, num_bytes));
    HIP_ERRCHK(hipMemcpy(d_x, static_cast<void *>(x.data()), num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(d_y, static_cast<void *>(y.data()), num_bytes,
                         hipMemcpyDefault));

    // Define grid dimensions + launch the device kernel
    const dim3 threads(64, 16, 1);
    const dim3 blocks(64, 64, 1);

    copy2d<<<blocks, threads>>>(static_cast<double *>(d_y),
                                static_cast<double *>(d_x), num_cols, num_rows);

    // Copy results back to CPU
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(y.data()), d_y, num_bytes,
                         hipMemcpyDefault));

    // Free device memory
    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    printf("reference: %f %f %f %f ... %f %f\n", x[0], x[1], x[2],
           x[3], x[num_values - 2], x[num_values - 1]);
    printf("   result: %f %f %f %f ... %f %f\n", y[0], y[1], y[2], y[3],
           y[num_values - 2], y[num_values - 1]);

    // Check result of computation on the GPU
    double error = 0.0;
    for (size_t i = 0; i < num_values; i++) {
        error += abs(x[i] - y[i]);
    }

    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", x[42 * num_rows + 42]);
    printf("     result: %f at (42,42)\n", y[42 * num_rows + 42]);

    return 0;
}
