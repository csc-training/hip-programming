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
 * Kernels always need the __global__ attribute
 *
 * The return type of a kernel is always void (i.e. kernels don't return
 * anything)
 *
 * The pointers accessed in kernels must be in memory accessible to the GPU
 *
 * Variables passed to the kernel are copied from the CPU to the GPU "directly"
 * by the runtime
 */
__global__ void saxpy(int n, float a, float *x, float *y) {
    /*
     * threadIdx.x gives the index of this thread *inside* its block (in the x
     direction)
     * Threads in different blocks may have the same value for threadIdx

     * This is why you need to use blockIdx and blockDim
     * in addition to the threadIdx to compute the global thread index

     * global tid = the index of this thread inside the block it belongs to
     *            + the index of the block the thread is in, in the grid of
                    blocks
     *            * the number of threads in each block
    */
    const int num_threads_before_my_block = blockIdx.x * blockDim.x;
    const int my_index_in_my_block = threadIdx.x;
    const int tid = my_index_in_my_block + num_threads_before_my_block;

    /*
     * Some examples
     *
     * first thread in the first block
     * threadIdx.x = 0
     * blockIdx.x = 0
     * blockDim.x = 1024 (see the value 'threads' in the main function)
     * tid = 0
     *
     * second thread in the first block
     * threadIdx.x = 1
     * blockIdx.x = 0
     * blockDim.x = 1024
     * tid = 1
     *
     * first thread in the second block
     * threadIdx.x = 0
     * blockIdx.x = 1
     * blockDim.x = 1024
     * tid = 1024
     *
     * second thread in the second block
     * threadIdx.x = 1
     * blockIdx.x = 1
     * blockDim.x = 1024
     * tid = 1025
     *
     * eighth thread in the hundreth block
     * threadIdx.x = 7
     * blockIdx.x = 99
     * blockDim.x = 1024
     * tid = 7 + 99 * 1024 = 101383
     */

    /*
     * Stride is equal to the total number of threads in the entire grid:
     * #threads/grid = #threads/block * #blocks/grid
     *
     * It's constant accross the kernel (i.e. same value for all
     * threads)
     *
     * With the launch parameters used in the main function
     * it's 1024 * 128 = 131072
     */
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        y[i] += a * x[i];
    }
}

int main() {
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
    // The problem size is 10^6, so this is not enough threads to cover the
    // whole range. That's ok, since we're using a for loop in the kernel above,
    // so that each thread computes multiple values, not just one. With this
    // implementation, the problem size can change and everything still works,
    // i.e. we've decoupled the code from data.
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
