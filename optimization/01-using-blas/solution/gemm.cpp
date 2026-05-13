#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>

#define HIP_CHECK(cmd) \
do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        printf("HIP error: %s\n", hipGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPBLAS_CHECK(cmd) \
do { \
    hipblasStatus_t s = cmd; \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        printf("HIPBLAS error\n"); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

double elapsed_ms(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void initialize_matrix(std::vector<float>& mat)
{
#ifdef INTEGER_INIT
    for (size_t i = 0; i < mat.size(); i++) {
        mat[i] = static_cast<float>(i % 13);
    }
#else
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < mat.size(); i++) {
        mat[i] = dist(gen);
    }
#endif
}

__global__
void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


int main(int argc, char* argv[])
{
    if (argc != 3) {

        printf("Usage:\n");
        printf("  %s order repeats\n", argv[0]);

        printf("\nExample:\n");
        printf("  %s 2048 10\n", argv[0]);

        return EXIT_FAILURE;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[1]);
    const int K = atoi(argv[1]);

    const int repeat = atoi(argv[2]);

    const int warmup = 1;

    printf("Matrix sizes:\n");
    printf("  A: %d x %d\n", M, K);
    printf("  B: %d x %d\n", K, N);
    printf("  C: %d x %d\n", M, N);

    printf("Benchmark repeats: %d\n", repeat);

#ifdef INTEGER_INIT
    printf("Initialization: INTEGER\n");
#else
    printf("Initialization: RANDOM\n");
#endif

    const size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    // Host matrices
    std::vector<float> hA(static_cast<size_t>(M) * K);
    std::vector<float> hB(static_cast<size_t>(K) * N);
    std::vector<float> hC(static_cast<size_t>(M) * N);

    initialize_matrix(hA);
    initialize_matrix(hB);

    // Device matrices
    float *A, *B, *C;

    HIP_CHECK(hipMalloc(&A, sizeA));
    HIP_CHECK(hipMalloc(&B, sizeB));
    HIP_CHECK(hipMalloc(&C, sizeC));

    HIP_CHECK(hipMemcpy(A, hA.data(), sizeA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B, hB.data(), sizeB, hipMemcpyHostToDevice));

    // Naive GEMM

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
    }

    HIP_CHECK(hipDeviceSynchronize());

    double total_ms = 0.0;

    for (int i = 0; i < repeat; i++) {

        auto start = std::chrono::high_resolution_clock::now();

        naive_gemm<<<grid, block>>>(A, B, C, M, N, K);

        HIP_CHECK(hipDeviceSynchronize());

        auto stop = std::chrono::high_resolution_clock::now();

        total_ms += elapsed_ms(start, stop);
    }

    double naive_time = total_ms / repeat;

    double flops = 2.0 * static_cast<double>(M) * N * K;

    double naive_gflops = flops / (naive_time * 1e6);

    printf("\nNaive GEMM:\n");
    printf("Average Time: %.3f ms\n", naive_time);
    printf("Performance: %.2f GFLOP/s\n", naive_gflops);

    // HIPBLAS GEMM

    hipblasHandle_t handle;

    HIPBLAS_CHECK(hipblasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        HIPBLAS_CHECK(hipblasSgemm(
                handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N));
    }

    HIP_CHECK(hipDeviceSynchronize());

    total_ms = 0.0;

    for (int i = 0; i < repeat; i++) {

        auto start = std::chrono::high_resolution_clock::now();

        HIPBLAS_CHECK(hipblasSgemm(
                handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N));

        HIP_CHECK(hipDeviceSynchronize());

        auto stop = std::chrono::high_resolution_clock::now();

        total_ms += elapsed_ms(start, stop);
    }

    double hipblas_time = total_ms / repeat;

    double hipblas_gflops = flops / (hipblas_time * 1e6);

    printf("\nhipBLAS GEMM:\n");
    printf("Average Time: %.3f ms\n", hipblas_time);
    printf("Performance: %.2f GFLOP/s\n", hipblas_gflops);

    // Copy back one value for sanity check
    HIP_CHECK(hipMemcpy(hC.data(), C, sizeC, hipMemcpyDeviceToHost));

    printf("\nSample output:\n");
    printf("C[0] = %f\n", hC[0]);

    // Cleanup
    HIPBLAS_CHECK(hipblasDestroy(handle));

    hipFree(A);
    hipFree(B);
    hipFree(C);

    return 0;
}
