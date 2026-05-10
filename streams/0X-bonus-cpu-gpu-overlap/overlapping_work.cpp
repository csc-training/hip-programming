/*
 * Exercise: Overlap CPU and GPU work
 *
 * Task:
 * - Launch a GPU kernel asynchronously in two functions
 * - Synchronize immediately after in the other (before doing CPU work)
 * - In the second function, do CPU work first and synchronize only afterwards
 * - Compile with: CC -xhip -lroctx64 test.cpp -o test
 * - Profile the runtime with: rocprof --hip-trace --roctx-trace ./<executable>
 *
 * Expected:
 *   synchronize immediately  ~= GPU work + CPU work
 *   overlap CPU/GPU work     ~= max(GPU work, CPU work)
 */
#include <cstdio>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include <roctx.h>

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n",
               hipGetErrorString(result), file, line);
        exit(EXIT_FAILURE);
    }
}

#define BLOCKSIZE 256

constexpr int N = 1 << 22;
constexpr int GPU_WORK_ITERS = 2000;
constexpr int CPU_WORK_SIZE = 20'000'000;

/*
 * Synthetic GPU workload.
 *
 * Think of this as a GPU-heavy part of an application, for example
 * computing values for a large array.
 */
__global__ void gpu_kernel(float *A, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float value = 0.0f;

    for (int i = 0; i < GPU_WORK_ITERS; ++i) {
      float x = static_cast<float>(idx + i);
      float s = sinf(x);
      float c = cosf(x);
      value = sqrtf(s * s + c * c);
    }

    A[idx] = value;
  }
}

/*
 * Synthetic CPU workload.
 *
 * Think of this as independent CPU-side work, for example preparing
 * coefficients, processing metadata, or doing work for another part
 * of the program.
 */
double compute_on_cpu(int n)
{
  double sum = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = static_cast<double>(i);
    double s = sin(x);
    double c = cos(x);
    sum += sqrt(s * s + c * c);
  }

  return sum / n;
}

/* Auxiliary function to check the results */
void checkResults(float *A, double cpu_result, int n,
                  const char *strategy, const double timing_ms)
{
  float max_error = 0.0f;

  for (int i = 0; i < n; i++) {
    float error = fabs(A[i] - 1.0f);
    if (error > max_error)
      max_error = error;
  }

  double cpu_error = fabs(cpu_result - 1.0);

  if (max_error < 1.0e-5f && cpu_error < 1.0e-12)
    printf("The results are OK! (%.3f ms - %s)\n",
           timing_ms, strategy);
  else
    printf("The results are incorrect! GPU max error: %e, CPU error: %e (%s)\n",
           max_error, cpu_error, strategy);
}

/* Run once without timing to avoid measuring first-use overheads */
void warmupRun(float *d_A, hipStream_t stream)
{
  constexpr int warmup_n = 4096;
  constexpr int blocksize = BLOCKSIZE;
  const int gridsize = (warmup_n + blocksize - 1) / blocksize;

  roctxRangePush("warmup");

  gpu_kernel<<<gridsize, blocksize, 0, stream>>>(d_A, warmup_n);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipStreamSynchronize(stream));

  roctxRangePop();
}

/*
 * Version 1:
 *
 * Launch GPU work, synchronize immediately, then do CPU work.
 */
void dontOverlapCpuGpuWork(float *A, float *d_A, int n,
                           int gridsize, int blocksize,
                           hipStream_t stream)
{
  #error TODO in this function: synchronize stream immediately after gpu kernel launch, before CPU work has started

  roctxRangePush("1. Don't overlap");

  auto tStart = std::chrono::steady_clock::now();

  // Launch kernel
  roctxRangePush("launch GPU");
  gpu_kernel<<<gridsize, blocksize, 0, stream>>>(d_A, n);
  HIP_ERRCHK(hipGetLastError());
  roctxRangePop();

  roctxRangePush("CPU waiting");
  roctxRangePop();

  // CPU work starts only after GPU work has finished.
  roctxRangePush("CPU executing");
  double cpu_result = compute_on_cpu(CPU_WORK_SIZE);
  roctxRangePop();

  // Copy GPU result back to host for checking
  HIP_ERRCHK(hipMemcpy(A, d_A, n * sizeof(float), hipMemcpyDeviceToHost));

  auto tStop = std::chrono::steady_clock::now();
  double timing =
      std::chrono::duration<double, std::milli>(tStop - tStart).count();

  roctxRangePop();

  checkResults(A, cpu_result, n, "Execute CPU/GPU work sequentially", timing);
}

/*
 * Version 2:
 *
 * Launch GPU work, do independent CPU work, then synchronize only when
 * the GPU result is needed.
 */
void overlapCpuGpuWork(float *A, float *d_A, int n,
                       int gridsize, int blocksize,
                       hipStream_t stream)
{
  #error TODO in this function: synchronize stream after CPU work has started

  roctxRangePush("2. Overlap work");

  auto tStart = std::chrono::steady_clock::now();

  // GPU kernel launch
  roctxRangePush("launch GPU");
  gpu_kernel<<<gridsize, blocksize, 0, stream>>>(d_A, n);
  HIP_ERRCHK(hipGetLastError());
  roctxRangePop();

  // CPU does independent work while the GPU kernel is running.
  roctxRangePush("CPU executing");
  double cpu_result = compute_on_cpu(CPU_WORK_SIZE);
  roctxRangePop();

  // Synchronize only here, when the GPU result is needed.
  roctxRangePush("CPU waiting");
  roctxRangePop();

  // Copy GPU result back to host for checking
  HIP_ERRCHK(hipMemcpy(A, d_A, n * sizeof(float), hipMemcpyDeviceToHost));

  auto tStop = std::chrono::steady_clock::now();
  double timing =
      std::chrono::duration<double, std::milli>(tStop - tStart).count();

  roctxRangePop();

  checkResults(A, cpu_result, n, "Overlap CPU/GPU work", timing);
}

int main()
{
  // Create a stream
  hipStream_t stream;
  HIP_ERRCHK(hipStreamCreate(&stream));

  const size_t bytes = N * sizeof(float);

  const int blocksize = BLOCKSIZE;
  const int gridsize = (N + blocksize - 1) / blocksize;

  float *A;
  float *d_A;

  // Allocate (pageable) host memory
  A = (float*)malloc(bytes);
  // Allocate device memory
  HIP_ERRCHK(hipMalloc((void**)&d_A, bytes));

  // Warmup
  warmupRun(d_A, stream);

  // Run with different synchronization strategies
  dontOverlapCpuGpuWork(A, d_A, N, gridsize, blocksize, stream);
  overlapCpuGpuWork(A, d_A, N, gridsize, blocksize, stream);

  HIP_ERRCHK(hipFree(d_A));
  free(A);

  HIP_ERRCHK(hipStreamDestroy(stream));
}