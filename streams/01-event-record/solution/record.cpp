#include <cstdio>
#include <time.h>
#include <hip/hip_runtime.h>
#include <chrono>

#define get_mus(X) std::chrono::duration_cast<std::chrono::microseconds>(X).count()
#define chrono_clock std::chrono::high_resolution_clock::now()

/* A simple GPU kernel definition */
__global__ void kernel(int *d_a, int n_total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_total)
    d_a[idx] = idx;
}

/* The main function */
int main(){
  // Problem size
  constexpr int n_total = 1<<22; // pow(2, 22);

  // Device grid sizes
  constexpr int blocksize = 256;
  constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;

  // Allocate host and device memory
  int *a, *d_a;
  const int bytes = n_total * sizeof(int);
  hipHostMalloc((void**)&a, bytes); // host pinned
  hipMalloc((void**)&d_a, bytes);   // device pinned

  hipEvent_t pre_kernel, post_kernel, end_event;
  // Create events
  hipEventCreate(&pre_kernel);
  hipEventCreate(&post_kernel);
  hipEventCreate(&end_event);
  float timing_a, timing_b, timing_c;

  // Create stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Start timed GPU kernel and device-to-host copy
  hipEventRecord(pre_kernel, stream);
  auto start_time = chrono_clock;

  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);

  // Record event after kernel execution
  hipEventRecord(post_kernel, stream);
  auto d2h_time = chrono_clock;

  hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream);

  // Record event after D2H memory copy
  hipEventRecord(end_event, stream);
  auto end_time = chrono_clock;

  hipStreamSynchronize(stream);

  // Exctract elapsed timings from event recordings
  hipEventElapsedTime(&timing_a, pre_kernel, post_kernel);
  hipEventElapsedTime(&timing_b, post_kernel, end_event);
  hipEventElapsedTime(&timing_c, pre_kernel, end_event);

  // Check that the results are right
  int error = 0;
  for(int i = 0; i < n_total; ++i){
    if(a[i] != i)
      error = 1;
  }

  // Print results
  if(error)
    printf("Results are incorrect!\n");
  else
    printf("Results are correct!\n");

  // Print event timings
  printf("Event timings:\n");
  printf("  %.3f ms - kernel\n", (timing_a) );
  printf("  %.3f ms - D2H copy\n", (timing_b) );
  printf("  %.3f ms - total time\n", (timing_c) );
  /* #error print event timings here */

  // Print clock timings
  printf("std::chrono timings:\n");
  printf("  %.3f ms - kernel\n", 1e3 * ((double)get_mus(d2h_time - start_time)) / CLOCKS_PER_SEC);
  printf("  %.3f ms - device to host copy\n", 1e3 * ((double)get_mus(end_time - d2h_time)) / CLOCKS_PER_SEC);
  printf("  %.3f ms - total time\n", 1e3 * (double)get_mus(end_time-start_time) / CLOCKS_PER_SEC);

  // Destroy Stream
  hipStreamDestroy(stream);

  // Destroy events
  /* #error destroy events here */
  hipEventDestroy(pre_kernel);
  hipEventDestroy(post_kernel);
  hipEventDestroy(end_event);

  // Deallocations
  hipFree(d_a); // Device
  hipHostFree(a); // Host
}
