#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

// TODO: add a device kernel that copies all elements of a vector
//       using GPU threads in a 2D grid


int main(void)
{
    const int n = 600;
    const int m = 400;
    const int size = n * m;
    double x[size], y[size], y_ref[size];
    double *x_, *y_;

    // initialise data
    for (int i=0; i < size; i++) {
        x[i] = (double) i / 1000.0;
        y[i] = 0.0;
    }
    // copy reference values (C ordered)
    for (int j=0; j < m; j++) {
      for (int i=0; i < n; i++) {
        y_ref[j * n + i] = x[j * n + i];
      }
    }

    // TODO: allocate vectors x_ and y_ on the GPU
    // TODO: copy initial values from CPU to GPU (x -> x_ and y -> y_)

    // TODO: define grid dimensions (use 2D grid!)
    // TODO: launch the device kernel

    // TODO: copy results back to CPU (y_ -> y)

    // confirm that results are correct
    double error = 0.0;
    for (int i=0; i < size; i++) {
        error += abs(y_ref[i] - y[i]);
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", y_ref[42 * m + 42]);
    printf("     result: %f at (42,42)\n", y[42 * m + 42]);

    return 0;
}
