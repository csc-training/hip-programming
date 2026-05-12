#include "hip/hip_runtime.h"
#include <sys/time.h>
#include <cstdio>
#include <mpi.h>
#include "jacobi.h"
#include "error_checks.h"
#include <roctx.h>
#include <iostream>

// Change this to 0 if CPU reference result is not needed
#define COMPUTE_CPU_REFERENCE 0
#define MAX_ITERATIONS 3000

// CPU kernel
void sweepCPU(double* phi, const double *phiPrev, const double *source, 
              double h2, int Nrows, int Ncols)
{ 
    int i, j;
    int index, i1, i2, i3, i4;

    for (j = 1; j < Nrows-1; j++) {
        for (i = 1; i < Ncols-1; i++) {
            index = i + j*Ncols; 
            i1 = (i-1) +   j   * Ncols;
            i2 = (i+1) +   j   * Ncols;
            i3 =   i   + (j-1) * Ncols;
            i4 =   i   + (j+1) * Ncols;
            phi[index] = 0.25 * (phiPrev[i1] + phiPrev[i2] + 
                                 phiPrev[i3] + phiPrev[i4] - 
                                 h2 * source[index]);
        } 
    }
} 

// GPU kernel
__global__ 
void sweepGPU(double *phi, const double *phiPrev, const double *source, 
              double h2, int Nrows, int Ncols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j*Ncols;
    int i1, i2, i3, i4;

    i1 = (i-1) +   j   * Ncols;
    i2 = (i+1) +   j   * Ncols;
    i3 =   i   + (j-1) * Ncols;
    i4 =   i   + (j+1) * Ncols;

    if (i > 0 && j > 0 && i < Ncols-1 && j < Nrows-1)
        phi[index] = 0.25 * (phiPrev[i1] + phiPrev[i2] + 
                             phiPrev[i3] + phiPrev[i4] - 
                             h2 * source[index]);
}

// GPU kernel
__global__ 
void sweepGPU_boundary(double *phi, const double *phiPrev, const double *source, 
              double h2, int Nrows, int Ncols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // First row
    int j = 1;
    int index = i + j*Ncols;
    int i1, i2, i3, i4;

    i1 = (i-1) +   j   * Ncols;
    i2 = (i+1) +   j   * Ncols;
    i3 =   i   + (j-1) * Ncols;
    i4 =   i   + (j+1) * Ncols;

    if (i > 0  && i < Ncols-1)
        phi[index] = 0.25 * (phiPrev[i1] + phiPrev[i2] + 
                             phiPrev[i3] + phiPrev[i4] - 
                             h2 * source[index]);

    // Last row
    j = Nrows-2;
    index = i + j*Ncols;

    i1 = (i-1) +   j   * Ncols;
    i2 = (i+1) +   j   * Ncols;
    i3 =   i   + (j-1) * Ncols;
    i4 =   i   + (j+1) * Ncols;

    if (i > 0  && i < Ncols-1)
        phi[index] = 0.25 * (phiPrev[i1] + phiPrev[i2] + 
                             phiPrev[i3] + phiPrev[i4] - 
                             h2 * source[index]);
}

double compareArrays(const double *a, const double *b, int Nrows, int Ncols)
{
    double error = 0.0;
    int i;
    for (i = 0; i < Nrows*Ncols; i++) {
        error += fabs(a[i] - b[i]);
    }
    return error/(Nrows*Ncols);
}


double diffCPU(const double *phi, const double *phiPrev, int Nrows, int Ncols)
{
    int i;
    double sum = 0;
    double diffsum = 0;
    
    for (i = 0; i < Nrows*Ncols; i++) {
        diffsum += (phi[i] - phiPrev[i]) * (phi[i] - phiPrev[i]);
        sum += phi[i] * phi[i];
    }

    return sqrt(diffsum/sum);
}


int main(int argc, char** argv) 
{ 
    int N = 8192;
    double t1, t2; // timing
    int Nrows, Ncols;
    double h = 1.0 / (N - 1);
    int iterations;
    const double tolerance = 5e-4; // Stopping condition
    int i, j, index;

    const int blocksize = 16;

    MPI_Init(&argc, &argv);
    int ntasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    if ( N % ntasks != 0 ) {
       printf("N needs to be multiple of ntasks! N=%d ntasks=%d\n", N, ntasks);
       MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    Nrows = N / ntasks;
    Ncols = N;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nghbrs[2] = {rank-1, rank+1};
    if (0 == rank) nghbrs[0] = MPI_PROC_NULL;
    if (ntasks - 1 == rank) nghbrs[1] = MPI_PROC_NULL;

    // Set GPU device
    int node_rank, node_size;
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);

    int num_devices = -1;
    HIP_CHECK( hipGetDeviceCount(&num_devices) );
    int my_device = node_rank % num_devices; 
    HIP_CHECK( hipSetDevice(my_device) );
  
    double *phi      = new double[Nrows*Ncols]; 
    double *phiPrev  = new double[Nrows*Ncols]; 
    double *source   = new double[Nrows*Ncols]; 
    double *phi_cuda = new double[Nrows*Ncols]; 

    double *phi_d, *phiPrev_d, *source_d; 
    // Size of the arrays in bytes
    const int size = Nrows*Ncols*sizeof(double); 
    double diff;
  
    // Source initialization
    for (i = 0; i < Nrows; i++) {
        for (j = 0; j < Ncols; j++) {      
            double x, y;
            x = (i + rank * Nrows- N / 2) * h;
            y = (j - N / 2) * h;
            index = j + i * Ncols;
            if (((x - 0.25) * (x - 0.25) + y * y) < 0.1 * 0.1)
                source[index] = 1e10*h*h;
            else if (((x + 0.25) * (x + 0.25) + y * y) < 0.1 * 0.1)
                source[index] = -1e10*h*h;
            else
                source[index] = 0.0;
        }            
    }

    HIP_CHECK( hipMalloc( (void**)&source_d, size) ); 
    HIP_CHECK( hipMemcpy(source_d, source, size, hipMemcpyHostToDevice) ); 

    // Reset values to zero
    for (i = 0; i < Nrows; i++) {
        for (j = 0; j < Ncols; j++) {      
            index = j + i * Ncols;
            phi[index] = 0.0; 
            phiPrev[index] = 0.0; 
        }            
    }

    HIP_CHECK( hipMalloc( (void**)&phi_d, size) ); 
    HIP_CHECK( hipMalloc( (void**)&phiPrev_d, size) ); 
    HIP_CHECK( hipMemcpy(phi_d, phi, size, hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(phiPrev_d, phiPrev, size, hipMemcpyHostToDevice) );

    // CPU version 
    if(COMPUTE_CPU_REFERENCE) { 
        t1 = MPI_Wtime();

        // Do sweeps untill difference is under the tolerance
        diff = tolerance * 2;
        iterations = 0;
        while (diff > tolerance && iterations < MAX_ITERATIONS) {
            // Halo exchange
            MPI_Sendrecv(&phi[Ncols], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                         &phi[(Nrows-1) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&phi[(Nrows-2) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                         &phi[0], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sweepCPU(phiPrev, phi, source, h * h, Nrows, Ncols);

            // Halo exchange
            MPI_Sendrecv(&phiPrev[Ncols], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                         &phiPrev[(Nrows-1) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&phiPrev[(Nrows-2) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                         &phiPrev[0], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sweepCPU(phi, phiPrev, source, h * h, Nrows, Ncols);
            
            iterations += 2;
            if (iterations % 100 == 0) {
                diff = diffCPU(phi, phiPrev, Nrows, Ncols);
                if (0 == rank) {
                  printf("%d %g\n", iterations, diff);
                }
            }
        }
        t2 = MPI_Wtime();
        if (0 == rank) {
          printf("CPU Jacobi: %g seconds, %d iterations\n", 
                 t2 - t1, iterations);
        }
    }

    // GPU version

    dim3 dimBlock(blocksize, blocksize); 
    dim3 dimGrid((Ncols + blocksize - 1) / blocksize, (Nrows + blocksize - 1) / blocksize); 

    //do sweeps until diff under tolerance
    diff = tolerance * 2;
    iterations = 0;

    t1 = MPI_Wtime();

    while (diff > tolerance && iterations < MAX_ITERATIONS) {

        roctxRangePush("halo_exchange");
        MPI_Sendrecv(&phi_d[Ncols], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                     &phi_d[(Nrows-1) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&phi_d[(Nrows-2) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                     &phi_d[0], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        roctxRangePop();
        sweepGPU<<<dimGrid, dimBlock>>>(phiPrev_d, phi_d, source_d, h*h, Nrows, Ncols); 
        HIP_CHECK( hipDeviceSynchronize() );

        roctxRangePush("halo_exchange");
        MPI_Sendrecv(&phiPrev_d[Ncols], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                     &phiPrev_d[(Nrows-1) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&phiPrev_d[(Nrows-2) * Ncols], Ncols, MPI_DOUBLE, nghbrs[1], 0,
                     &phiPrev_d[0], Ncols, MPI_DOUBLE, nghbrs[0], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        roctxRangePop();
        sweepGPU<<<dimGrid, dimBlock>>>(phi_d, phiPrev_d, source_d, h*h, Nrows, Ncols); 
        HIP_CHECK( hipDeviceSynchronize() );
        CHECK_ERROR_MSG("Jacobi kernels");
        iterations += 2;
        
        if (iterations % 100 == 0) {
            // diffGPU is defined in the header file, it uses
            // Thrust library for reduction computation
            diff = diffGPU<double>(phiPrev_d, phi_d, Nrows, Ncols);
            CHECK_ERROR_MSG("Difference computation");
            if (0 == rank) {
              printf("%d %g\n", iterations, diff);
            }
        }
    }
    
    HIP_CHECK( hipMemcpy(phi_cuda, phi_d, size, hipMemcpyDeviceToHost) ); 

    t2 = MPI_Wtime();
    if (0 == rank) {
      printf("GPU Jacobi: %g seconds, %d iterations\n", 
             t2 - t1, iterations);
    }

    //// Add here the clean up code for all allocated HIP resources
    HIP_CHECK( hipFree(phi_d) ); 
    HIP_CHECK( hipFree(phiPrev_d) );
    HIP_CHECK( hipFree(source_d) ); 

    if (COMPUTE_CPU_REFERENCE) {
        if (0 == rank) {
          printf("Average difference is %g\n", compareArrays(phi, phi_cuda, Nrows, Ncols));
        }
    }
    
    delete[] phi; 
    delete[] phi_cuda;
    delete[] phiPrev; 
    delete[] source; 
    
    return EXIT_SUCCESS; 
} 
