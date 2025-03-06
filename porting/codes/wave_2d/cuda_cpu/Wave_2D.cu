// Wave 2D GPU
// nvcc -arch=sm_35 -O3 Wave_2D.cu
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"
#include "sys/time.h"

#define DAT double
#define GPU_ID       0
#define BLOCK_X      16
#define BLOCK_Y      16
#define GRID_X       8
#define GRID_Y       8
#define OVERLENGTH_X 1
#define OVERLENGTH_Y 1

#define zeros(A,nx,ny)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny))*sizeof(DAT)); \
                        for(i=0; i < ((nx)*(ny)); i++){ A##_h[i]=(DAT)0.0; }              \
                        cudaMalloc(&A##_d      ,((nx)*(ny))*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,((nx)*(ny))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)     free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny) cudaMemcpy( A##_h,A##_d,((nx)*(ny))*sizeof(DAT),cudaMemcpyDeviceToHost);
#define tic() ( {time_s = second();} )
#define toc() ( {time_s = second() - time_s;} )

double time_s=0.0;
double second(){
         struct timeval tp;         // struct timeval { long tv_sec; long tv_usec; };
         struct timezone tzp;       // struct timezone { int tz_minuteswest; int tz_dsttime; };
         gettimeofday(&tp,&tzp);    // int gettimeofday(struct timeval *tp, struct timezone *tzp);
         return ( (double)tp.tv_sec + (double)tp.tv_usec*1.e-6 );
}

void save_info(int precis, int nx, int ny, int me){
    FILE* fid; if (me==0){ fid=fopen("infos.inf","w"); fprintf(fid,"%d %d %d", precis, nx, ny); fclose(fid); }
}

void save_array(DAT* A, int nx, int ny, const char A_name[]){
    char* fname; FILE* fid; asprintf(&fname, "%s.dat" , A_name);
    fid=fopen(fname, "wb"); fwrite(A, sizeof(DAT), (nx)*(ny), fid); fclose(fid); free(fname);
}
#define SaveArray(A,nx,ny,A_name) gather(A,nx,ny); save_array(A##_h,nx,ny,A_name);

void clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
    cudaDeviceReset();
}
// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0;
const DAT Ly  = 10.0;
const DAT k   = 1.0;
const DAT rho = 1.0;
// Numerics
const int nx  = BLOCK_X*GRID_X - OVERLENGTH_X;
const int ny  = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
const int nt  = 220;
const int nIO = 9;
const DAT dx  = Lx/((DAT)nx-1.0);
const DAT dy  = Ly/((DAT)ny-1.0);
const DAT dt  = min(dx,dy)/sqrt(k/rho)/4.1;
// Computing physics kernels
__global__ void init(DAT* x, DAT* y, DAT* P, const DAT Lx, const DAT Ly, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy<ny && ix<nx){ x[ix + iy*nx] = (DAT)ix*dx - Lx/2.0; }
    if (iy<ny && ix<nx){ y[ix + iy*nx] = (DAT)iy*dy - Ly/2.0; }
    if (iy<ny && ix<nx){ P[ix + iy*nx] = exp(-(x[ix + iy*nx]*x[ix + iy*nx]) -(y[ix + iy*nx]*y[ix + iy*nx])); }
}
__global__ void compute_V(DAT* Vx, DAT* Vy, DAT* P, const DAT dt, const DAT rho, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy<ny && ix>0 && ix<nx){
        Vx[ix + iy*(nx+1)] = Vx[ix + iy*(nx+1)] - dt*(P[ix + iy*nx]-P[ix-1 +  iy   *nx])/dx/rho; }
    if (iy>0 && iy<ny && ix<nx){
        Vy[ix + iy*(nx  )] = Vy[ix + iy*(nx  )] - dt*(P[ix + iy*nx]-P[ix   + (iy-1)*nx])/dy/rho; }
}
__global__ void compute_P(DAT* Vx, DAT* Vy, DAT* P, const DAT dt, const DAT k, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy>0 && iy<ny-1 && ix>0 && ix<nx-1){
        P[ix + iy*nx] = P[ix + iy*nx] - dt*k*((Vx[(ix+1) + iy    *(nx+1)]-Vx[ix + iy*(nx+1)])/dx 
                                            + (Vy[ ix    + (iy+1)* nx   ]-Vy[ix + iy* nx   ])/dy ); }
}
int main(){
    int i, it;
    int me = 0;
    DAT GBs;
    // Set up GPU
    int  gpu_id=-1;
    dim3 grid, block;
    block.x = BLOCK_X; grid.x = GRID_X;
    block.y = BLOCK_Y; grid.y = GRID_Y;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d .\n",gpu_id);
    // Initial arrays
    zeros(x  ,nx  ,ny  );
    zeros(y  ,nx  ,ny  );
    zeros(P  ,nx  ,ny  );
    zeros(Vx ,nx+1,ny  );
    zeros(Vy ,nx  ,ny+1);
    // Initial conditions
    init<<<grid,block>>>(x_d, y_d, P_d, Lx, Ly, dx, dy, nx, ny);              cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        if (it==11) tic(); // start time counter
        compute_V<<<grid,block>>>(Vx_d, Vy_d, P_d, dt, rho, dx, dy, nx, ny);  cudaDeviceSynchronize();
        compute_P<<<grid,block>>>(Vx_d, Vy_d, P_d, dt, k,   dx, dy, nx, ny);  cudaDeviceSynchronize();
    }//it
    time_s = toc();
    GBs    = (DAT)sizeof(DAT)*(DAT)nx*(DAT)ny*(DAT)nIO*((DAT)nt-10.0)*(DAT)1e-9/time_s;
    printf("Perf: %d iterations took %1.3e seconds @ %1.4f GB/s.\n", nt, time_s, GBs);
    save_info(sizeof(DAT),nx,ny,me);
    SaveArray(P ,nx  ,ny  ,"P" );
    SaveArray(Vx,nx+1,ny  ,"Vx");
    SaveArray(Vy,nx  ,ny+1,"Vy");
    free_all(x );
    free_all(y );
    free_all(P );
    free_all(Vx);
    free_all(Vy);
    clean_cuda();
}
