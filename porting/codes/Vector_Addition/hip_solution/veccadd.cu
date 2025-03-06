#include "hip/hip_runtime.h"
/*
hipcc vecadd.cu
*/
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vecAdd(int *A,int *B,int *C,int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i<N)
   {
     C[i] = A[i] + B[i];
     //printf("C[%d] %d\n", i, C[i]);
   }
}

void vecAdd_h(int *A1,int *B1, int *C1, int N)
{
   for(int i=0;i<N;i++)
      C1[i] = A1[i] + B1[i];
}
int main(int argc,char **argv)
{
   printf("Begin \n");

  int *a, *b;  // host data
  int *c, *c2;  // results
  int n=45000000;
  size_t nBytes = n*sizeof(int);
  int blockSize, gridSize;
  blockSize=512;
  gridSize=(n+blockSize-1)/blockSize ;
  a = (int *)malloc(nBytes);
  b = (int *)malloc(nBytes);
  c = (int *)malloc(nBytes);
  c2 = (int *)malloc(nBytes);

  int *a_d,*b_d,*c_d;

  for(int i=0;i<n;i++)
     a[i]=i,b[i]=i;

  printf("Allocating device memory on host..\n");
   hipMalloc((void **)&a_d,n*sizeof(int));
   hipMalloc((void **)&b_d,n*sizeof(int));
   hipMalloc((void **)&c_d,n*sizeof(int));
   printf("Copying to device..\n");
   hipMemcpy(a_d,a,nBytes,hipMemcpyHostToDevice);
   hipMemcpy(b_d,b,nBytes,hipMemcpyHostToDevice);
   clock_t start_d=clock();
   printf("Doing GPU Vector add\n");
   hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(blockSize), 0, 0, a_d,b_d,c_d,n);
   hipDeviceSynchronize();
   clock_t end_d = clock();
   clock_t start_h = clock();
   printf("Doing CPU Vector add\n");
   vecAdd_h(a,b,c2,n);
   clock_t end_h = clock();
   double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
   double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;
   hipMemcpy(c,c_d,nBytes,hipMemcpyDeviceToHost);
   printf("%d %f %f\n",n,time_d,time_h);

   for(int i=0; i<n; i++)
   {
     if(fabs(c2[i]-c[i])>1.0e-5)
     printf("Error at position %d.\n", i );
   }
   hipFree(a_d);
   hipFree(b_d);
   hipFree(c_d);
   free(c2);
   free(c);
   free(a);
   free(b);
   return 0;
}
