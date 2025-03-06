#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <gsl/gsl_errno.h>
//#include <gsl/gsl_spline.h>
// rm *.txt a.out*;nvcc -O2 -Xptxas -v -arch=sm_30 Code.cu -lgsl -lgslcblas; optirun --no-xorg ./a.out
// rm *.txt a.out*;nvcc -O2 -Xptxas -v -arch=sm_35 Code.cu -lgsl -lgslcblas; ./a.out

__device__ double dd[3][3];
__device__ double cdt[3];
__device__ double bb[3];
typedef struct {
   double *host;
   double *device;
} vector;

typedef struct {
   double2 *host;
   double2 *device;
} vector2;

typedef struct {
  double invmaxbessel;
  vector r;
  vector k;
  vector2 monetwo;
  vector invmonetwo;
  vector bessolutions;
  vector normbessolutions;
} hankelvectors;

#define tpb 1024
#define bpg (10*1024/tpb)
#define minnbpsmp 2048/tpb
__device__ double atomicdoubleAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =(unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do 
  {
  	assumed = old;
	old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ 
__launch_bounds__(tpb,minnbpsmp) 
void gpu_ht_transf_otpp(const double * __restrict__ in,double * __restrict__ out,const double * __restrict__ invmonetwo,const double2 * __restrict__ monetwo, const double * __restrict__ bessolutions,const int nr)


{      
   int i = blockIdx.x*tpb+threadIdx.x;
   double aa;
   aa=0.0;
   int j;
   double hcij;
   __shared__ double bessol[tpb];
   __shared__ double innn[tpb];
   
   double mybessol;
   //if(i<nr)
   {
      mybessol=bessolutions[i];
//      mymonetwo=monetwo[i];
   }
   for(int p=0;p<bpg;p++)
   {
      j=p*tpb+threadIdx.x;
      //if(j<nr)
      {
            bessol[threadIdx.x]=bessolutions[j];
            innn[threadIdx.x]=in[j]*invmonetwo[j];
      }
      __syncthreads();
//#pragma unroll 100
      for(int jt=0;jt<tpb; jt++)
      {
            j=p*tpb+jt;
            //if(j<nr)
            {
                  hcij=j0(bessol[jt]*mybessol)*innn[jt];
                  aa+=hcij;
            }
      }
      __syncthreads();
   }
   //if(i<nr)
   {
      out[i]=aa*2.0;
   }
}


__host__ void gpuhk_test(hankelvectors mvectors,const int nr,const double rmax)
{   
    printf(" Hankel transform test with hat function.  \n");
    double kmax=mvectors.bessolutions.host[nr]/(rmax*2*acos(-1.0));
    FILE * pFile;
    clock_t start, end;
    double cpu_time_used;
    double *rfhat=(new double[nr]);
    double *kfhat=(new double[nr]);
    double *fhatout=(new double[nr]); 
    double *d_rfhat,*d_kfhat,*d_fhatout;
    hipMalloc((void**)&d_rfhat, sizeof(double)*(nr));
    hipMalloc((void**)&d_kfhat, sizeof(double)*(nr));
    hipMalloc((void**)&d_fhatout, sizeof(double)*(nr));
    for(int i=0;i<nr;i++)
    {
      rfhat[i]=0.0;
      if(mvectors.r.host[i]<=1.0)
      {
            rfhat[i]=1.0;
      }
    }    
    start = clock(); 
    hipMemcpy(d_rfhat, rfhat, sizeof(double)*(nr),hipMemcpyHostToDevice);
    hipLaunchKernelGGL(gpu_ht_transf_otpp, dim3(bpg), dim3(tpb), 0, 0, d_rfhat,d_kfhat,mvectors.invmonetwo.device,mvectors.monetwo.device, mvectors.normbessolutions.device,nr);
    
    hipMemcpy(fhatout, d_kfhat, sizeof(double)*(nr),hipMemcpyDeviceToHost); 
    
    pFile=fopen("gpuhattestfofk.txt","w"); 

    for(int i=0;i<nr ;i++)
    {
    fprintf(pFile,"%22.16lf %22.16lf\n",mvectors.k.host[i]*2*acos(-1.0),fhatout[i]*rmax/kmax); //The result must be scaled by rmax/kmax
    }
    fclose(pFile);
    
    hipMemset(d_fhatout, 0, sizeof(double)*nr);
    hipLaunchKernelGGL(gpu_ht_transf_otpp, dim3(bpg), dim3(tpb), 0, 0, d_kfhat,d_fhatout,mvectors.invmonetwo.device,mvectors.monetwo.device, mvectors.normbessolutions.device,nr);

    hipMemcpy(fhatout, d_fhatout, sizeof(double)*(nr),hipMemcpyDeviceToHost); 
    pFile=fopen("gpuhattestfofr.txt","w"); 

    for(int i=0;i<nr ;i++)
    {
    fprintf(pFile,"%22.16lf %22.16lf\n",mvectors.r.host[i],fhatout[i]);
    }
    fclose(pFile);
    printf("GPU Test done!\n\n");
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Two transforms in %22.6lf seconds on GPU.\n",cpu_time_used); 
    free(rfhat);
    free(kfhat); 
    free(fhatout); 
    hipFree(d_rfhat);
    hipFree(d_kfhat);
    hipFree(d_fhatout);
}
__host__ void hankelinit(hankelvectors *mvectors,const int nr,const double rmax)
{
    double dummy;
    
    mvectors[0].bessolutions.host=(new double[nr+1]);
    mvectors[0].normbessolutions.host=(new double[nr+1]);    
    mvectors[0].monetwo.host=(new double2[nr]);
    mvectors[0].invmonetwo.host=(new double[nr]);
    mvectors[0].r.host=(new double[nr]);
    mvectors[0].k.host=(new double[nr]);
   size_t free, total;
   printf("\n");
   hipMemGetInfo(&free,&total);   
   printf("%lu KB free of total %lu KB at the beginning. \n\n",free/1024,total/1024);
    
    
    FILE * pFile;
    
    hipMalloc((void**)&mvectors[0].bessolutions.device, sizeof(double)*(nr+1));
    hipMalloc((void**)&mvectors[0].normbessolutions.device, sizeof(double)*(nr+1));
    hipMalloc((void**)&mvectors[0].monetwo.device, sizeof(double2)*nr);
    hipMalloc((void**)&mvectors[0].invmonetwo.device, sizeof(double)*(nr));
    hipMalloc((void**)&mvectors[0].r.device, sizeof(double)*(nr));
    hipMalloc((void**)&mvectors[0].k.device, sizeof(double)*(nr));
    pFile=fopen("bessel_zeros.in","r");
    for(int i=0;i<=nr;i++)
    {
      fscanf(pFile,"%lf\n",&dummy);
      mvectors[0].bessolutions.host[i]=dummy;
    }
    fclose(pFile);
    double kmax=mvectors[0].bessolutions.host[nr]/(rmax*2*acos(-1.0));
    printf("rmax= %lf,     kmax/(2*Pi)= %lf \n \n", rmax,kmax);
    for(int i=0;i<nr;i++)
    {      
      mvectors[0].normbessolutions.host[i]=mvectors[0].bessolutions.host[i]/sqrt(mvectors[0].bessolutions.host[nr]);
      mvectors[0].r.host[i]=mvectors[0].bessolutions.host[i]*rmax/mvectors[0].bessolutions.host[nr];
      mvectors[0].k.host[i]=mvectors[0].bessolutions.host[i]/(2.0*rmax*acos(-1.0));
      mvectors[0].monetwo.host[i].x=fabs(j1(mvectors[0].bessolutions.host[i]));
      mvectors[0].monetwo.host[i].y=1.0/fabs(j1(mvectors[0].bessolutions.host[i]));
      mvectors[0].invmonetwo.host[i]=1.0/(pow(mvectors[0].monetwo.host[i].x,2)*mvectors[0].bessolutions.host[nr]);
    }
    hipMemcpy(mvectors[0].bessolutions.device, mvectors[0].bessolutions.host, sizeof(double)*(nr+1),hipMemcpyHostToDevice);
    hipMemcpy(mvectors[0].normbessolutions.device, mvectors[0].normbessolutions.host, sizeof(double)*(nr+1),hipMemcpyHostToDevice);
    hipMemcpy(mvectors[0].monetwo.device, mvectors[0].monetwo.host, sizeof(double2)*(nr),hipMemcpyHostToDevice);
    hipMemcpy(mvectors[0].invmonetwo.device, mvectors[0].invmonetwo.host, sizeof(double)*(nr),hipMemcpyHostToDevice);
    hipMemcpy( mvectors[0].r.device, mvectors[0].r.host, sizeof(double)*(nr),hipMemcpyHostToDevice);
    hipMemcpy( mvectors[0].k.device, mvectors[0].k.host, sizeof(double)*(nr),hipMemcpyHostToDevice);
    
   printf("\n");
   hipMemGetInfo(&free,&total);   
   printf("%lu KB free of total %lu KB after basic allocations.\n",free/1024,total/1024);
    
}

int main( int argc, char* argv[] )
{
    printf(" Example of how to do CUDA quasi-discreta Hankel transform. \n The space grid is obtained from the solutions of the first kind Bessel functions j0(x)=0. \n The solution must be provided in advanced in the file  bessel_zeros.in. \n The same function is used for for doing forwaard or backward transform. \n If a forward (r->k) transform is made the results must be multiplied by rmax/kmax.  \n All the arrays needed for transform are kept in a structure which is iniliazed by the hankelinit function.\n Only double precision is available. \n The flag -arch=sm_xx should have xx>=20 \n");
    // Size of vectors
    int nr=bpg*tpb;
    printf("%d grid points \n\n",nr);
    const double rmax=0.005*nr;
    hankelvectors hvecs[1];
    
    //FILE * pFile;
    hankelinit(hvecs, nr, rmax); // allocate memory and initialize the vectors
    gpuhk_test(hvecs[0], nr,rmax);
}
