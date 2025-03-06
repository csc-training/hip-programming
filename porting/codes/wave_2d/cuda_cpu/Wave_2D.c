// Wave 2D GPU
// g++ -O3 Wave_2D.c
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"

#define DAT double

#define zeros(A,nx,ny)  DAT *A##_h; A##_h = (DAT*)malloc(((nx)*(ny))*sizeof(DAT)); \
                        for(i=0; i < ((nx)*(ny)); i++){ A##_h[i]=(DAT)0.0; }
#define free_all(A)     free(A##_h);
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

void save_array(DAT* A, int nx, int ny, int me, const char A_name[]){
    char* fname; FILE* fid; asprintf(&fname, "%d_%s.res" , me, A_name);
    fid=fopen(fname, "wb"); fwrite(A, sizeof(DAT), (nx)*(ny), fid); fclose(fid); free(fname);
}
#define SaveArray(A,nx,ny,A_name) save_array(A##_h,nx,ny,me,A_name);

// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0;
const DAT Ly  = 10.0;
const DAT k   = 1.0;
const DAT rho = 1.0;
// Numerics
const int nx  = 127;
const int ny  = 127;
const int nt  = 220;
const int nIO = 9;
const DAT dx  = Lx/((DAT)nx-1.0);
const DAT dy  = Ly/((DAT)ny-1.0);
const DAT dt  = fmin(dx,dy)/sqrt(k/rho)/4.1;
// Computing physics kernels
void init(DAT* x, DAT* y, DAT* P, const DAT Lx, const DAT Ly, const DAT dx, const DAT dy, const int nx, const int ny){
    for (int iy=0; iy<ny; iy++){
        for (int ix=0; ix<nx; ix++){
            x[ix + iy*nx] = (DAT)ix*dx - Lx/2.0;
            y[ix + iy*nx] = (DAT)iy*dy - Ly/2.0;
            P[ix + iy*nx] = exp(-(x[ix + iy*nx]*x[ix + iy*nx]) -(y[ix + iy*nx]*y[ix + iy*nx]));
        }
    }
}
void compute_V(DAT* Vx, DAT* Vy, DAT* P, const DAT dt, const DAT rho, const DAT dx, const DAT dy, const int nx, const int ny){
    for (int iy=0; iy<ny; iy++){
        for (int ix=1; ix<nx; ix++){
            Vx[ix + iy*(nx+1)] = Vx[ix + iy*(nx+1)] - dt*(P[ix + iy*nx]-P[ix-1 +  iy   *nx])/dx/rho;
        }
    }
    for (int iy=1; iy<ny; iy++){
        for (int ix=0; ix<nx; ix++){
            Vy[ix + iy*(nx  )] = Vy[ix + iy*(nx  )] - dt*(P[ix + iy*nx]-P[ix   + (iy-1)*nx])/dy/rho;
        }
    }
}
void compute_P(DAT* Vx, DAT* Vy, DAT* P, const DAT dt, const DAT k, const DAT dx, const DAT dy, const int nx, const int ny){
    for (int iy=0; iy<ny; iy++){
        for (int ix=0; ix<nx; ix++){
            P[ix + iy*nx] = P[ix + iy*nx] - dt*k*((Vx[(ix+1) + iy    *(nx+1)]-Vx[ix + iy*(nx+1)])/dx 
                                                + (Vy[ ix    + (iy+1)* nx   ]-Vy[ix + iy* nx   ])/dy );
        }
    }
}
int main(){
    int i, it;
    int me = 0;
    DAT GBs;
    // Initial arrays
    zeros(x  ,nx  ,ny  );
    zeros(y  ,nx  ,ny  );
    zeros(P  ,nx  ,ny  );
    zeros(Vx ,nx+1,ny  );
    zeros(Vy ,nx  ,ny+1);
    // Initial conditions
    init(x_h, y_h, P_h, Lx, Ly, dx, dy, nx, ny);
    // Action
    for (it=0;it<nt;it++){
        if (it==11) tic(); // start time counter
        compute_V(Vx_h, Vy_h, P_h, dt, rho, dx, dy, nx, ny);
        compute_P(Vx_h, Vy_h, P_h, dt, k,   dx, dy, nx, ny);
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
}
