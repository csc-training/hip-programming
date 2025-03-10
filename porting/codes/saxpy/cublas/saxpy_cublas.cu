#include <iostream>
#include "cublas_v2.h"
using namespace std;

const int N = 1 << 30;

int main(){
	float *a_h, *b_h;
	a_h = new float[N];
	b_h = new float[N];
	float *a_d, *b_d;
	for(int i = 0; i < N; i++){
		a_h[i] = 1.0f;
                b_h[i] = 2.0f ;
	}
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMalloc((void**) &a_d, sizeof(float) * N);
	cudaMalloc((void**) &b_d, sizeof(float) * N);
	cublasSetVector( N, sizeof(float), a_h, 1, a_d, 1);
	cublasSetVector( N, sizeof(float), b_h, 1, b_d, 1);
	const float s = 2.0f;
	cublasSaxpy( handle, N, &s, a_d, 1, b_d, 1);
	cublasGetVector( N, sizeof(float), b_d, 1, b_h, 1);
	cudaFree(a_d);
	cudaFree(b_d);
	cublasDestroy(handle);
        float maxError = 0.0f;

	for(int i = 0; i < N; i++)
		maxError = max(maxError, abs(b_h[i]-4.0f));

        cout << "Max error: " << maxError << endl;


	delete[] a_h;
	delete[] b_h;
	return 0;
}
