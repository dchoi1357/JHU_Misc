#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#define DEF_SIMS 20
#define N_AR 3
#define START_X 1.000, 0.800, 1.200
#define PHI 0.227606, 0.689134, -0.315415
#define T_MAX 9

unsigned int N_SIMS, N_BYTES;
unsigned int N_BYTES_PARM = N_AR * size(float);

__constant__ float d_phi[N_AR]; // autoregressive parameters

void genRandArray(float *arr, const unsigned int n, const float max) {
	for (int i = 0; i < n; i++) {
		arr[i] = rand() / (RAND_MAX / max);
	}
}

__global__ void calc_w_register(float *x0, float *x1, float *x2) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	float r;
	curandState_t state; // initialize rand state
	currand_init(tid, 0, 0, &state); // set seed to thread indx
	
	// Copy values of X to register
	float r_x0 = x0[tid]; 
	float r_x1 = x1[tid];
	float r_x2 = x2[tid];
	float x;
	
	for (int t=0; t < T_MAX; t++) {
		x = d_phi[2] * r_x2 + d_phi[1] * r_x1 + d_phi[0] * r_x0 ;
		r_x2 = x;
		r_x3 = r_x2;
		r_x2 = r_x1;
		r_x0 = r_x1;
	}
}

int main(int argc, char* argv[]) {
	if (argc > 1) { // no command line inputs
		N_SIMS = atoi(argv[1]);
	} else {
		N_SIMS = DEF_SIMS;
	}
	N_BYTES = N_SIMS * sizeof(float); // size of array 
	
	h_x1 = (float*) malloc(N_BYTES); // allocate input
	h_x2 = (float*) malloc(N_BYTES); // allocate input
	h_x3 = (float*) malloc(N_BYTES); // allocate input
	
	float *h_x0, *h_x1, *h_x2;
	float start_x = {START_X};
	for (int i = 0; i < N_SIMS; i++) { // set all X0 to the same number
		// x0s[i] = rand() / (RAND_MAX / 1.0);
		h_x0[i] = start_x[0];
		h_x1[i] = start_x[1];
		h_x2[i] = start_x[2];
	}
	
	float *d_x0, *d_x1, *d_x2; // device memory for storing X
	cudaMalloc((void **)&d_X, N_BYTES); // device input
	cudaMalloc((void **)&d_X, N_BYTES); // device input
	cudaMalloc((void **)&d_X, N_BYTES); // device input
	cudaMemcpy(d_x0, h_x0, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x1, h_x1, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x2, h_x2, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	
	float h_phi [] = {PHI};
	d_phi = (float*) malloc(N_BYTES_PARM); // initialize AR parameters
	cudaMemcpyToSymbol(d_phi, h_phi, N_BYTES_PARM); // copy to constant
	
	calc_w_register<<<1, N_SIMS>>>(d_x1, d_x2, d_x3);
}