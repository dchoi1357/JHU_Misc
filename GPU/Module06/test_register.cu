#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#define DEF_SIMS 20
#define N_AR 3
#define START_X 0.800, 0.900, 1.100
#define PHI -0.315415, 0.427606, 0.189134
#define C 1
#define T_MAX 9

unsigned int N_SIMS, N_BYTES;
unsigned int N_BYTES_PARM = N_AR * sizeof(float);

__constant__ float c_phi[N_AR]; // autoregressive parameters as constant

float calcMean(float *arr, unsigned int const n) {
	float sum = 0.0;
	for (int i=0; i<n; i++) {
		sum += arr[i];
	}
	return sum / n;
}

__global__ void calc_w_register(float *x0, float *x1, float *x2, float *x_out) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	float w; // white noise for AR process
	curandState_t state; // initialize rand state
	curand_init(tid, 0, 0, &state); // set seed to thread index
	
	float r_x0 = x0[tid]; // Copy values of X to register
	float r_x1 = x1[tid];
	float r_x2 = x2[tid];
	float r_x; // initialize r_x
	
	// Simulate AR(n)) process for T_MAX periods
	for (int t=0; t < T_MAX; t++) {
		w = curand_normal(&state) / 2; // w ~ Normal(0, 0.5)
		r_x = C + c_phi[2] * r_x2 + c_phi[1] * r_x1 + c_phi[0] * r_x0 + w;
		r_x2 = r_x1;
		r_x1 = r_x0;
		r_x0 = r_x;
	}
	
	x_out[tid] = r_x; // save x as output
}

int main(int argc, char* argv[]) {
	if (argc > 1) { // no command line inputs
		N_SIMS = atoi(argv[1]);
	} else {
		N_SIMS = DEF_SIMS;
	}
	N_BYTES = N_SIMS * sizeof(float); // size of array 
	printf("Running %u simulations...\n", N_SIMS);
	
	float *h_x0, *h_x1, *h_x2, *h_x; 
	h_x0 = (float*) malloc(N_BYTES); // allocate input
	h_x1 = (float*) malloc(N_BYTES); // allocate input
	h_x2 = (float*) malloc(N_BYTES); // allocate input
	h_x = (float*) malloc(N_BYTES); // allocate input
		
	float start_x [N_AR] = {START_X};
	for (int i = 0; i < N_SIMS; i++) { // set all host Xs to the same number
		h_x0[i] = start_x[0];
		h_x1[i] = start_x[1];
		h_x2[i] = start_x[2];
	}
	
	float *d_x0, *d_x1, *d_x2, *d_out; // device memory for storing X
	cudaMalloc((void **)&d_x0, N_BYTES); // device input
	cudaMalloc((void **)&d_x1, N_BYTES); // device input
	cudaMalloc((void **)&d_x2, N_BYTES); // device input
	cudaMalloc((void **)&d_out, N_BYTES); // device input
	cudaMemcpy(d_x0, h_x0, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x1, h_x1, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x2, h_x2, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	
	float h_phi [N_AR] = {PHI};
	cudaMemcpyToSymbol(c_phi, h_phi, N_BYTES_PARM); // copy params to constant
	
	calc_w_register<<<1, N_SIMS>>>(d_x0, d_x1, d_x2, d_out); // kernel
	cudaMemcpy(h_x, d_out, N_BYTES, cudaMemcpyDeviceToHost ); // copy back res
	cudaMemcpy(h_x0, d_x0, N_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy(h_x1, d_x1, N_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy(h_x2, d_x2, N_BYTES, cudaMemcpyDeviceToHost );
	
	for (int i=0; i<N_SIMS; i++) {
		printf("%.4f ", h_x[i]);
		if (i%10 == 9) {
			printf("\n");
		}
	}
	float mean = calcMean(h_x, N_SIMS);
	printf("\nResult of simulation is: %f\n", mean);
	
	cudaFree(d_x2); cudaFree(d_x1); cudaFree(d_x0); cudaFree(d_out);
	free(h_x0); free(h_x1); free(h_x2); free(h_x);
}
