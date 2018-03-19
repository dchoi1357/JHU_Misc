#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

// In the following section, define the prob distribution parameters
#define N_ET 2
#define LAMBDA 25.0, 3.0
#define A 0.8f, 3.0f
#define B 5.0f, 0.5f

// parameters saved as constants
__constant__ float c_lambda[N_ET]; // parameters as constant
__constant__ float c_A[N_ET]; 
__constant__ float c_B[N_ET]; 
unsigned int N_BYTES_PARM = N_AR * sizeof(float); // size of parameter constant

unsigned int N_SIMS, N_BLK, N_THRD, N_BYTES, T_MAX;
const unsigned int MAX_THREADS = 512; // max threads per block 

// Calculate and return mean of an array of floats
float calcMean(float *arr, unsigned int const n) {
	float sum = 0.0;
	for (int i=0; i<n; i++) {
		sum += arr[i];
	}
	return sum / n; 
}

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__global__ void sim_freq(unsigned int *f_out, const unsigned int ET, 
						const unsigned int N) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (tid < N) {
		curandState_t state; // initialize rand state
		curand_init(tid, 0, 0, &state); // set seed to thread index

		f_out[tid] = curand_poisson(&state, c_lambda[ET]); // save loss frequency
	}
}

__global__ void sim_severity(float *s_out, unsigned int *freq, 
							const unsigned int ET, const unsigned int N) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (tid < N) {
		curandState_t state; // initialize rand state
		curand_init(tid, 0, 0, &state); // set seed to thread index
		float sum = 0.0f;
		float unif = 0.0f; // temp var for storing uniform rand 
		for (int f=0; f < freq[tid]; f++ {
			unif = curand_uniform(&state);
			sum += c_b[ET] / powf(1-unif, 1/c_a[ET]);
		}
		
		s_out[tid] = sum;
	}
}


void simulate() {
	float *h_x0, *h_x1, *h_x2, *h_x; 
	h_x0 = (float*) malloc(N_BYTES); // allocate input
	h_x1 = (float*) malloc(N_BYTES); // allocate input
	h_x2 = (float*) malloc(N_BYTES); // allocate input
	h_x = (float*) malloc(N_BYTES); // allocate output 
		
	float start_x [N_AR] = {START_X};
	for (int i = 0; i < N_SIMS; i++) { // set all host Xs to the same number
		h_x0[i] = start_x[0];
		h_x1[i] = start_x[1];
		h_x2[i] = start_x[2];
	}
	
	float *d_x0, *d_x1, *d_x2, *d_out; // device memory for storing X
	cudaMalloc((void **)&d_x0, N_BYTES); // allocate device input
	cudaMalloc((void **)&d_x1, N_BYTES); // allocate device input
	cudaMalloc((void **)&d_x2, N_BYTES); // allocate device input
	cudaMalloc((void **)&d_out, N_BYTES); // allocate device output

	cudaMemcpy(d_x0, h_x0, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x1, h_x1, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x2, h_x2, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	
	sim_register<<<1, N_SIMS>>>(d_x0, d_x1, d_x2, d_out, N_SIMS, T_MAX); 
	
	cudaMemcpy(h_x, d_out, N_BYTES, cudaMemcpyDeviceToHost ); // copy back
	
	// Free up memory
	cudaFree(d_x2); cudaFree(d_x1); cudaFree(d_x0); 
	cudaFree(d_out); cudaFree(c_phi);
	free(h_x0); free(h_x1); free(h_x2); free(h_x);
}

int main(int argc, char* argv[]) {
	if (argc == 3) { // get number of simulations based on CMDLINE input
		N_SIMS = atoi(argv[1]);
	} else {
		printf("Usage: %s [nSimulations].\n", argv[0]);
		return EXIT_FAILURE;
	}

	N_BYTES = N_SIMS * sizeof(float); // size of array 
	printf("Running %u simulations ..\n", N_SIMS);

	float h_phi [N_ET] = {LAMBDA}; // lambda for  Poisson
	float h_phi [N_ET] = {LAMBDA};
	float h_phi [N_ET] = {LAMBDA};
	cudaMemcpyToSymbol(c_phi, h_phi, N_BYTES_PARM); // copy params to constant	
	
	cudaEvent_t start = get_time(); // start time 	
	simulate(); // simulating with registers
	cudaEvent_t stop = get_time(); // stop time
	cudaEventSynchronize(stop);
	// Calculate and print simulation results and timing
	float x_mu = calcMean(h_x, N_SIMS);
	float dur = 0;
	cudaEventElapsedTime(&dur, start, stop);
	printf("\twith %s, result=%f, %.3f ms taken, \n", typeName, x_mu, dur);

	
	return EXIT_SUCCESS;
}
