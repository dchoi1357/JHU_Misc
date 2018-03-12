#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

// In the following section, define the model Parameters
#define N_AR 3
#define START_X 0.800, 0.900, 1.100
#define PHI -0.315415, 0.427606, 0.189134
#define C 1.500
// End model parameters

unsigned int N_SIMS, N_BLK, N_THRD, N_BYTES, T_MAX;
const unsigned int MAX_THREADS = 512; // max threads per block 
__constant__ float c_phi[N_AR]; // autoregressive parameters as constant
unsigned int N_BYTES_PARM = N_AR * sizeof(float); // size of parameter constant

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

// Simulate a AR(n) process saving temp results to registers
__global__ void sim_w_register(float *x0, float *x1, float *x2, float *x_out, 
								const unsigned int N, const unsigned int T) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (tid < N) {
		curandState_t state; // initialize rand state
		curand_init(tid, 0, 0, &state); // set seed to thread index

		float r_x0 = x0[tid]; // Copy values of X to register
		float r_x1 = x1[tid];
		float r_x2 = x2[tid];
		float r_x; // initialize r_x
		
		float w; // white noise for AR process
		for (int t=0; t < T; t++) { // Simulate for T_MAX periods
			w = curand_normal(&state) / 2; // w ~ Normal(0, 0.5)
			r_x = C + c_phi[2]*r_x2 + c_phi[1]*r_x1 + c_phi[0]*r_x0 + w;
			r_x2 = r_x1;
			r_x1 = r_x0;
			r_x0 = r_x;
		}

		x_out[tid] = r_x; // save x as output
	}
}

// Simulate a AR(n) process saving work to global mem directly
__global__ void sim_w_gmem(float *x0, float *x1, float *x2, float *x_out, 
								const unsigned int N, const unsigned int T) {
	unsigned int const i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (i < N) {
		curandState_t state; // initialize rand state
		curand_init(i, 0, 0, &state); // set seed to thread index

		float w; // white noise for AR process
		for (int t=0; t < T; t++) { // Simulate for T_MAX periods
			w = curand_normal(&state) / 2; // w ~ Normal(0, 0.5)
			x_out[i] = C + c_phi[2]*x2[i] + c_phi[1]*x1[i] + c_phi[0]*x0[i] + w;
			x2[i] = x1[i];
			x1[i] = x0[i];
			x0[i] = x_out[i];
		}
	}
}


int main(int argc, char* argv[]) {
	if (argc == 3) { // get number of simulations based on CMDLINE input
		N_SIMS = atoi(argv[1]);
		T_MAX = atoi(argv[2]);
	} else {
		printf("Usage: %s [nSimulations] [maxTimePeriods].\n", argv[0]);
		return EXIT_FAILURE;
	}
	N_BLK = N_SIMS / MAX_THREADS + 1; // min of one block
	N_THRD = std::min(N_SIMS, MAX_THREADS); // num of threads per block
	N_BYTES = N_SIMS * sizeof(float); // size of array 
	printf("Running %u simulations over %u time periods...\n", N_SIMS, T_MAX);
	
	float *h_x0, *h_x1, *h_x2, *h_x_reg, *h_x_gmem; 
	h_x0 = (float*) malloc(N_BYTES); // allocate input
	h_x1 = (float*) malloc(N_BYTES); // allocate input
	h_x2 = (float*) malloc(N_BYTES); // allocate input
	h_x_reg = (float*) malloc(N_BYTES); // allocate output for register run
	h_x_gmem = (float*) malloc(N_BYTES); // allocate output for gmem run
		
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

	float h_phi [N_AR] = {PHI}; // constant for AR parms
	cudaMemcpyToSymbol(c_phi, h_phi, N_BYTES_PARM); // copy params to constant
	
	/**** Simulate with Registers *****/
	cudaEvent_t start1 = get_time(); // start time 
	cudaMemcpy(d_x0, h_x0, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x1, h_x1, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x2, h_x2, N_BYTES, cudaMemcpyHostToDevice); //copy to device

	sim_w_register<<<N_BLK, N_THRD>>>(d_x0, d_x1, d_x2, d_out, N_SIMS, T_MAX); 
	cudaMemcpy(h_x_reg, d_out, N_BYTES, cudaMemcpyDeviceToHost ); // copy back
	cudaEvent_t stop1 = get_time(); // stop time
	cudaEventSynchronize(stop1);
	
	/**** Simulate with Global Memory *****/
	cudaEvent_t start2 = get_time(); // start time 
	cudaMemcpy(d_x0, h_x0, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x1, h_x1, N_BYTES, cudaMemcpyHostToDevice); //copy to device
	cudaMemcpy(d_x2, h_x2, N_BYTES, cudaMemcpyHostToDevice); //copy to device

	sim_w_gmem<<<N_BLK, N_THRD>>>(d_x0, d_x1, d_x2, d_out, N_SIMS, T_MAX); 
	cudaMemcpy(h_x_gmem, d_out, N_BYTES, cudaMemcpyDeviceToHost ); // copy back
	cudaEvent_t stop2 = get_time(); // stop time
	cudaEventSynchronize(stop2);
	
	// Calculate and print simulation results and timing
	float mean1= calcMean(h_x_reg, N_SIMS);
	float tmp1 = 0;
	cudaEventElapsedTime(&tmp1, start1, stop1);
	printf("\twith registers, result=%f, %.3f ms taken, \n", mean1, tmp1);
	
	float mean2 = calcMean(h_x_gmem, N_SIMS);
	float tmp2 = 0;
	cudaEventElapsedTime(&tmp2, start2, stop2);
	printf("\twith global memory, result=%f, %.3f ms taken, \n", mean2, tmp2);
	printf("\n");
	
	cudaFree(d_x2); cudaFree(d_x1); cudaFree(d_x0); cudaFree(d_out);
	free(h_x0); free(h_x1); free(h_x2); free(h_x_reg); free(h_x_gmem);
	return EXIT_SUCCESS;
}
