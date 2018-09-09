#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

unsigned int N_SIMS, N_RANDS, N_BLK, N_THRD, N_BYTES;
const unsigned int MAX_THREADS = 512; // max threads per block 

// Calculate and return mean of an array of floats
float calcMean(float arr[], unsigned int const n) {
	float sum = 0.0;
	for (unsigned int i=0; i<n; i++) {
		sum += (arr[i] / n);
	}
	return sum; 
}

float calcRMSE(float arr[], unsigned int const n) {
	double sum = 0.0;
	double err = 0.0;
	for (unsigned int i=0; i<n; i++) {
		err = abs(arr[i] - M_PI);
		sum += err * err;
	}
	return (float) (sum / n);
}

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// Estimate pi using Monte Carlo simulations
__global__ void est_pi(float *pi, unsigned int N, unsigned int R) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (tid < N) {
		curandState_t state; // initialize rand state
		curand_init(tid, 0, 0, &state); // set seed to thread index
		
		int inBound = 0; // number of points within circle
		float x, y;
		for (int i = 0; i < R; i++) {
			x = curand_uniform(&state)*2 - 1; // x-coord
			y = curand_uniform(&state)*2 - 1; // y-coord
			inBound += (abs(y) < sqrtf( 1.0 - x*x )); // assume circle of radius=1
		}

		pi[tid] = 4.0f * inBound / R; // pi / 4 = inBound / total
	}
}

int main(int argc, char* argv[]) {
	if (argc == 3) { // get number of simulations based on CMDLINE input
		N_SIMS = atoi(argv[1]);
		N_RANDS = atoi(argv[2]);
	} else {
		printf("Usage: %s [nSimulations] [nRandomNumbers].\n", argv[0]);
		return EXIT_FAILURE;
	}
	N_BLK = N_SIMS / MAX_THREADS + 1; // min of one block
	N_THRD = std::min(N_SIMS, MAX_THREADS); // num of threads per block
	N_BYTES = N_SIMS * sizeof(float); // size of loss array 
	printf("Running %u simulations of %u points each...\n", N_SIMS, N_RANDS);
	
	cudaEvent_t start = get_time(); // start clock
	float *h_pi, *d_pi;
	h_pi = (float*) malloc(N_BYTES); // allocate host output
	cudaMalloc((void **) &d_pi, N_BYTES);
	est_pi<<<N_BLK, N_THRD>>>(d_pi, N_SIMS, N_RANDS);
	cudaMemcpy(h_pi, d_pi, N_BYTES, cudaMemcpyDeviceToHost ); // copy back
	
	cudaEvent_t stop = get_time(); // stop clock
	cudaEventSynchronize(stop);
	
	float dur, mean_pi, rmse;
	mean_pi = calcMean(h_pi, N_SIMS);
	rmse = calcRMSE(h_pi, N_SIMS);
	cudaEventElapsedTime(&dur, start, stop);
	
	printf("\tTook %.3f ms, output = %f, RMSE = %f, total error = %f\n", dur, 
		mean_pi, rmse, abs(mean_pi-M_PI));
	
	return EXIT_SUCCESS;
}

