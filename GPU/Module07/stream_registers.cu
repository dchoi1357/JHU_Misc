#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

// In the following section, define the prob distribution parameters
#define N_PARAMS 3
#define PARAM1 50.0f, 3.0f, 0.5f // format: LAMBDA, A, B
#define PARAM2 1.5f, 0.8f, 5.0f

// parameters saved as constants
unsigned int N_BYTES_PRM = N_PARAMS * sizeof(float); // size of parameter 

unsigned int N_SIMS, N_BLK, N_THRD, N_BYTES_I, N_BYTES_F;
const unsigned int MAX_THREADS = 512; // max threads per block 

// Calculate and return mean of an array of floats
float calcMean(float arr[], unsigned int const n) {
	double sum = 0.0;
	for (unsigned int i=0; i<n; i++) {
		sum += (arr[i] / n);
	}
	return sum; 
}

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__global__ void sim_freq(unsigned int *f_out, float *prm, unsigned int N) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (tid < N) {
		float lambda = prm[0];
		curandState_t state; // initialize rand state
		curand_init(tid, 0, 0, &state); // set seed to thread index

		f_out[tid] = curand_poisson(&state, lambda); // save loss frequency
	}
}

__global__ void sim_severity(float *loss_out, unsigned int *freq, float *prm,
							const unsigned int N) {
	unsigned int const tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (tid < N) {
		double A = prm[1];
		double B = prm[2];
		
		curandState_t state; // initialize rand state
		curand_init(tid, 0, 0, &state); // set seed to thread index
		double sum = 0.0;
		double unif = 0.0; // temp var for storing uniform rand 
		for (int f=0; f < freq[tid]; f++) {
			unif = curand_uniform_double(&state);
			sum += B / pow(1-unif, 1/A);
		}		
		loss_out[tid] = (float) sum;
	}
}

void asynch() {
	return;
}

int main(int argc, char* argv[]) {
	if (argc == 2) { // get number of simulations based on CMDLINE input
		N_SIMS = atoi(argv[1]);
	} else {
		printf("Usage: %s [nSimulations].\n", argv[0]);
		return EXIT_FAILURE;
	}
	N_BLK = N_SIMS / MAX_THREADS + 1; // min of one block
	N_THRD = std::min(N_SIMS, MAX_THREADS); // num of threads per block
	N_BYTES_F = N_SIMS * sizeof(float); // size of loss array 
	N_BYTES_I = N_SIMS * sizeof(unsigned int); // size of frequency array
	printf("Running %u simulations ...\n", N_SIMS);
	
	cudaStream_t s1, s2; // Create and initialize streams
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);
	
	// allocate and copy parameter to device
	float h_prm1 [N_PARAMS] = {PARAM1};
	float h_prm2 [N_PARAMS] = {PARAM2};
	cudaHostRegister(h_prm1, N_BYTES_PRM, cudaHostRegisterDefault);
	cudaHostRegister(h_prm2, N_BYTES_PRM, cudaHostRegisterDefault);
	float *d_prm1, *d_prm2;
	cudaMalloc((void **)&d_prm1, N_BYTES_PRM);
	cudaMalloc((void **)&d_prm2, N_BYTES_PRM);
	cudaMemcpyAsync(d_prm1, h_prm1, N_BYTES_PRM, cudaMemcpyHostToDevice, s1); 
	cudaMemcpyAsync(d_prm2, h_prm2, N_BYTES_PRM, cudaMemcpyHostToDevice, s2); 
	
	unsigned int *h_freq1, *d_freq1, *h_freq2, *d_freq2; // frequency arrays 
	float *h_loss1, *d_loss1, *h_loss2, *d_loss2; // loss arrays
	cudaMalloc((void **)&d_freq1, N_BYTES_I); // device array 
	cudaMalloc((void **)&d_loss1, N_BYTES_F);
	cudaMalloc((void **)&d_freq2, N_BYTES_I);
	cudaMalloc((void **)&d_loss2, N_BYTES_F);
	cudaMallocHost((void**)&h_freq1, N_BYTES_I); // pinned host array
	cudaMallocHost((void**)&h_loss1, N_BYTES_F);
	cudaMallocHost((void**)&h_freq2, N_BYTES_I);
	cudaMallocHost((void**)&h_loss2, N_BYTES_F);

	float dur, mean1, mean2; // to record duration and averages
	
	// ---- asynchronus run ----
	cudaEvent_t start = get_time();
	cudaEvent_t copyEnd1, copyEnd2;
	cudaEventCreate(&copyEnd1); cudaEventCreate(&copyEnd2); 
	sim_freq<<<N_BLK, N_THRD, 0, s1>>>(d_freq1, d_prm1, N_SIMS);
	sim_freq<<<N_BLK, N_THRD, 0, s2>>>(d_freq2, d_prm2, N_SIMS);
	sim_severity<<<N_BLK, N_THRD, 0, s1>>>(d_loss1, d_freq1, d_prm1, N_SIMS);
	sim_severity<<<N_BLK, N_THRD, 0, s2>>>(d_loss2, d_freq2, d_prm2, N_SIMS); 
	cudaMemcpyAsync(h_loss1, d_loss1, N_BYTES_F, cudaMemcpyDeviceToHost, s1);
	cudaEventRecord(copyEnd1, s1);
	cudaMemcpyAsync(h_loss2, d_loss2, N_BYTES_F, cudaMemcpyDeviceToHost, s2);
	cudaEventRecord(copyEnd2, s2);
	cudaMemcpyAsync(h_freq1, d_freq1, N_BYTES_I, cudaMemcpyDeviceToHost, s1);
	cudaMemcpyAsync(h_freq2, d_freq2, N_BYTES_I, cudaMemcpyDeviceToHost, s2);
	cudaEventSynchronize(copyEnd1);
	mean1 = calcMean(h_loss1, N_SIMS);
	cudaEventSynchronize(copyEnd2);
	mean2 = calcMean(h_loss2, N_SIMS);
	cudaStreamSynchronize( s1 );
	cudaStreamSynchronize( s2 );
	cudaEvent_t stop = get_time(); // stop time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dur, start, stop);
	
	printf("\tasynchronously:\t loss1=%.3f, loss2=%.3f, %.3f ms taken, \n", 
			mean1, mean2, dur);
	
	
	// ---- synchronus run ----
	start = get_time();
	sim_freq<<<N_BLK, N_THRD>>>(d_freq1, d_prm1, N_SIMS);
	sim_severity<<<N_BLK, N_THRD>>>(d_loss1, d_freq1, d_prm1, N_SIMS);
	cudaMemcpy(h_freq1, d_freq1, N_BYTES_I, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_loss1, d_loss1, N_BYTES_F, cudaMemcpyDeviceToHost);
	sim_freq<<<N_BLK, N_THRD>>>(d_freq2, d_prm2, N_SIMS);
	sim_severity<<<N_BLK, N_THRD>>>(d_loss2, d_freq2, d_prm2, N_SIMS); 
	cudaMemcpy(h_freq2, d_freq2, N_BYTES_I, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_loss2, d_loss2, N_BYTES_F, cudaMemcpyDeviceToHost);
	mean1 = calcMean(h_loss1, N_SIMS);
	mean2 = calcMean(h_loss2, N_SIMS);
	stop = get_time(); // stop time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dur, start, stop);

	printf("\tsynchronously:\t loss1=%.3f, loss2=%.3f, %.3f ms taken, \n", 
			mean1, mean2, dur);
	
	return EXIT_SUCCESS;
}

