#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#define DEF_SIMS 20
#define X0 1.0342

unsigned int N_SIMS, N_BYTES;

int main(int argc, char* argv[]) {
	if (argc > 1) { // no command line inputs
		N_SIMS = atoi(argv[1]);
	} else {
		N_SIMS = DEF_SIMS;
	}
	
	N_BYTES = N_SIMS * sizeof(float);
	float *d_X;
	cudaMalloc((void **)&d_X, N_BYTES); // device input
	
}