#include <stdio.h>
#include <cstdlib>

__global__
/* Kernel to square array on GPU */
void squareArray(float *input, float *result, unsigned int n) {
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < n) { 
		result[idx] = input[idx] * input[idx];
	}
}

unsigned int nIter; // number of iterations
unsigned int nElem, nBytes; // dim and mem size of array, 
unsigned int nBlocks; // number of blocks per grid
unsigned int nThreads = 64;

/* Print array of integers, 20 elements per line */
void printArray(const float * a, const char* name) {
	printf("Printing %s array:", name);
	for (int i = 0; i < nElem; i++) {
		if (i%20 == 0) {
			printf("\n");
		}
		printf("%2f", a[i]);
	}
	printf("\n");
}

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

void genRandArray(float *arr, const unsigned int n) {
	for (int i = 0; i < n; i++) {
		arr[i] = rand() / (RAND_MAX / 5.0);
	}
}


float bchmkPagedMem() {
	cudaEvent_t start = get_time(); // start time
	
	float *h_pagedIn, *h_pagedOut; // paged memory on host
	float *d_input, *d_result; // pointers for GPU arrays and allocate
	cudaMalloc((void **)&d_input, nBytes); // device input
	cudaMalloc((void **)&d_result, nBytes); // device output
	
	h_pagedIn = (float*) malloc(nBytes); // allocate input 
	h_pagedOut = (float*) malloc(nBytes); // allocate output 
	
	for (int it=0; it < nIter; it++) {
		genRandArray(h_pagedIn, nElem); // overwrite with random numbers
		cudaMemcpy(d_input, h_pagedIn, nBytes, cudaMemcpyHostToDevice); //copy
		
		// Execute the kernel
		squareArray<<<nBlocks, nThreads>>>(d_input, d_result, nElem);
		
		// Free the arrays on the GPU as now we're done with them 
		cudaMemcpy( h_pagedOut, d_result, nBytes, cudaMemcpyDeviceToHost );
	}

	cudaFree(d_input); // Freeing memory 
	cudaFree(d_result);
	free(h_pagedIn);
	free(h_pagedOut);
	
	cudaEvent_t stop = get_time(); // stop time 
	cudaEventSynchronize(stop); 
	
	float dur = 0;
	cudaEventElapsedTime(&dur, start, stop); // calculate duration
	return dur;
}

float bchmkPinnedMem() {
	cudaEvent_t start = get_time(); // start time
	
	float *h_pinnedIn, *h_pinnedOut; // paged memory on host
	float *d_input, *d_result; // pointers for GPU arrays and allocate
	cudaMalloc((void **)&d_input, nBytes); // device input
	cudaMalloc((void **)&d_result, nBytes); // device output
	
	cudaMallocHost((void**)&h_pinnedIn, nBytes); // allocate input 
	cudaMallocHost((void**)&h_pinnedOut, nBytes); // allocate output 
	
	for (int it=0; it < nIter; it++) {
		genRandArray(h_pinnedIn, nElem); // overwrite with random numbers
		cudaMemcpy(d_input, h_pinnedIn, nBytes, cudaMemcpyHostToDevice); //copy
		
		// Execute the kernel
		squareArray<<<nBlocks, nThreads>>>(d_input, d_result, nElem);
		
		// Free the arrays on the GPU as now we're done with them 
		cudaMemcpy( h_pinnedOut, d_result, nBytes, cudaMemcpyDeviceToHost );
	}
	
	cudaFree(d_input); // freeing memory
	cudaFree(d_result);
	cudaFreeHost(h_pinnedIn);
	cudaFreeHost(h_pinnedOut);
	
	cudaEvent_t stop = get_time(); // stop time 
	cudaEventSynchronize(stop); 
	
	float dur = 0;
	cudaEventElapsedTime(&dur, start, stop); // calculate duration
	return dur;
}

int main(int argc, char* argv[]) {
	if (argc != 3) { // expect 2 cmd line args: nThreads and nBlocks
		printf("Usage: %s [nIterations] [arraySizeExponent].\n", argv[0]);
		return EXIT_FAILURE;
	}
	nIter = atoi(argv[1]);
	nElem = 2 << atoi(argv[2]); // number of elem in array
	nBytes = nElem * sizeof(float); // size of array 
	nBlocks = (nBytes+nThreads-1)/nThreads;
	
	printf("Running %u iterations of squaring %u doubles" 
		" using pageable memory...\n", nIter, nElem);
	float t1 = bchmkPagedMem();
	printf("\tElapsed: %f ms.\n", t1);
	
	printf("Running %u iterations of squaring %u doubles" 
		" using pinned memory...\n", nIter, nElem);
	float t2 = bchmkPinnedMem();
	printf("\tElapsed: %f ms.\n", t2);
	
	return EXIT_SUCCESS;
}