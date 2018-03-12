#include <stdio.h>
#include <cstdlib>

__global__
/* Kernel to square array on GPU */
void squareArray(unsigned int *input, unsigned int *result) {
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[idx] = input[idx] * input[idx];
}

unsigned int ARRAY_SIZE, ARRAY_BYTES;

/* Print array of integers, 20 elements per line */
void printArray(const unsigned int * a, const char* name) {
	printf("Printing %s array:", name);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i%20 == 0) {
			printf("\n");
		}
		printf("%4u", a[i]);
	}
	printf("\n");
}

int main(int argc, char* argv[]) {
	if (argc != 3) { // expect 2 cmd line args: nThreads and nBlocks
		printf("Usage: %s [nThreadsPerBlock] [nBlocks].\n", argv[0]);
		return EXIT_FAILURE;
	}
	const unsigned int nThreads = atoi(argv[1]); // parse cmd line inputs
	const unsigned int nBlocks = atoi(argv[2]);
	
	printf("Running %u threads per block, %u total blocks, ", nThreads, nBlocks);
	printf("array size of %u\n", nThreads * nBlocks);
	
	// Calculate array size etc and allocate host memory
	ARRAY_SIZE = nThreads * nBlocks;
	ARRAY_BYTES = (sizeof(unsigned int) * (ARRAY_SIZE));
	unsigned int cpu_result[ARRAY_SIZE]; // result of calculation

	// allocate and generate input array
	unsigned int inputArray[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		inputArray[i] = rand() % 10; // rand int in range of 0 to 9
	}

	// Declare pointers for GPU arrays, allocate + copy
	unsigned int *gpu_input, *gpu_result;
	cudaMalloc((void **)&gpu_result, ARRAY_BYTES);
	cudaMalloc((void **)&gpu_input, ARRAY_BYTES);
	cudaMemcpy( gpu_input, inputArray, ARRAY_BYTES, cudaMemcpyHostToDevice );

	// Execute kernel
	squareArray<<<nBlocks, nThreads>>>(gpu_input, gpu_result);

	// Free the arrays on the GPU as now we're done with them 
	cudaMemcpy( cpu_result, gpu_result, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_input);
	cudaFree(gpu_result);

	// Iterate through the arrays and print 
	printArray(inputArray, "input");
	printArray(cpu_result, "output");
 	printf("\n");
	return EXIT_SUCCESS;
}