#include <stdio.h>
#include <stdlib.h>

const unsigned int N_FIB = 12;
const unsigned int N_BYTES_FIB = N_FIB * sizeof(unsigned int);
const unsigned int N_THREAD = N_FIB;
unsigned int N_ELEM, N_BYTES_ARR, N_BLOCKS;

__constant__ unsigned int fib_const[N_FIB];
__device__ static unsigned int fib_gmem[N_FIB];

/* Print array of integers, 20 elements per line */
void printArray(unsigned int * a, const char* name, const unsigned int n) {
	printf("Printing %s array:", name);
	for (int i = 0; i < n; i++) {
		if (i%20 == 0) {
			printf("\n");
		}
		printf("%5u", a[i]);
	}
	printf("\n");
}

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// Generate random integer array between 0 and 9
void genRandArray(unsigned int *arr, const unsigned int n) {
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % 10;
	}
}

// Generate fibonacci sequence
void genFibSequence(unsigned int* arr, int nMax) {
	arr[0] = 1;
	arr[1] = 1;
	
	for (int n=2; n < nMax; n++) {
		arr[n] = arr[n-1] + arr[n-2];
	}
}

__global__ void calc_w_const(unsigned int * const inArr, unsigned int *outArr, 
													unsigned int const N) {
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx < N) {
		outArr[idx] = 0; // initialize 0
		for (int i=(idx%2); i<N_FIB; i+=2) { // add up all even/odd elements
			outArr[idx] += fib_const[i];
		}
		outArr[idx] *= inArr[idx];
	}
}

__global__ void calc_w_gmem(unsigned int * const inArr, unsigned int *outArr, 
													unsigned int const N) {
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx < N) {
		outArr[idx] = 0; // initialize 0
		for (int i=(idx%2); i<N_FIB; i+=2) { // add up all even/odd elements
			outArr[idx] += fib_gmem[i];
		}
		outArr[idx] *= inArr[idx];
	}
}

__global__ void calc_w_shared(unsigned int * const inArr, unsigned int *outArr, 
							unsigned int const *inFib, unsigned int const N) {
	__shared__ unsigned int tmpSum[N_FIB]; // shared memory, size = nThreads
	const unsigned int tid = threadIdx.x;
	tmpSum[tid] = inFib[tid]; // initialize to same as fib sequence
	__syncthreads();
	
	if (tid < 2) { // add up all even or odd elements
		for (int s=tid+2; s<blockDim.x; s+=2) { // start from elem 2 or 3
			tmpSum[tid] += tmpSum[s];
		}
	}
	__syncthreads();
	
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < N) {
		outArr[idx] = inArr[idx] * tmpSum[idx%2]; // perform multiplication
	}
}

void validate_output(unsigned int *o1, unsigned int *o2, unsigned int *o3) {
	bool div = false;
	for (unsigned int n=0; n<N_ELEM && !div ; n++) {
		if ( (o1[n] != o2[n]) || (o2[n] != o3[n]) ) {
			div = true;
			printf("idx=%u: %u, %u, %u\n", n, o1[n], o2[n], o3[n]);
		}
	}
	if (!div) {
		printf("Results for all three runs are the identical!\n");
	} else {
		printf("Some results are different...\n");
	}
}


int main(int argc, char* argv[]) {
	if (argc != 2) { // expect 1 cmd line args: [arraySizeExponent]
		printf("Usage: %s [arraySizeExponent].\n", argv[0]);
		return EXIT_FAILURE;
	}
	unsigned int N_ELEM = 2 << atoi(argv[1]); // n input array
	unsigned int N_BYTES_ARR = N_ELEM * sizeof(unsigned int);
	unsigned int N_BLOCKS = (N_ELEM + N_THREAD-1) / N_THREAD;

	printf("Running with %u elements in input array...\n", N_ELEM);
	
	unsigned int *h_in, *h_out1, *h_out2, *h_out3, *h_fib;
	unsigned int *d_in, *d_out, *d_fib;
	h_fib = (unsigned int*) malloc(N_BYTES_FIB); // initialize fibonacci
	genFibSequence(h_fib, N_FIB); // generate fibonacci sequence
		
	h_in = (unsigned int*) malloc(N_BYTES_ARR); // allocate input
	genRandArray(h_in, N_ELEM); // generate random input array 
	h_out1 = (unsigned int*) malloc(N_BYTES_ARR); // allocate output 1
	h_out2 = (unsigned int*) malloc(N_BYTES_ARR); // allocate output 2
	h_out3 = (unsigned int*) malloc(N_BYTES_ARR); // allocate output 3
	
	cudaMallocHost((void**)&d_in, N_BYTES_ARR); // allocate input 
	cudaMallocHost((void**)&d_out, N_BYTES_ARR); // allocate output 
	cudaMallocHost((void**)&d_fib, N_BYTES_FIB); // allocate output 
	cudaMemcpy(d_in, h_in, N_BYTES_ARR, cudaMemcpyHostToDevice); // copy 2 dev
	
	// Timing using constant memory to store Fibonacci sequence
	cudaEvent_t start1 = get_time(); // start time 
	cudaMemcpyToSymbol(fib_const, h_fib, N_BYTES_FIB); // copy to constant
	calc_w_const<<<N_BLOCKS, N_THREAD>>>(d_in, d_out, N_ELEM);
	cudaMemcpy( h_out1, d_out, N_BYTES_ARR, cudaMemcpyDeviceToHost );
	cudaEvent_t stop1 = get_time(); // stop time
	cudaEventSynchronize(stop1);
	
	// Timing using global memory to store Fibonacci sequence
	cudaEvent_t start2 = get_time(); // start time 
	cudaMemcpyToSymbol(fib_gmem, h_fib, N_BYTES_FIB); // copy to global mem
	calc_w_gmem<<<N_BLOCKS, N_THREAD>>>(d_in, d_out, N_ELEM);
	cudaMemcpy( h_out2, d_out, N_BYTES_ARR, cudaMemcpyDeviceToHost );
	cudaEvent_t stop2 = get_time(); // stop time
	cudaEventSynchronize(stop2);
	
	// Timing using shared memory to smartly calculate
	cudaEvent_t start3 = get_time(); // start time 
	cudaMemcpy(d_fib, h_fib, N_BYTES_FIB, cudaMemcpyHostToDevice); //copy 2 dev
	calc_w_shared<<<N_BLOCKS, N_THREAD>>>(d_in, d_out, d_fib, N_ELEM);
	cudaMemcpy(h_out3, d_out, N_BYTES_ARR, cudaMemcpyDeviceToHost );
	cudaEvent_t stop3 = get_time(); // stop time
	cudaEventSynchronize(stop3);

	// Checking that the results of the three runs are identical
	validate_output(h_out1, h_out2, h_out3);
	
	// Free memory 
	cudaFree(d_in); cudaFree(d_out); cudaFree(d_fib);
	free(h_in); free(h_out1); free(h_out2); free(h_out3); free(h_fib); 
	
	// Printing time benchmarks
	float tmp = 0;
	cudaEventElapsedTime(&tmp, start1, stop1);
	printf("\tUsing constant memory: elapsed %.3f ms\n", tmp);
	cudaEventElapsedTime(&tmp, start2, stop2);
	printf("\tUsing global memory: elapsed %.3f ms\n", tmp);
	cudaEventElapsedTime(&tmp, start3, stop3);
	printf("\tUsing shared memory: elapsed %.3f ms\n", tmp);
	printf("\n");
}