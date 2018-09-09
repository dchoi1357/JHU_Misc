#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <ctime>

size_t N;

__host__ static __inline__ float rand_abs5() {
    return ((rand()%11) - 5);
}

struct saxpy_func {
    const int a;
    saxpy_func(int _a) : a(_a) {}

    __host__ __device__
        int operator()(const int& x, const int& y) const { 
            return a * x + y;
        }
};

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Usage: %s [nElementExp]\n", argv[0]);
		return -1;
	} else {
		N = 2 << atoi(argv[1])-1;
	}
	printf("Running SAXPY on two random vectors of %zu elements\n", N);
	
	// generate host vector and fill with rand number between -5 and 5
	thrust::host_vector<int> h_x(N);
	thrust::host_vector<int> h_y(N);
	thrust::generate(h_x.begin(), h_x.end(), rand_abs5);
	thrust::generate(h_y.begin(), h_y.end(), rand_abs5);
	
	// create device vectors
	thrust::device_vector<int> d_x = h_x;
	thrust::device_vector<int> d_y = h_y;
	//thrust::host_vector<int> h_res(N);
	
	// y = 2*x + y, on both device and host
	clock_t start = std::clock();
	thrust::transform(h_x.begin(), h_x.end(), 
					h_y.begin(), h_y.begin(), saxpy_func(2));
	clock_t stop = std::clock();
	double host_dur = double(stop-start) / (CLOCKS_PER_SEC/1000);
	
	start = std::clock();
	thrust::transform(d_x.begin(), d_x.end(), 
					d_y.begin(), d_y.begin(), saxpy_func(2));
	stop = std::clock();
	double dev_dur = double(stop-start) / (CLOCKS_PER_SEC/1000);

	printf("\tHost vector operation took %f ms\n", host_dur);
	printf("\tDevice vector operation took %f ms\n", dev_dur);
	
	bool allSame = true;
	size_t i;
	for (i=0; i < N; i++) {
		if ( (d_y[i] - h_y[i]) != 0 ) { 
			allSame = false;
			break;
		}
	}

	if (allSame) {
		printf("\tOperation on device and host vector produced same results\n");
		return 0;
	} else {
		printf("Element %zu not the same...\n", i);
		return -1;
	}
}
