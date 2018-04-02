#include <stdio.h>
#include <stdlib.h>

//#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

// look into cusparseXcsrgemmNnz 

unsigned int N_BYTES_MAT;

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for(int row = 0 ; row < m ; row++){
		for(int col = 0 ; col < n ; col++){
			double Areg = A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
		}
	}
}


int main(int argc, char*argv[]) {
	cusparseHandle_t cuSpHdl; // cuSparse handle
	cusparseCreate(&cuSpHdl);
	
/*  	| 2  0  0 |
	A = | 0  0 -1 |
		| 3 -2  0 |
		| 0  1  0 |
	
	x = (2.1034 2.7241 1.0000)'
	b = (4 -1 1 3)'
*/
    const int nrows = 4; // rows of matrix A
    const int ncols = 3; // rows of matrix A
	N_BYTES_MAT = sizeof(float)*nrows*ncols;
	
	// Generate host side dense matrix
	float h_denseA[nrows*ncols] = {2.0, 0.0, 3.0, 0.0,   0.0, 0.0, -2.0, 1.0, 
		0.0, -1.0, 0.0, 0.0};
	float *d_denseA;
	cudaMalloc((void**)&d_denseA, N_BYTES_MAT);
	cudaMemcpy(d_denseA, h_denseA, N_BYTES_MAT, cudaMemcpyHostToDevice);
	
	// Set descriptions of sparse matrix A
	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr(&descrA);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	
	int non0; // number of non-zero elements
	int *d_non0vec; // number of non-zero elements per row
	cudaMalloc(&d_non0vec, nrows*sizeof(int));
	
	cusparseSnnz(cuSpHdl, CUSPARSE_DIRECTION_ROW, nrows, ncols, descrA, 
				d_denseA, nrows, d_non0vec, &non0);
				
	
	// print number of non-zero elements per row
	int *h_non0vec; 
	h_non0vec = (int*) malloc(sizeof(int)*nrows);
	cudaMemcpy(h_non0vec, d_non0vec, sizeof(int)*nrows, cudaMemcpyDeviceToHost); 
	for (int i=0; i < nrows; i++) {
		printf("%u ", h_non0vec[i]);
	}
	printf("\n");
	printf("Non-zero: %u\n", non0);
	
	
	// Device side sparse matrix;
	float *d_sprsA;
	int *d_rowIdx, *d_colPtr;
	cudaMalloc(&d_sprsA, non0*sizeof(float));
	cudaMalloc(&d_rowIdx, (nrows+1) * sizeof(int));
	cudaMalloc(&d_colPtr, non0 * sizeof(int));
	cusparseSdense2csr(cuSpHdl, nrows, ncols, descrA, d_denseA, nrows, 
					d_non0vec, d_sprsA, d_rowIdx, d_colPtr);
	
	// Get sparse matrix on the host side
	float *h_sprsA;
	int *h_rowIdx, *h_colPtr;
	h_sprsA = (float*) malloc(non0 * sizeof(float));
	h_rowIdx = (int*) malloc( (nrows+1) * sizeof(int) );
	h_colPtr = (int*) malloc( non0 * sizeof(int) );
	cudaMemcpy(h_sprsA, d_sprsA, non0*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rowIdx, d_rowIdx, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_colPtr, d_colPtr, non0*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < non0; ++i) {
		printf("sparse_A[%i] = %.0f \n", i, h_sprsA[i]);
	} 
	for (int i = 0; i < (nrows + 1); ++i) {
		printf("row_idx[%i] = %i \n", i, h_rowIdx[i]);
	}
	for (int i = 0; i < non0; ++i) {\
		printf("col_ptr[%i] = %i \n", i, h_colPtr[i]);
	}
	
	// define b vector on host and device
	float h_b[nrows] = {4.0, -1.0, 1.0, 3.0};
	float *d_b;
	cudaMalloc(&d_b, nrows*sizeof(float));
	cudaMemcpy(d_b, h_b, nrows*sizeof(float), cudaMemcpyHostToDevice);
	
	// allocate solution vector
	float *d_x, *h_x;
	cudaMalloc(&d_x, ncols*sizeof(float));
	h_x = (float*) malloc(ncols*sizeof(float));
	
	
	// initialize cuSolver
	cusolverSpHandle_t cuSolvHdl;
    cusolverSpCreate(&cuSolvHdl);
	int *d_p, *h_p;
	cudaMalloc(&d_p, ncols*sizeof(int));
	h_p = (int*) malloc(ncols*sizeof(int));
	int rankA;
	float minNorm;
	
	/*
	cusolverSpScsrlsqvqr(cuSolvHdl, nrows, ncols, non0, descrA, d_sprsA, 
		d_rowIdx, d_colPtr, d_b, 1e-6, rankA, d_x, d_p, &minNorm);
	*/
	cusolverSpScsrlsqvqrHost(cuSolvHdl, nrows, ncols, non0, descrA, h_sprsA, 
		h_rowIdx, h_colPtr, h_b, 1e-6, &rankA, h_x, h_p, &minNorm);
	
	printf("Good!\n");
	return EXIT_SUCCESS;
}
