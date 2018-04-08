#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

int source_vert;

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}
int main(int argc, char **argv) {
	int source_vert = 0;
	if (argc != 2) {
		printf("Usage: %s [sourceVertex]\n", argv[0]);
		return -1;
	} else {
		source_vert = atoi(argv[1]);
	}
		
	const size_t  n = 7, nnz = 12, vert_sets = 1, edge_sets = 1;
	void** vertex_dim;
	// nvgraph variables
	nvgraphStatus_t status;
	nvgraphHandle_t hdl;
	nvgraphGraphDescr_t grf;
	nvgraphCSCTopology32I_t CSC_input;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t* vertex_dimT;
	// Init host data
	float *h_sssp;
	h_sssp = (float*)malloc(n*sizeof(float));
	vertex_dim  = (void**)malloc(vert_sets*sizeof(void*));
	vertex_dimT = (cudaDataType_t*)malloc(vert_sets*sizeof(cudaDataType_t));
	CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
	vertex_dim[0]= (void*)h_sssp;
	vertex_dimT[0] = CUDA_R_32F;
	float h_weights[] = {2.0, 2.0, 2.0, 1.0, 2.0, 4.0, 1.0, 5.0, 3.0, 2.0, 6.0, 3.0};
	int h_destOffsets[] = {0, 1, 2, 5, 8, 9, 11, 12};
	int h_sourceIdxs[] = {1, 3, 0, 3, 6, 1, 2, 5, 3, 4, 5, 2};
	check(nvgraphCreate(&hdl));
	check(nvgraphCreateGraphDescr (hdl, &grf));
	CSC_input->nvertices = n; CSC_input->nedges = nnz;
	CSC_input->destination_offsets = h_destOffsets;
	CSC_input->source_indices = h_sourceIdxs;
	
	// Set graph connectivity and properties (tranfers)
	check(nvgraphSetGraphStructure(hdl, grf, (void*)CSC_input, NVGRAPH_CSC_32));
	check(nvgraphAllocateVertexData(hdl, grf, vert_sets, vertex_dimT));
	check(nvgraphAllocateEdgeData  (hdl, grf, edge_sets, &edge_dimT));
	check(nvgraphSetEdgeData(hdl, grf, (void*)h_weights, 0));
	
	// Solve
	check(nvgraphSssp(hdl, grf, 0,  &source_vert, 0));
	// Get and print result
	check(nvgraphGetVertexData(hdl, grf, (void*)h_sssp, 0));
	printf("Shortest distance from vertex %u to :\n", source_vert);
	for (int i=0; i<n; i++) {
		printf("\tvertex %u =\t%.1f\n", i, h_sssp[i]);
	}
	
	//Clean 
	free(h_sssp); free(vertex_dim);
	free(vertex_dimT); free(CSC_input);
	check(nvgraphDestroyGraphDescr(hdl, grf));
	check(nvgraphDestroy(hdl));
	return 0;
}