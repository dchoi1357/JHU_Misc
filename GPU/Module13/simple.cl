__kernel void infSum(__global double * input, int N, __global double * out) {
	size_t id = get_global_id(0);
	
	double base = input[id]; 
	double tmpOut = 0;
	for (int i=0; i < N; i++) { // N summation of base taken to i-th power
		tmpOut = tmpOut + pown(base, i);
	}
	
	// printf("ID=%u, base=%f, Out=%f\n", id, base, tmpOut);
	out[id] = tmpOut;
}