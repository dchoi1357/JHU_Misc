__kernel void square(__global int * buffer, int N, __global float * out) {
	size_t id = get_global_id(0);

	if (id==0) {
		int tmp = 0;
		for (int i=0; i<get_global_size(0); i++) {
			tmp += buffer[id+i];
			printf("N=%d, i=%d, b=%d\n", N, i, buffer[id+i]);
		}
		out[N] = (float)tmp / get_global_size(0);
		
		printf("N=%d, out=%f\n", N, out[N]);
	}
}