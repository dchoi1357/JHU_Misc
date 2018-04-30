__kernel void math_kernel(__global const float *x, __global const float *y,
						 __global float *results) {
							 
	int gid = get_global_id(0);
	results[gid] = x[gid] - y[gid];
}