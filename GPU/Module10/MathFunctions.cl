__kernel void math_kernel(__global const float *x, __global const float *y,
						 __global int *results) {
	int gid = get_global_id(0);

	float y_hat = 2.0 * sin( pow(x[gid],3)/4.0 + 3.0 ) - 0.5;
	if (y_hat > 0) {
		results[gid] = (y[gid] < y_hat && y[gid]>0);
	} else {
		results[gid] = - (y[gid] > y_hat && y[gid] < 0);
	}
}