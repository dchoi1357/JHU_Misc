__kernel void convolve(
	const __global  uint * const input,
    __constant uint * const mask,
    __global  uint * const output,
    const int inputWidth,
    const int maskWidth) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
	
	const int outputWidth = 6;
	const int outRow = x/outputWidth;
	const int outCol = x%outputWidth;
	
    uint sum = 0;
	int inputRowIdx, inputIdx, maskIdx; 
    for (int r = 0; r < maskWidth; r++)    {
        inputRowIdx = (outRow+r)*inputWidth + outCol ;
		
        for (int c = 0; c < maskWidth; c++) {
			maskIdx = (r*maskWidth) + c;
			inputIdx = inputRowIdx + c;
			sum += mask[maskIdx] * input[inputIdx];
			if (x==13) {
				printf("mask=%u, rowIdx=%u, input=%u\n", maskIdx, inputRowIdx, inputIdx);
			}
        }
    }
	const int idx = y * get_global_size(0) + x;
	//printf("x=%u, y=%u, idx=%u, sum=%u\n", x, y, idx, sum);


	output[y * get_global_size(0) + x] = sum;
	//printf("x=%u, y=%u, ind=%u, res=%u\n", x, y, y*get_global_size(0)+x, sum);
}