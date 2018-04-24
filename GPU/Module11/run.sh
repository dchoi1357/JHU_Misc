g++ Convolution.cpp -I/usr/local/cuda-8.0/targets/x86_64-linux/include \
	-lOpenCL -o Convolution
	
./Convolution 49 49 1 > output_49x49.txt
./Convolution 15 15 1
./Convolution 40 15 1
./Convolution 50 50 0
./Convolution 100 100 0
./Convolution 500 500 0

rm Convolution