g++ simple_subbuffer.cpp -lOpenCL -o simple.exe \
	-I/usr/local/cuda-8.0/targets/x86_64-linux/include/
	
./simple.exe
rm *.exe