g++ async_N.cpp -lOpenCL -o simple.exe \
	-I/usr/local/cuda-8.0/targets/x86_64-linux/include/
	
# -Wno-deprecated-declarations
	
./simple.exe 1
#rm *.exe