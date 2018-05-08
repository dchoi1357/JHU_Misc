g++ async_N.cpp -lOpenCL -o simple.exe \
	-I/usr/local/cuda-8.0/targets/x86_64-linux/include/

	
./simple.exe 1
./simple.exe 2
./simple.exe 5
./simple.exe 10
./simple.exe 2 4 6 8 10
./simple.exe 1 5 10 15 20


rm *.exe