 g++ MathFunctions.cpp -I/usr/local/cuda-8.0/targets/x86_64-linux/include/ \
	-lOpenCL -o MathFunc.exe

./MathFunc.exe 100
./MathFunc.exe 1000
./MathFunc.exe 10000
./MathFunc.exe 100000

rm *.exe