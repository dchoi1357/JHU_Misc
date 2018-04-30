 g++ MathFunctions.cpp -I/usr/local/cuda-8.0/targets/x86_64-linux/include/ \
	-lOpenCL -o MathFunc.exe

./MathFunc.exe 20 add
./MathFunc.exe 20 subtract
./MathFunc.exe 20 multiply
./MathFunc.exe 20 divide
./MathFunc.exe 20 power

rm *.exe