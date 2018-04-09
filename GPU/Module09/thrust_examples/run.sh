nvcc -Wno-deprecated-gpu-targets thrust_saxpy.cu -o saxpy.exe;

./saxpy.exe 5
./saxpy.exe 10
./saxpy.exe 15
./saxpy.exe 20

rm *.exe