nvcc squareArray.cu -Wno-deprecated-gpu-targets -o squareArray.out
./squareArray.out 64 4
./squareArray.out 20 10
./squareArray.out 1 50