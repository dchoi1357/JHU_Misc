nvcc mem_bchmk.cu -Wno-deprecated-gpu-targets -o bch.exe

shifts=23

./bch.exe 1 $shifts
./bch.exe 2 $shifts
./bch.exe 5 $shifts
./bch.exe 10 $shifts
./bch.exe 25 $shifts
