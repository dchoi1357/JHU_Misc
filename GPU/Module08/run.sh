nvcc estimate_pi.cu -Wno-deprecated-gpu-targets -o pi.exe
echo "===Estimating pi using Monte Carlo simulation==="
./pi.exe 500 100
./pi.exe 1000 100
./pi.exe 5000 100
./pi.exe 500 1000
./pi.exe 1000 1000
./pi.exe 5000 1000


echo ""
echo "===Multiplying sparse matrix with vector==="
nvcc sparse_mm.cu -Wno-deprecated-gpu-targets -lcudart -lcuda -lcusparse \
	-lcusolver -lcublas -I common/inc -o s.exe

./s.exe 
rm -f *.exe
