#nvcc estimate_pi.cu -o pi.exe
#./pi.exe 500 100
#./pi.exe 1000 100
#./pi.exe 5000 100
#./pi.exe 500 1000
#./pi.exe 1000 1000
#./pi.exe 5000 1000

rm -f *.exe

nvcc test.cu -lcudart -lcuda -lcusparse -lcusolver -lcublas -I common/inc -o t.exe

./t.exe 
