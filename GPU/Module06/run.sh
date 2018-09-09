nvcc -Wno-deprecated-gpu-targets --ptxas-options=-v test_register.cu -o t.exe
echo ""

nSims=(500 2000 10000) 
nPeriods=(10 100 500)

for s in "${nSims[@]}"
do
	for p in "${nPeriods[@]}"
	do
		t.exe "$s" "$p"
	done
	#echo ""
done

rm -f t.exe 
