fName="const_v_shared_test.cu"
exeName="test.exe"

nvcc $fName -Wno-deprecated-gpu-targets -o $exeName

./$exeName 5
./$exeName 10
./$exeName 12
./$exeName 15
./$exeName 18
./$exeName 20

rm -f $exeName
