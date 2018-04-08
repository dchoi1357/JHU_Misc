cd ./npp_examples
make
cd ../nvgraph_examples
make

cd ../npp_examples
./nppExamples oswald.pgm 1
./nppExamples oswald.pgm 2
./nppExamples oswald.pgm 3
rm nppExamples nppExamples.o


cd ../nvgraph_examples
verts=(0 1 2 3 4 5 6)
for v in "${verts[@]}"
do
	./nvgraph_SSSP "$v"
done
rm nvgraph_SSSP nvgraph_SSSP.o
