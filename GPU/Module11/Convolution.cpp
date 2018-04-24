#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;
cl_uint inputSignal[inputSignalWidth][inputSignalHeight];

const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

cl_uint outputSignal[outputSignalWidth][outputSignalHeight];
cl_float outputNormed[outputSignalWidth][outputSignalHeight];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;
cl_uint mask[maskWidth][maskHeight] =
{
	{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 2, 1, 0, 0}, {0, 1, 2, 3, 2, 1, 0},
	{1, 2, 3, 4, 3, 2, 1}, {0, 1, 2, 3, 2, 1, 0}, {0, 0, 1, 2, 1, 0, 0},
	{0, 0, 0, 1, 0, 0, 0}
};
const float normalizingConst = 44.0;

template <class typ, unsigned int rows, unsigned int cols>
void printArray(typ (&a)[rows][cols]) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j)
			std::cout << a[i][j] << " ";
		std::cout << std::endl;
	}
}

//////////////////////////////////////////////
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

///
//	main() for Convoloution example
//
int main(int argc, char** argv) {
	// allocate 0 < x < 9 for all elements
	for (int i=0; i<inputSignalHeight; i++) {
		for (int j=0; j<inputSignalWidth; j++) {
			inputSignal[i][j] = rand() % 10;
		}
	}
	std::cout << "The generated input array: " << std::endl;
	printArray<cl_uint, inputSignalHeight, inputSignalWidth>(inputSignal);
	std::cout << std::endl;

	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
	cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	// First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 

	platformIDs = (cl_platform_id *)alloca(
			sizeof(cl_platform_id) * numPlatforms);

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
			platformIDs[i], 
			CL_DEVICE_TYPE_GPU, 
			0,
			NULL,
			&numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
		{
			checkErr(errNum, "clGetDeviceIDs");
		}
		else if (numDevices > 0) 
		{
			deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

    // Next, create an OpenCL context on the selected platform.  
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[i],
		0
	};
	context = clCreateContext(
		contextProperties, 
		numDevices,
		deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), 
			buildLog, 
			NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight * maskWidth,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &outputSignalWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[1] = { outputSignalWidth * outputSignalHeight };
	const size_t localWorkSize[1]  = { 1 };

	// Queue the kernel up for execution across the array
	errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		1, 
		NULL,
		globalWorkSize, 
		localWorkSize,
		0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");

	errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
		0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth, 
		outputSignal,
		0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

	// Output the raw result buffer
	std::cout << "The raw output array " << std::endl;
	printArray<cl_uint, outputSignalHeight, outputSignalWidth>(outputSignal);
	std::cout << std::endl;
	
	std::cout << "The normalized output array " << std::endl;
	for (int i=0; i < outputSignalHeight; i++) {
		for (int j=0; j < outputSignalWidth; j++) {
			outputNormed[i][j] = outputSignal[i][j] / normalizingConst;
		}
	}
	printArray<float, outputSignalHeight, outputSignalWidth>(outputNormed);
	std::cout << std::endl;

	std::cout << std::endl << "Executed program succesfully." << std::endl;
	return 0;
}
