#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false
#define BUFF_SIZE 16
#define SUBBUFF_SIZE 4

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
	cl_context context;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	
	int platform = DEFAULT_PLATFORM; 
	bool useMap  = DEFAULT_USE_MAP;

	std::cout << "Simple buffer and sub-buffer Example" << std::endl;

	// First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 

	platformIDs = (cl_platform_id *)alloca(
			sizeof(cl_platform_id) * numPlatforms);

	std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	std::ifstream srcFile("simple.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	deviceIDs = NULL;
	DisplayPlatformInfo(
		platformIDs[platform], 
		CL_PLATFORM_VENDOR, 
		"CL_PLATFORM_VENDOR");

	errNum = clGetDeviceIDs(
		platformIDs[platform], 
		CL_DEVICE_TYPE_ALL, 
		0,
		NULL,
		&numDevices);
	if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	{
		checkErr(errNum, "clGetDeviceIDs");
	}       

	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
	errNum = clGetDeviceIDs(
		platformIDs[platform],
		CL_DEVICE_TYPE_ALL,
		numDevices, 
		&deviceIDs[0], 
		NULL);
	checkErr(errNum, "clGetDeviceIDs");

	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[platform],
		0
	};

	context = clCreateContext(
		contextProperties, 
		numDevices,
		deviceIDs, 
		NULL,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

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
		"-I.",
		NULL,
		NULL);
	if (errNum != CL_SUCCESS) {
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), 
			buildLog, 
			NULL);

			std::cerr << "Error in OpenCL C source: " << std::endl;
			std::cerr << buildLog;
			checkErr(errNum, "clBuildProgram");
	}

	// create buffers and sub-buffers
	int * inputs = new int[BUFF_SIZE];
	for (unsigned int i = 0; i < BUFF_SIZE; i++) {
		inputs[i] = i;
	}

	// create a single buffer to cover all the input data
	cl_mem buffer = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(int) * BUFF_SIZE,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer");
	buffers.push_back(buffer);
	
	// create a buffer for output
	cl_mem outSum = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * BUFF_SIZE/SUBBUFF_SIZE,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer");

	// now for all devices other than the first create a sub-buffer
	for (unsigned int i = 0; i < BUFF_SIZE/SUBBUFF_SIZE; i++) {
		cl_buffer_region region = 
			{
				i * SUBBUFF_SIZE * sizeof(int), 
				SUBBUFF_SIZE * sizeof(int)
			};
		buffer = clCreateSubBuffer(
			buffers[0],
			CL_MEM_READ_WRITE,
			CL_BUFFER_CREATE_TYPE_REGION,
			&region,
			&errNum);
		checkErr(errNum, "clCreateSubBuffer");

		buffers.push_back(buffer);
	}

	// Create command queues
	for (unsigned int i = 0; i < BUFF_SIZE/SUBBUFF_SIZE; i++) {
		cl_command_queue queue = clCreateCommandQueue(
									context,
									deviceIDs[0],
									0,
									&errNum);
		checkErr(errNum, "clCreateCommandQueue");

		queues.push_back(queue);

		cl_kernel kernel = clCreateKernel(
			program,
			"square",
			&errNum);
		checkErr(errNum, "clCreateKernel(square)");

		errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i+1]);
		errNum |= clSetKernelArg(kernel, 1, sizeof(int), &i);
		errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outSum);
		checkErr(errNum, "clSetKernelArg(square)");

		kernels.push_back(kernel);
	}

	// Write input data (only first buffer 16x16 needed)
	errNum = clEnqueueWriteBuffer(
		queues[0],
		buffers[0],
		CL_TRUE,
		0,
		sizeof(int) * BUFF_SIZE,
		(void*)inputs,
		0,
		NULL,
		NULL);

	std::vector<cl_event> events;
	// call kernel for each device
	for (unsigned int i = 0; i < queues.size(); i++) {
		cl_event event;
		size_t gWI = SUBBUFF_SIZE;

		errNum = clEnqueueNDRangeKernel(
			queues[i], 
			kernels[i], 
			1, 
			NULL,
			(const size_t*)&gWI, 
			(const size_t*)NULL, 
			0, 
			0, 
			&event);

		events.push_back(event);
	}

	// Technically don't need this as we are doing a blocking read
	// with in-order queue.
	clWaitForEvents(events.size(), &events[0]);
	
	float outputs[BUFF_SIZE/SUBBUFF_SIZE];	
	// Read back computed data
	clEnqueueReadBuffer(
		queues[0],
		outSum,
		CL_TRUE,
		0,
		sizeof(float) * BUFF_SIZE/SUBBUFF_SIZE,
		(void*)outputs,
		0,
		NULL,
		NULL);

	// Display output in rows
	for (unsigned i = 0; i < BUFF_SIZE/SUBBUFF_SIZE; i++) {
		std::cout << " " << outputs[i];
	}
	std::cout << std::endl;

	std::cout << "Program completed successfully" << std::endl;

	return 0;
}
