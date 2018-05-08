#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define ARR_SIZE 10

unsigned int nEvents;
std::vector<int> eventList;

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void printArray(const double * a) {
	for (int i = 0; i < ARR_SIZE; i++) {
		if (i%10 == 0) {
			printf("\n");
		}
		printf("%.10f ", a[i]);
	}
	printf("\n");
}


double run_events(){	
	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
	cl_context context;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_mem> outputs;
	std::vector<cl_event> events;
	std::vector<int> nSums;
	
	// First, select an OpenCL platform to run on.  
	int platform = DEFAULT_PLATFORM;
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
	platformIDs = (cl_platform_id *)alloca(
			sizeof(cl_platform_id) * numPlatforms);
	//std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 
	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");
	
	// Read in Kernel file
	std::ifstream srcFile("simple.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");
	std::string srcProg( std::istreambuf_iterator<char>(srcFile), 
		(std::istreambuf_iterator<char>()) );
	const char * src = srcProg.c_str();
	size_t length = srcProg.length();
	
	// Get device ID
	deviceIDs = NULL;
	errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, 0,
		NULL, &numDevices);
	if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND){
		checkErr(errNum, "clGetDeviceIDs");
	}
	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
	errNum = clGetDeviceIDs( platformIDs[platform], CL_DEVICE_TYPE_ALL,
		numDevices, &deviceIDs[0], NULL);
	checkErr(errNum, "clGetDeviceIDs");
	
	// Get and create context
	cl_context_properties contextProperties[] = {
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
	program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");
	
	// Build program
	errNum = clBuildProgram(program, numDevices, deviceIDs, "-I.", NULL, NULL);
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
	
	// create input buffers
	double * inputs = new double[ARR_SIZE];
	for (size_t i = 0; i < ARR_SIZE; i++) {
		inputs[i] = (1.0) / (i%10 + 1);
	}
	cl_mem inBuffer = clCreateBuffer(context, 
						CL_MEM_READ_ONLY,
						sizeof(double) * ARR_SIZE,
						NULL,
						&errNum);
	checkErr(errNum, "clCreateBuffer");
	
	// create output buffers for each event 
	for (size_t i = 0; i < nEvents; i++) {
		cl_mem tmpOut = clCreateBuffer(context, 
							CL_MEM_WRITE_ONLY,
							sizeof(double) * ARR_SIZE,
							NULL,
							&errNum);
		checkErr(errNum, "clCreateBuffer");
		outputs.push_back(tmpOut);
	}
	
	// Create command queue for all events 
	cl_command_queue queue = clCreateCommandQueue(
								context,
								deviceIDs[0],
								CL_QUEUE_PROFILING_ENABLE,
								&errNum);
	checkErr(errNum, "clCreateCommandQueue");
	
	// Create kernels
	for (size_t i = 0; i < nEvents; i++) {
		cl_kernel kernel = clCreateKernel(
			program,
			"infSum",
			&errNum);
		checkErr(errNum, "clCreateKernel(infSum)");
		
		int t = eventList[i];
		errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuffer);
		errNum |= clSetKernelArg(kernel, 1, sizeof(int), &t);
		errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputs[i]);
		checkErr(errNum, "clSetKernelArg(infSum)");
		
		nSums.push_back(t);
		kernels.push_back(kernel);
	}
	
	// Write input data (only first buffer needed)
	errNum = clEnqueueWriteBuffer(
		queue,
		inBuffer,
		CL_TRUE,
		0,
		sizeof(double) * ARR_SIZE,
		(void*)inputs,
		0,
		NULL,
		NULL);
	
	// use a event for each run
	for (size_t i = 0; i < nEvents; i++) {
		cl_event event;
		size_t gWI = ARR_SIZE;

		errNum = clEnqueueNDRangeKernel(
			queue, 
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
	
	// Wait for events and calculate time elapsed
	clWaitForEvents(events.size(), &events[0]);
	clFinish(queue); // wait for all events to finish
	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, 
						sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(events[events.size()-1], CL_PROFILING_COMMAND_END,
						sizeof(time_end), &time_end, NULL);
	double elapsed = time_end-time_start;
	
	// Print output
	double output[ARR_SIZE];
	for (size_t i = 0; i < nEvents; i++) {
		errNum = clEnqueueReadBuffer(
			queue,
			outputs[i],
			CL_TRUE,
			0,
			sizeof(double) * ARR_SIZE,
			(void*)output,
			0,
			NULL,
			NULL);
		
		printf("Printing array with %u sums:", nSums[i]);
		printArray(output);
	}
	return elapsed;
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv) {
	if (argc == 1) {
		printf("Usage: %s [list of events]\n", argv[0]);
		return 1;
	} else {
		nEvents = argc - 1;
		for (int i=1; i < argc; i++) {
			eventList.push_back( atoi(argv[i]) );
		}
	}
			
	printf("Running %u events...\n", nEvents);
	double timed = run_events();
	printf("Program completed successfully in %.3f ms.\n\n", timed);
	return 0;
}
