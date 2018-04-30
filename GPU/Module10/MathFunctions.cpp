#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <cstring>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
size_t ARRAY_SIZE;
std::string ops = "";

void printArray(const float * a, const char* name) {
	printf("\tPrinting %s array:", name);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i%10 == 0) {
			printf("\n");
		}
		printf("%8.3f", a[i]);
	}
	printf("\n");
}

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext() {
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
									  NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
										  NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete [] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], 
		CL_QUEUE_PROFILING_ENABLE, NULL);
	if (commandQueue == NULL)
	{
		delete [] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete [] devices;
	return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
										(const char**)&srcStr,
										NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
							  sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input), b (input), c (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
					float *a, float *b) {
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								   sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								   sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
								   sizeof(float) * ARRAY_SIZE, NULL, NULL);

	if (memObjects[0]==NULL || memObjects[1]==NULL || memObjects[2]==NULL) {
		std::cerr << "Error creating memory objects." << std::endl;
		return false;
	}

	return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3]) {
	for (int i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (context != 0)
		clReleaseContext(context);

}

//	main() for MathFunctions
int main(int argc, char** argv) {
	if (argc != 3) {
		printf("Usage: %s [nSims] [add/subtract/multipl/divide/power]\n", argv[0]);
		return 1;
	} else {
		ARRAY_SIZE = atoi(argv[1]);
		ops = ops + argv[2] + ".cl";
	}
	printf("Performing %s, %zu elements...\n", ops.c_str(), ARRAY_SIZE);
	
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0};
	cl_int errNum;

	// Create an OpenCL context on first available platform
	context = CreateContext();
	if (context == NULL) {
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return 1;
	}

	// Create a command-queue on the first device available
	// on the created context
	commandQueue = CreateCommandQueue(context, &device);
	if (commandQueue == NULL) {
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL program from MathFunctions.cl kernel source
	program = CreateProgram(context, device, ops.c_str());
	if (program == NULL) {
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "math_kernel", NULL);
	if (kernel == NULL) {
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create memory objects that will be used as arguments to
	// kernel.  First create host memory arrays that will be
	// used to store the arguments to the kernel
	float result[ARRAY_SIZE];
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		a[i] = ((float) rand()) / RAND_MAX * 6.0 - 3.0; // -3 < a < 3
		b[i] = ((float) rand()) / RAND_MAX * 6.0 - 3.0; // -3 < b < 3
	}

	if (!CreateMemObjects(context, memObjects, a, b)) {
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Set the kernel arguments (result, a, b)
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	if (errNum != CL_SUCCESS) {
		std::cerr << "Error setting kernel arguments." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };

	// Queue the kernel up for execution across the array
	cl_event event;
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
									globalWorkSize, localWorkSize,
									0, NULL, &event);
	if (errNum != CL_SUCCESS){
		std::cerr << "Error queuing kernel for execution." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
	clWaitForEvents(1, &event);
	clFinish(commandQueue);

	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
								 0, ARRAY_SIZE * sizeof(float), result,
								 0, NULL, NULL);
	if (errNum != CL_SUCCESS) {
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Calculate time elapsed from profile
	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
							sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
							sizeof(time_end), &time_end, NULL);
	double elapsed = time_end-time_start;

	// Print result
	printArray(a, "Input 1");
	printArray(b, "Input 2");
	printArray(result, "results");

	
	printf("Array size %zu , elapsed %f ms\n\n", ARRAY_SIZE, elapsed/1000000.0);
	
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return 0;
}
