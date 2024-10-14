#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <OpenCL/opencl.h>

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " (" << err << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Function to read the kernel source code from a file
std::string readKernelSource(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file." << std::endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();  // Read the file into the buffer
    return buffer.str();     // Convert the buffer to a string
}

int main() {
    const int count = 1000;
    std::vector<int> output(count, 0);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem outputBuffer;
    cl_int err;

    // Get platform and device information
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERROR(err, "Failed to get platform ID");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_ERROR(err, "Failed to get device ID");

    // Create OpenCL context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERROR(err, "Failed to create OpenCL context");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "Failed to create command queue");

    // Read the kernel source from the file
    std::string kernelSource = readKernelSource("count_kernel.cl");
    const char* kernelSourceCStr = kernelSource.c_str();  // Convert string to C-style string

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, nullptr, &err);
    CHECK_ERROR(err, "Failed to create program with source");

    // Build the program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Error building program: " << log.data() << std::endl;
        exit(1);
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "count_kernel", &err);
    CHECK_ERROR(err, "Failed to create kernel");

    // Create a buffer to hold the output data
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count, nullptr, &err);
    CHECK_ERROR(err, "Failed to create output buffer");

    // Set the kernel argument (the output buffer)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    CHECK_ERROR(err, "Failed to set kernel argument");

    // Execute the kernel over the range of 10000 elements
    size_t globalSize = count;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to enqueue kernel");

    // Wait for the command queue to finish
    err = clFinish(queue);
    CHECK_ERROR(err, "Failed to finish queue");

    // Read back the results from the device
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(int) * count, output.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read buffer");

    // Print the results
    for (int i = 0; i < count; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
