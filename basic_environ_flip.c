////////////////////////////////////////////////////////////////////
// File: basic_environ.c
//
// Description: base file for environment exercises with openCL
//
//
////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef _APPLE_
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define cimg_use_jpeg
#include <iostream>
#include "/home/a756976/Documents/PACS/PACS-lab5/CImg/CImg.h" // Path to CImg.h
using namespace cimg_library;

// check error, in such a case, it exits

void cl_error(cl_int code, const char *string)
{
  if (code != CL_SUCCESS)
  {
    printf("%d - %s\n", code, string);
    exit(-1);
  }
}
////////////////////////////////////////////////////////////////////////////////
// program for flipping and image
int main(int argc, char **argv)
{
  if (argc != 5)
  {
    printf("Usage: %s <image_path> <output_path> <width> <height>\n", argv[0]);
  }

  const char *input_image_path = argv[1];
  const char *output_image_path = argv[2];
  int width = atoi(argv[3]);
  int height = atoi(argv[4]);
  // Calculate the total number of pixels in the image
  int count = width * height;

  int err;                // error code returned from api calls
  size_t t_buf = 50;      // size of str_buffer
  char str_buffer[t_buf]; // auxiliary buffer
  size_t e_buf;           // effective size of str_buffer in use

  // size_t global_size[2]; // global domain size for our calculation
  // size_t local_size[2];  // local domain size for our calculation

  const cl_uint num_platforms_ids = 10;                         // max of allocatable platforms
  cl_platform_id platforms_ids[num_platforms_ids];              // array of platforms
  cl_uint n_platforms;                                          // effective number of platforms in use
  const cl_uint num_devices_ids = 10;                           // max of allocatable devices
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids]; // array of devices
  cl_uint n_devices[num_platforms_ids];                         // effective number of devices in use for each platform

  cl_device_id device_id;         // compute device id
  cl_context context;             // compute context
  cl_command_queue command_queue; // compute command queue

  // 1. Scan the available platforms:
  err = clGetPlatformIDs(num_platforms_ids, platforms_ids, &n_platforms);
  cl_error(err, "Error: Failed to Scan for Platforms IDs");
  printf("Number of available platforms: %d\n\n", n_platforms);

  for (int i = 0; i < n_platforms; i++)
  {
    err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
    // cl_int clGetDeviceInfo (cl_device_id device, cl_device_info param_name,
    //                         size_t param_value_size, void *param_value, size_t *param_value_size_ret)
    // param_name: CL_DEVICE_ADDRESS_BITS CL_DEVICE_COMPILER_AVAILABLE,�~@�
    cl_error(err, "Error: Failed to get info of the platform\n");
    printf("\t[%d]-Platform Name: %s\n", i, str_buffer);
  }
  printf("\n");
  // **Task**: print on the screen the name, host_timer_resolution, vendor, versionm, ...

  // 2. Scan for devices in each platform
  for (int i = 0; i < n_platforms; i++)
  {
    err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
    // cl_int clGetDeviceIDs (cl_platform_id platform, cl_device_type device_type,
    //                        cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices)
    //  device_type: CL_DEVICE_TYPE_{ACCELERATOR, ALL, CPU},
    cl_error(err, "Error: Failed to Scan for Devices IDs");
    printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

    for (int j = 0; j < n_devices[i]; j++)
    {
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device name");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j, str_buffer);

      cl_uint max_compute_units_available;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
    }
  }
  // **Task**: print on the screen the cache size, global mem size, local memsize, max work group size, profiling timer resolution and ... of each device
  // print the cache size
  for (int i = 0; i < n_platforms; i++)
  {
    for (int j = 0; j < n_devices[i]; j++)
    {
      cl_ulong cache_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cache_size), &cache_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device cache size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %lu\n\n", i, j, cache_size);
    }
  }

  // print the global mem size
  for (int i = 0; i < n_platforms; i++)
  {
    for (int j = 0; j < n_devices[i]; j++)
    {
      cl_ulong global_mem_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device global mem size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_GLOBAL_MEM_SIZE: %lu\n\n", i, j, global_mem_size);
    }
  }

  // print the local mem size
  for (int i = 0; i < n_platforms; i++)
  {
    for (int j = 0; j < n_devices[i]; j++)
    {
      cl_ulong local_mem_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device local mem size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_LOCAL_MEM_SIZE: %lu\n\n", i, j, local_mem_size);
    }
  }

  // print the max work group size
  for (int i = 0; i < n_platforms; i++)
  {
    for (int j = 0; j < n_devices[i]; j++)
    {
      size_t max_work_group_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max work group size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n\n", i, j, max_work_group_size);
    }
  }

  // print the profiling timer resolution
  for (int i = 0; i < n_platforms; i++)
  {
    for (int j = 0; j < n_devices[i]; j++)
    {
      size_t profiling_timer_resolution;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(profiling_timer_resolution), &profiling_timer_resolution, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device profiling timer resolution");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_PROFILING_TIMER_RESOLUTION: %lu\n\n", i, j, profiling_timer_resolution);
    }
  }

  // Read the image using CImg
  CImg<unsigned char> img(width, height, 1, 3); // Create an empty image with the specified width and height
  img.load_jpeg(input_image_path);              // Load the image from the specified file

  // 3. Create a context, with a device
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
  context = clCreateContext(properties, n_devices[0], devices_ids[0], NULL, NULL, &err);
  // cl_context clCreateContext ( const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (CL_CALLBACK*pfn_notify) (const char *errinfo, const void *private_info, size_t cb, void *user_data), void *user_data, cl_int *errcode_ret)
  cl_error(err, "Failed to create a compute context\n");

  // 4. Create a command queue
  cl_command_queue_properties proprt[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
  // cl_command_queue clCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_command_queue_properties *properties, cl_int *errcode_ret)
  cl_error(err, "Failed to create a command queue\n");

  // Calculate size of the file
  FILE *fileHandler = fopen("kernel_flip.cl", "r"); // put path to kernel.cl
  // FILE *fopen(const char *filename, const char *mode)
  fseek(fileHandler, 0, SEEK_END);
  size_t fileSize = ftell(fileHandler);
  rewind(fileHandler);

  // read kernel source into buffer
  char *sourceCode = (char *)malloc(fileSize + 1);
  sourceCode[fileSize] = '\0';
  fread(sourceCode, sizeof(char), fileSize, fileHandler);
  fclose(fileHandler);

  // create program from buffer

  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &fileSize, &err);
  // cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
  cl_error(err, "Failed to create program with source\n");
  free(sourceCode);

  // Build the executable and check errors
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];

    printf("Error: Some error at building process.\n");
    clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    // cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
    printf("%s\n", buffer);
    exit(-1);
  }

  // Create the kernel
  cl_kernel kernel = clCreateKernel(program, "image_flip", &err);

  // cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)
  cl_error(err, "Failed to create kernel from the program\n");

  // Create the input and output arrays in device memory for our calculation
  // CImg<unsigned char> img("image.jpg");
  // int count = img.width() * img.height();
  // float *in_host_object = img.data();
  // float out_host_object = (float) malloc(sizeof(float) * count);

  // Create memory buffers for the image data
  cl_mem in_image_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * width * height * 3, img.data(), &err);
  cl_error(err, "Failed to create memory buffer at device\n");
  cl_mem out_image_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height * 3, NULL, &err);
  cl_error(err, "Failed to create memory buffer at device\n");

  // Copy the image data to the memory buffer (from host to device)
  // Write date into the memory object
  err = clEnqueueWriteBuffer(command_queue, in_image_buffer, CL_TRUE, 0, sizeof(unsigned char) * width * height * 3,
                             img.data(), 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a write command\n");

  // Set the arguments to our compute kernel
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_image_buffer);
  cl_error(err, "Failed to set argument 0\n");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_image_buffer);
  cl_error(err, "Failed to set argument 1\n");
  // Third, the number of elements of the input and output arrays.
  err = clSetKernelArg(kernel, 2, sizeof(int), &width);
  // cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
  cl_error(err, "Failed to set argument 2\n");
  err = clSetKernelArg(kernel, 3, sizeof(int), &height);
  cl_error(err, "Failed to set argument 3\n");

  // Launch Kernel

  // Ajustar el tamaño global y local para que sea divisible uniformemente
  size_t local_size[2] = {256, 1}; // Tamaño local ajustado a 256 en X y 1 en Y
  size_t global_size[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};

  // Asegurarse de que el tamaño global sea divisible uniformemente por el tamaño local
  size_t remainder_x = width % local_size[0];
  size_t remainder_y = height % local_size[1];

  if (remainder_x != 0 || remainder_y != 0)
  {
    // Si no es divisible uniformemente, ajustar el tamaño global
    global_size[0] = (width + local_size[0] - remainder_x);
    global_size[1] = (height + local_size[1] - remainder_y);
  }

  // size_t local_size[2] = {128, 128}; // Define local_size as an array of size_t
  // size_t global_size[2] = {static_cast<size_t>(count), static_cast<size_t>(count)};

  err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
  cl_error(err, "Failed to launch kernel to the device");

  // Read the modified image data back to the host (from device to host)
  unsigned char *out_image_data = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 3);
  err = clEnqueueReadBuffer(command_queue, out_image_buffer, CL_TRUE, 0, sizeof(unsigned char) * width * height * 3, out_image_data, 0, NULL, NULL);
  cl_error(err, "Failed to read modified image data from device\n");

  // Save the modified image data to the output path
  CImg<unsigned char> out_image(out_image_data, width, height, 1, 3);
  out_image.save_jpeg(output_image_path);

  free(out_image_data); // Free the memory allocated for the output image data

  // Release OpenCL resources
  clReleaseMemObject(in_image_buffer);
  clReleaseMemObject(out_image_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}