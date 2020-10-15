#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else
#include <CL/opencl.h>
#endif

#include <time.h>

#define N 1024*1024
#define TS 256
#define FUNCTION_NAME "pref_sum"
#define DELTA log2(N)

char *read_program(size_t *sz) {
    FILE *f = fopen("../prefsum.cl", "rb");
    fseek(f, 0L, SEEK_END);
    *sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *str = malloc(sizeof(char) * (*sz));
    fread(str, sizeof(char), *sz, f);
    fclose(f);
    return str;
}

cl_device_id get_device() {
    cl_device_id resultDevice = (cl_device_id) -1;
    cl_device_type resultDeviceType;
    cl_bool resultDiscrete = CL_FALSE;

    cl_int result;

    cl_uint platformCount;
    cl_platform_id *platforms;
    result = clGetPlatformIDs(0, NULL, &platformCount);
    if (result != CL_SUCCESS) {
        return resultDevice;
    }
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platformCount);
    result = clGetPlatformIDs(platformCount, platforms, NULL);
    if (result != CL_SUCCESS) {
        return resultDevice;
    }

    for (size_t i = 0; i < platformCount; ++i) {
        cl_platform_id curPlatform = platforms[i];

        size_t platformInfoSize;
        clGetPlatformInfo(curPlatform, CL_PLATFORM_NAME, 0, 0, &platformInfoSize);
        char *platformName = (char *) malloc(platformInfoSize * sizeof(char));
        clGetPlatformInfo(curPlatform, CL_PLATFORM_NAME, platformInfoSize, platformName, &platformInfoSize);

        printf("%s\n", platformName);
        free(platformName);

        cl_uint deviceCount = 0;
        result = clGetDeviceIDs(curPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        if (result != CL_SUCCESS || deviceCount == 0) {
            continue;
        }

        cl_device_id *devices = (cl_device_id *) malloc(deviceCount * sizeof(cl_device_id));
        result = clGetDeviceIDs(curPlatform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        if (result == CL_SUCCESS) {
            for (size_t j = 0; j < deviceCount; j++) {
                cl_device_id curDevice = devices[j];

                size_t deviceNameSize;
                clGetDeviceInfo(curDevice, CL_DEVICE_NAME, 0, 0, &deviceNameSize);
                char *deviceName = (char *) malloc(deviceNameSize * sizeof(char));
                clGetDeviceInfo(curDevice, CL_DEVICE_NAME, deviceNameSize, deviceName, &deviceNameSize);
                printf("\t%s\n", deviceName);
                free(deviceName);

                cl_device_type curDeviceType;
                clGetDeviceInfo(curDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &curDeviceType, NULL);
                if (resultDevice == (cl_device_id) -1 || curDeviceType > resultDeviceType) {
                    resultDevice = curDevice;
                    resultDeviceType = curDeviceType;

                    cl_bool memoryInfo;
                    clGetDeviceInfo(curDevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &memoryInfo, NULL);
                    resultDiscrete = !memoryInfo;
                } else if (curDeviceType == resultDeviceType && !resultDiscrete) {
                    cl_bool memoryInfo;
                    clGetDeviceInfo(curDevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &memoryInfo, NULL);
                    if (memoryInfo == CL_FALSE) {
                        resultDevice = curDevice;
                        resultDeviceType = curDeviceType;
                        resultDiscrete = !memoryInfo;
                    }
                }
            }
        }
        free(devices);
    }
    free(platforms);


    size_t deviceNameSize;
    clGetDeviceInfo(resultDevice, CL_DEVICE_NAME, 0, 0, &deviceNameSize);
    char *deviceName = (char *) malloc(deviceNameSize * sizeof(char));
    clGetDeviceInfo(resultDevice, CL_DEVICE_NAME, deviceNameSize, deviceName, &deviceNameSize);
    printf("chosen: %s\n", deviceName);
    free(deviceName);

    return resultDevice;
}

bool verify_result(const float *arr, const float *result, size_t n) {
    float t = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        t += arr[i];
        if (fabs(t - result[i]) > DELTA) {
            printf("verify result: %f\nresult: %f\nstep: %d\n", t, result[i], i);
            return false;
        }
    }
    printf("verify result last: %f\nopencl result last: %f\n", t, result[n - 1]);
    return true;
}

cl_kernel create_kernel(cl_context *context, cl_command_queue *queue, cl_device_id device) {
    cl_int errcode_ret;
    *context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode_ret);

    printf("create context: ");

    if (context == NULL || errcode_ret != CL_SUCCESS) {
        printf("%d\n", errcode_ret);
    } else {
        printf("ok\n");
        *queue = clCreateCommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
        printf("create queue: ");

        if (queue == NULL || errcode_ret != CL_SUCCESS) {
            printf("%d\n", errcode_ret);
        } else {
            printf("ok\n");

            size_t sz;
            char *str = read_program(&sz);

            cl_program prog = clCreateProgramWithSource(*context, 1, &str, &sz, &errcode_ret);

            char *options = (char *) malloc(64 * sizeof(char));

            sprintf(options, "-D TS=%d", TS);

            cl_int prog_ret = clBuildProgram(prog, 1, &device, options, NULL, NULL);

            printf("build program: ");
            if (prog_ret != CL_SUCCESS) {
                printf("%d\n", prog_ret);
                size_t s;
                clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, 0, &s);
                char *ptr4 = (char *) malloc(s * sizeof(char));
                clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, s, ptr4, &s);
                printf("%s", ptr4);
                free(ptr4);
            } else {
                printf("ok\n");
                cl_kernel ker = clCreateKernel(prog, FUNCTION_NAME, &errcode_ret);
                printf("create kernel: ");
                if (errcode_ret != CL_SUCCESS) {
                    printf("%d\n", errcode_ret);
                } else {
                    printf("ok\n");
                    return ker;
                }
            }
        }
    }
    return (cl_kernel) -1;
}

cl_ulong calculate(cl_kernel kernel, cl_command_queue queue, cl_context context,
                   cl_float *arr, cl_float *result, cl_uint n) {
    cl_long calc_time = -1;

    cl_int arr_err, result_err;

    cl_mem buf_arr = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * n, NULL, &arr_err);
    cl_mem buf_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n, NULL, &result_err);

    if (arr_err != CL_SUCCESS || result_err != CL_SUCCESS) {
        printf("create buffer error: arr)%d, result)%d\n", arr_err, result_err);
    } else {
        arr_err = clEnqueueWriteBuffer(queue, buf_arr, CL_TRUE, 0, sizeof(cl_float) * n, arr, 0, NULL, NULL);

        if (arr_err != CL_SUCCESS) {
            printf("write in buffer error: arr)%d\n", arr_err);
        } else {
            cl_int set_arg_1 = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_arr);
            cl_int set_arg_2 = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_result);

            cl_int set_arg_3 = clSetKernelArg(kernel, 2, sizeof(cl_uint), &n);

            if (set_arg_1 != CL_SUCCESS || set_arg_2 != CL_SUCCESS || set_arg_3 != CL_SUCCESS) {
                printf("Error in setting kernel: arr)%d, result)%d, n)%d\n",
                       set_arg_1, set_arg_2, set_arg_3);
            } else {
                size_t *global = (size_t *) malloc(sizeof(size_t));
                size_t *local = (size_t *) malloc(sizeof(size_t));
                global[0] = N;
                local[0] = TS;

                cl_event event;

                cl_int range_error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global, local, 0, NULL, &event);

                if (range_error != CL_SUCCESS) {
                    printf("Error enqueue range: %d\n", range_error);
                } else {
                    result_err = clEnqueueReadBuffer(queue, buf_result, CL_TRUE, 0, sizeof(cl_float) * n, result, 0,
                                                     NULL, NULL);
                    if (result_err != CL_SUCCESS) {
                        printf("Error read buffer: %d\n", result_err);
                    } else {
                        cl_ulong t_start, t_end;
                        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0);
                        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0);
                        calc_time = t_end - t_start;
                    }
                }
                free(global);
                free(local);
            }
        }
    }

    clReleaseMemObject(buf_result);
    clReleaseMemObject(buf_arr);
    return calc_time;
}

int main() {
    cl_int n = N;

    cl_float *arr = (cl_float *) malloc(sizeof(cl_float) * n);
    cl_float *result = (cl_float *) malloc(sizeof(cl_float) * n);

    for (size_t i = 0; i < n; i++) {
        result[i] = 0;
    }

    srand(time(NULL));
    for (size_t i = 0; i < n; i++) {
        arr[i] = (rand() % 20);
    }

    cl_device_id device = get_device();

    printf("find device: ");
    if (device != (cl_device_id) -1) {
        printf("ok\n");
        cl_context context;
        cl_command_queue queue;
        cl_kernel kernel = create_kernel(&context, &queue, device);
        if (kernel != (cl_kernel) -1) {
            cl_long opencl_t = calculate(kernel, queue, context, arr, result, n);

            if (opencl_t == -1) {
                printf("can't calculate on openCL");
                return 0;
            }

            int verify_res = verify_result(arr, result, n);

            if (verify_res) {
                printf("check: ok\n");
            } else {
                printf("check: bad\n");
            }

            printf("\n");

            double opencl_t_f = opencl_t * 1.0 / 1e9;
            printf("openCL: %0.5lf s\n", opencl_t_f);
        }
    }

    free(arr);
    free(result);
    return 0;
}