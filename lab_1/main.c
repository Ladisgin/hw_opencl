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
#include <omp.h>

#define N 2048
#define K 1024
#define M 512
#define TS 8
#define FUNCTION_NAME "matrix_mul"
#define DELTA 1e-10

char *read_program(size_t *sz) {
    FILE *f = fopen("../matrixmul.cl", "rb");
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

void print_matrices(const cl_float *a, const cl_float *b, const cl_float *c,
                    cl_uint n, cl_uint k, cl_uint m) {

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k; j++) {
            printf("%f\t", a[i * k + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (size_t j = 0; j < k; j++) {
        for (size_t t = 0; t < m; t++) {
            printf("%f\t", b[j * m + t]);
        }
        printf("\n");
    }
    printf("\n");

    for (size_t i = 0; i < n; i++) {
        for (size_t t = 0; t < m; t++) {
            printf("%f\t", c[i * m + t]);
        }
        printf("\n");
    }
    printf("\n");
}

long long verify_result(const cl_float *a, const cl_float *b, const cl_float *c,
                        cl_uint n, cl_uint k, cl_uint m) {
    cl_float *verify_c = (cl_float *) malloc(sizeof(cl_float) * n * m);
    memset(verify_c, 0, sizeof(cl_float) * n * m);

    long long begin = clock();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t t = 0; t < m; ++t) {
                verify_c[i * m + t] += a[i * k + j] * b[j * m + t];
            }
        }
    }
    long long end = clock();


    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (fabs(verify_c[i * m + j] - c[i * m + j]) > DELTA) {
                return -1;
            }
        }
    }
    return end - begin;
}

long long verify_result_openmp(const cl_float *a, const cl_float *b, const cl_float *c,
                               cl_uint n, cl_uint k, cl_uint m) {
    cl_float *verify_c = (cl_float *) malloc(sizeof(cl_float) * n * m);
    memset(verify_c, 0, sizeof(cl_float) * n * m);

    long long begin = clock();
#pragma omp parallel for default(none) shared(n, k, m, a, b, verify_c) schedule(dynamic, 10)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t t = 0; t < m; ++t) {
                verify_c[i * m + t] += a[i * k + j] * b[j * m + t];
            }
        }
    }
    long long end = clock();


    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (fabs(verify_c[i * m + j] - c[i * m + j]) > DELTA) {
                return -1;
            }
        }
    }
    return end - begin;
}

cl_kernel create_kernel(cl_context *context, cl_command_queue *queue, cl_device_id device) {
    cl_int errcode_ret = malloc(sizeof(cl_int));
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

void fill_matrices(cl_float *a, cl_float *b, cl_float *c,
                   cl_uint n, cl_uint k, cl_uint m) {
    srand(time(0));

    for (size_t l = 0; l < n * k; ++l) {
        a[l] = rand() % 100;
    }
    for (size_t l = 0; l < k * m; ++l) {
        b[l] = rand() % 100;
    }
}

cl_ulong calculate(cl_kernel kernel, cl_command_queue queue, cl_context context,
                   cl_float *a, cl_float *b, cl_float *c,
                   cl_uint n, cl_uint k, cl_uint m) {
    cl_long calc_time = -1;

    cl_int a_err, b_err, c_err;

    cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * n * k, NULL, &a_err);
    cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * k * m, NULL, &b_err);
    cl_mem buf_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n * m, NULL, &c_err);

    if (a_err != CL_SUCCESS || b_err != CL_SUCCESS || c_err != CL_SUCCESS) {
        printf("create buffer error: a)%d, b)%d, c)%d\n", a_err, b_err, c_err);
    } else {
        a_err = clEnqueueWriteBuffer(queue, buf_a, CL_TRUE, 0, sizeof(cl_float) * n * k, a, 0, NULL, NULL);
        b_err = clEnqueueWriteBuffer(queue, buf_b, CL_TRUE, 0, sizeof(cl_float) * k * m, b, 0, NULL, NULL);

        if (a_err != CL_SUCCESS || b_err != CL_SUCCESS) {
            printf("write in buffer error: a)%d, b)%d\n", a_err, b_err);
        } else {
            cl_int set_arg_1 = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
            cl_int set_arg_2 = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
            cl_int set_arg_3 = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c);

            cl_int set_arg_4 = clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
            cl_int set_arg_5 = clSetKernelArg(kernel, 4, sizeof(cl_uint), &k);
            cl_int set_arg_6 = clSetKernelArg(kernel, 5, sizeof(cl_uint), &m);

            if (set_arg_1 != CL_SUCCESS || set_arg_2 != CL_SUCCESS || set_arg_3 != CL_SUCCESS
                || set_arg_4 != CL_SUCCESS || set_arg_5 != CL_SUCCESS || set_arg_6 != CL_SUCCESS) {
                printf("Error in setting kernel: 1)%d, 2)%d, 3)%d, 4)%d, 5)%d, 6)%d\n",
                       set_arg_1, set_arg_2, set_arg_3, set_arg_4, set_arg_5, set_arg_6);
            } else {

                size_t *global = (size_t *) malloc(2 * sizeof(size_t));
                size_t *local = (size_t *) malloc(2 * sizeof(size_t));
                global[0] = N;
                global[1] = M;
                local[0] = TS;
                local[1] = TS;

                cl_event event;

                cl_int range_error = clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, NULL, &event);

                if (range_error != CL_SUCCESS) {
                    printf("Error enqueue range: %d\n", range_error);
                } else {
                    c_err = clEnqueueReadBuffer(queue, buf_c, CL_TRUE, 0, sizeof(cl_float) * n * m, c, 0, NULL, NULL);
                    if (c_err != CL_SUCCESS) {
                        printf("Error read buffer: %d\n", c_err);
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

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_c);
    return calc_time;
}

int main() {
    cl_int n = N;
    cl_int k = K;
    cl_int m = M;

    cl_float *a = (cl_float *) malloc(sizeof(cl_float) * n * k);
    cl_float *b = (cl_float *) malloc(sizeof(cl_float) * k * m);
    cl_float *c = (cl_float *) malloc(sizeof(cl_float) * n * m);

    fill_matrices(a, b, c, n, k, m);

    cl_device_id device = get_device();

    printf("find device: ");
    if (device != (cl_device_id) -1) {
        printf("ok\n");
        cl_context context;
        cl_command_queue queue;
        cl_kernel kernel = create_kernel(&context, &queue, device);
        if (kernel != (cl_kernel) -1) {
            cl_long opencl_t = calculate(kernel, queue, context, a, b, c, n, k, m);

            if (opencl_t == -1) {
                printf("can't calculate on openCL");
                return 0;
            }

            long openmp_t = verify_result_openmp(a, b, c, n, k, m);
            long naive_t = verify_result(a, b, c, n, k, m);


            printf("%d, %d, %d\n", opencl_t, openmp_t, naive_t);

            printf("\n");

            double opencl_t_f = opencl_t * 1.0 / 1e9;
            double openmp_t_f = openmp_t * 1.0 / CLOCKS_PER_SEC;
            double naive_t_f = naive_t * 1.0 / CLOCKS_PER_SEC;
            printf("openCL: %0.5lf s\n", opencl_t_f);
            printf("openMP: %0.5lf s\n", openmp_t_f);
            printf("naive: %0.5lf s\n", naive_t_f);

            printf("\n");

            double opencl_GFLOPS = 1e3 * n * m * k * 2.0 / opencl_t;
            double openmp_GFLOPS = n * m * k * 2.0 / openmp_t;
            double naive_GFLOPS = n * m * k * 2.0 / naive_t;
            printf("openCL: %0.5lf GFLOPS\n", opencl_GFLOPS);
            printf("openMP: %0.5lf GFLOPS\n", openmp_GFLOPS);
            printf("naive: %0.5lf GFLOPS\n", naive_GFLOPS);

            if (n + k + m < 16) {
                print_matrices(a, b, c, n, k, m);
            }
        }
    }
    free(a);
    free(b);
    free(c);
    return 0;
}