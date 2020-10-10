#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

// define variables
const int N = 16777216;
const int radix = 256;

float *twiddle_Host, *F_Host, *in_Host, *result_Host, *standard_Host;
float *twiddle_Device, *F_Device, *in_Device, *result_Device, *standard_Device;

// define functions
__global__ void do_one_time(float *F_Device, float *in_Device, float *result_Device)
{
    int m = (blockDim.y * blockIdx.y + threadIdx.y) * 4;
    int n = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    float accumulator[4][4];

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            accumulator[i][j] = 0.0;

    for (int bigK = 0; bigK < 256; bigK+=16)
        for (int smallK = 0; smallK < 16; ++smallK)
        {
            int k = smallK + bigK;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                {
                    int row = m + i;
                    int col = n + j;
                    accumulator[i][j] += F_Device[row * 256 + k] * in_Device[k * 65536 + col];
                }
        }

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            int row = m + i;
            int col = n + j;
            result_Device[row * 65536 + col] = accumulator[i][j];
        }
}

void doit(int iter)
{
    dim3 blocks(256, 16);
    dim3 threads(64, 4);
    for (int i = 0; i < iter; ++i)
        do_one_time<<<blocks, threads>>>(F_Device, in_Device, result_Device);
}

double gettime()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_usec * 1.0e-6 + tv.tv_sec;
}

// main
int main()
{
    // host data initial
    in_Host = (float *)malloc(sizeof(float) * N);
    twiddle_Host = (float *)malloc(sizeof(float) * N);
    result_Host = (float *)malloc(sizeof(float) * N);
    standard_Host = (float *)malloc(sizeof(float) * N);
    F_Host = (float *)malloc(sizeof(float) * radix * radix);

    for (int j = 0; j < radix; ++j)
        for (int k = 0; k < N / radix; ++k)
            twiddle_Host[N / radix * j + k] = cosf(M_PI * 2 * j * k / N);

    for (int i = 0; i < radix; ++i)
        for (int j = 0; j < radix; ++j)
            F_Host[i * radix + j] = cosf(M_PI * 2 * j * i / radix);

    srand(42);
    for (int i = 0; i < N; ++i)
        in_Host[i] = 0.0001f * rand() / RAND_MAX;

    // device data inital
    cudaMalloc((void **)&in_Device, sizeof(float) * N);
    cudaMalloc((void **)&twiddle_Device, sizeof(float) * N);
    cudaMalloc((void **)&result_Device, sizeof(float) * N);
    cudaMalloc((void **)&standard_Device, sizeof(float) * N);
    cudaMalloc((void **)&F_Device, sizeof(float) * radix * radix);

    cudaMemcpy(in_Device, in_Host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(twiddle_Device, twiddle_Host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(F_Device, F_Host, sizeof(float) * radix * radix, cudaMemcpyHostToDevice);

    // do cublasSgemm
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    const float one = 1.0;
    const float zero = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 65536, 256, 256, &one, in_Device, 65536, F_Device, 256, &zero, standard_Device, 65536);
    cudaMemcpy(standard_Host, standard_Device, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // do one time
    doit(1);
    cudaMemcpy(result_Host, result_Device, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // check resutls
    double error = 0.0;
    for (int i = 0; i < 16777216; ++i)
        error += std::abs((result_Host[i] - standard_Host[i]) / standard_Host[i]);
    error /= 16777216;
    printf("avg relative error: %.2e\n", error);

    // speed test
    double run_time;
    double t_min = 4;
    int max_times = 1 << 30, iter;
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
    }
    printf("n: 16777216, iter: %d, time per iter: %lf\n", iter, run_time / iter);

    return 0;
}

/*
avg relative error: 1.85e-06
n: 16777216, iter: 8192, time per iter: 0.000630
*/