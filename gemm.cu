#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// define variables
const int N = 16777216;
const int radix = 256;
const int ROW = N / radix;
const int shared_n = 64, shared_m = 32, shared_k = 64;

half *twiddle_Host, *F_Host, *in_Host, *result_Host, *standard_Host;
half *twiddle_Device, *F_Device, *in_Device, *result_Device, *standard_Device;

// define functions
__global__ void do_one_time(half *F_Device, half *in_Device, half *result_Device)
{
    int block_m = blockIdx.y * shared_m;
    int block_n = blockIdx.x * shared_n;

    __shared__ half smem_F[shared_m][shared_k];
    __shared__ half smem_in[shared_k][shared_n];

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> out;
    wmma::fill_fragment(out, 0.0);
    int warp_m = threadIdx.y / 4 * 16, warp_n = threadIdx.y % 4 * 16;

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0-255

    for (int block_k = 0; block_k < radix; block_k += shared_k)
    {
        // store F and in in shared memory
        for (int i = 0; i < shared_m * shared_k / 256; ++i)
        {
            int e_id = i * 256 + tid;
            int F_m = e_id / shared_k, F_n = e_id % shared_k;
            smem_F[F_m][F_n] = F_Device[(block_m + F_m) * radix + F_n + block_k];
            // if (block_m == 0 && block_n == 0 && block_k == 64)
            //     printf("%d %d: %f\n", F_m, F_n, __half2float(smem_F[F_m][F_n]));
        }

        for (int i = 0; i < shared_k * shared_n / 256; ++i)
        {
            int e_id = i * radix + tid;
            int in_m = e_id / shared_n, in_n = e_id % shared_n;
            smem_in[in_m][in_n] = in_Device[(in_m + block_k) * ROW + block_n + in_n];
            // if (block_m == 0 && block_n == 0 && block_k == 64)
            //     printf("%d %d: %f\n", in_m, in_n, __half2float(smem_in[in_m][in_n]));
        }
        __syncthreads();

        for (int warp_k = 0; warp_k < shared_k; warp_k += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_F;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_in;
            wmma::load_matrix_sync(frag_F, (half *)smem_F + warp_m * shared_k + warp_k, shared_k);
            wmma::load_matrix_sync(frag_in, (half *)smem_in + warp_k * shared_n + warp_n, shared_n);
            wmma::mma_sync(out, frag_F, frag_in, out);
        }
        __syncthreads();
    }
    // if(block_m == 0 && block_n == 0 && threadIdx.y == 0)
    wmma::store_matrix_sync(result_Device + (block_m + warp_m) * ROW + block_n + warp_n, out, ROW, wmma::mem_row_major);
}

void doit(int iter)
{
    dim3 blocks(ROW / shared_n, radix / shared_m);
    dim3 threads(32, 8);
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
    in_Host = (half *)malloc(sizeof(half) * N);
    twiddle_Host = (half *)malloc(sizeof(half) * N);
    result_Host = (half *)malloc(sizeof(half) * N);
    standard_Host = (half *)malloc(sizeof(half) * N);
    F_Host = (half *)malloc(sizeof(half) * radix * radix);

    for (int j = 0; j < radix; ++j)
        for (int k = 0; k < N / radix; ++k)
            twiddle_Host[N / radix * j + k] = cosf(M_PI * 2 * j * k / N);

    for (int i = 0; i < radix; ++i)
        for (int j = 0; j < radix; ++j)
            F_Host[i * radix + j] = cosf(M_PI * 2 * j * i / radix);

    srand(42);
    for (int i = 0; i < N; ++i)
        in_Host[i] = 0.001f * rand() / RAND_MAX;

    // device data inital
    cudaMalloc((void **)&in_Device, sizeof(half) * N);
    cudaMalloc((void **)&twiddle_Device, sizeof(half) * N);
    cudaMalloc((void **)&result_Device, sizeof(half) * N);
    cudaMalloc((void **)&standard_Device, sizeof(half) * N);
    cudaMalloc((void **)&F_Device, sizeof(half) * radix * radix);

    cudaMemcpy(in_Device, in_Host, sizeof(half) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(twiddle_Device, twiddle_Host, sizeof(half) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(F_Device, F_Host, sizeof(half) * radix * radix, cudaMemcpyHostToDevice);

    // do cublasHgemm
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    const half one = 1.0;
    const half zero = 0.0;
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ROW, radix, radix, &one, in_Device, ROW, F_Device, radix, &zero, standard_Device, ROW);
    cudaMemcpy(standard_Host, standard_Device, sizeof(half) * N, cudaMemcpyDeviceToHost);

    // do one time
    doit(1);
    cudaMemcpy(result_Host, result_Device, sizeof(half) * N, cudaMemcpyDeviceToHost);

    // check resutls
    float error = 0.0;
    for (int i = 0; i < 16777216; ++i)
        if (__half2float(standard_Host[i]) != 0.0)
            error += std::abs((__half2float(result_Host[i]) - __half2float(standard_Host[i])) / __half2float(standard_Host[i]));
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
float
avg relative error: 1.85e-06
n: 16777216, iter: 8192, time per iter: 0.000630
*/

/*
half
avg relative error: nan
n: 16777216, iter: 16384, time per iter: 0.000330
*/