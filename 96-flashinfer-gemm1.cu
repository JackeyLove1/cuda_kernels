#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_pipeline.h>

// 双缓冲矩阵向量乘法
__global__ void matmul_double_buffer(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int N)
{
    // 共享内存双缓冲区
    __shared__ float s_A[2][128][32];  // 两个缓冲区
    __shared__ float s_x[2][32];

    int tid = threadIdx.x;
    int row = blockIdx.x * 128 + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + 31) / 32;

    // 预加载第一个tile
    int write_buf = 0;
    if (tid < 128 && 0 < num_tiles) {
        for (int i = 0; i < 32; i++) {
            int col = i;
            if (col < N) {
                __pipeline_memcpy_async(&s_A[write_buf][tid][i],
                                       &A[row * N + col],
                                       sizeof(float));
            }
        }
    }
    if (tid < 32 && 0 < num_tiles) {
        if (tid < N) {
            __pipeline_memcpy_async(&s_x[write_buf][tid],
                                   &x[tid],
                                   sizeof(float));
        }
    }
    __pipeline_commit();

    // 流水线主循环
    for (int tile = 0; tile < num_tiles; tile++) {
        int read_buf = write_buf;
        write_buf = 1 - write_buf;  // 切换缓冲区

        // 异步加载下一个tile (如果存在)
        if (tile + 1 < num_tiles) {
            if (tid < 128) {
                for (int i = 0; i < 32; i++) {
                    int col = (tile + 1) * 32 + i;
                    if (col < N) {
                        __pipeline_memcpy_async(&s_A[write_buf][tid][i],
                                               &A[row * N + col],
                                               sizeof(float));
                    }
                }
            }
            if (tid < 32) {
                int col = (tile + 1) * 32 + tid;
                if (col < N) {
                    __pipeline_memcpy_async(&s_x[write_buf][tid],
                                           &x[col],
                                           sizeof(float));
                }
            }
            __pipeline_commit();
        }

        // 等待当前tile的数据
        __pipeline_wait_prior(1);
        __syncthreads();

        // 计算当前tile
        if (tid < 128 && row < M) {
            for (int i = 0; i < 32; i++) {
                int col = tile * 32 + i;
                if (col < N) {
                    sum += s_A[read_buf][tid][i] * s_x[read_buf][i];
                }
            }
        }
        __syncthreads();
    }

    if (tid < 128 && row < M) {
        y[row] = sum;
    }
}

// 测试代码
void test_double_buffer() {
    const int M = 1024;
    const int N = 1024;

    float *h_A, *h_x, *h_y;
    float *d_A, *d_x, *d_y;

    h_A = new float[M * N];
    h_x = new float[N];
    h_y = new float[M];

    // 初始化数据
    for (int i = 0; i < M * N; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < N; i++) h_x[i] = rand() / (float)RAND_MAX;

    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((M + 127) / 128);

    matmul_double_buffer<<<grid, block>>>(d_A, d_x, d_y, M, N);

    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        float expected = 0.0f;
        for (int j = 0; j < N; j++) {
            expected += h_A[i * N + j] * h_x[j];
        }
        max_error = max(max_error, fabs(h_y[i] - expected));
    }

    printf("Double Buffer - Max error: %f\n", max_error);

    delete[] h_A;
    delete[] h_x;
    delete[] h_y;
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
  test_double_buffer();
}