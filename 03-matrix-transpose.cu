#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// 调整为 32，适配现代 GPU 的 Warp Size (32)
// 32x32 = 1024 线程，刚好是 CUDA Block 的上限，能最大化利用带宽
constexpr int TILE_DIM = 32;

__global__ void matrix_transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols) {

    // 1. Shared Memory Padding
    // 声明 [32][33] 而不是 [32][32]
    // 作用：消除 Shared Memory 的 Bank Conflict。
    // 当我们按列读取 Smem 时，如果不加 padding，同一个 Warp 的线程会访问同一个 Bank 的不同地址，导致串行化。
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 2. 计算输入的全局坐标 (x, y)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 3. 联合读取 (Coalesced Read)
    // 即使矩阵尺寸不对齐，也要保证读取不越界
    if (x < cols && y < rows) {
        // 读取 Input(Row: y, Col: x)
        // 这里的访问是连续的：因为 x 随 threadIdx.x 变化
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    // 必须同步，等待整个 Block 加载完毕
    __syncthreads();

    // 4. 计算输出的全局坐标 (转置逻辑)
    // 关键技巧：我们重新计算 x, y，使其对应 Output 矩阵的坐标
    // 
    // 原来的 Block(bx, by) 负责读取 Input 的块 (bx, by)
    // 这个块转置后，应该位于 Output 的 (by, bx) 位置
    //
    // 为了保证写入时 Coalesced（合并访问），我们必须让 threadIdx.x 对应 Output 的列索引 (连续维度)
    // 所以：
    // new_x (Output Col) = blockIdx.y * TILE_DIM + threadIdx.x
    // new_y (Output Row) = blockIdx.x * TILE_DIM + threadIdx.y
    
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 5. 联合写入 (Coalesced Write)
    // 此时我们需要从 Smem 中取出对应的数据
    // 当前线程负责写 Output(Row: y, Col: x)
    // 该位置的数据对应 Input(Row: x, Col: y) 
    // 在 Smem 中，Input(Row: x, Col: y) 被映射到了 tile[threadIdx.x][threadIdx.y] 
    // (注意这里下标互换了，因为 threadIdx.x 现在代表原来 Input 的 Row 偏移)
    
    if (x < rows && y < cols) { // 注意 rows/cols 在输出矩阵中也是互换的
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }

}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    
    // Grid 维度计算保持不变，覆盖整个矩阵
    dim3 blocksPerGrid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );

    matrix_transpose_optimized<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    
    // 生产环境建议加上错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}