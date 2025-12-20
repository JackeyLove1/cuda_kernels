#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
__global__ void index_print_kernel()
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_idx = threadIdx.x / warpSize;
    const int lane_idx = threadIdx.x & (warpSize - 1);
    if ((lane_idx & (warpSize/2 -1)) == 0)
    {
        printf("id: %d, warpId: %d, laneId: %d\n", idx, warp_idx, lane_idx);
    }
    namespace cg = cooperative_groups;
    const auto tid = cg::this_grid().thread_rank();
    printf("tid: %d\n", tid);
}

int main(){
    index_print_kernel<<<2, 8>>>();
    cudaDeviceSynchronize();

}