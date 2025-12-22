#include <cfloat>
#include <cuda_runtime.h>

#define WARP_SIZE    32
#define FULL_MASK    0xFFFFFFFF
#define MAX_IDENTITY (-FLT_MAX)

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

struct __align__(32) Stat
{
    float max_;
    float sum_;
};

__forceinline__ __device__ Stat combine_stats(const Stat a, const Stat b)
{
    if (a.max_ == MAX_IDENTITY)
        return b;
    if (b.max_ == MAX_IDENTITY)
        return a;

    Stat result;
    result.max_        = fmax(a.max_, b.max_);
    const auto scale_a = __expf(a.max_ - result.max_);
    const auto scale_b = __expf(b.max_ - result.max_);
    result.sum_        = scale_a * a.sum_ + scale_b * b.sum_;
    return result;
}

__forceinline__ __device__ Stat WarpReduceStat(Stat val)
{
#pragma unroll
    Stat rhs;
    for (auto offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        rhs.max_ = __shfl_down_sync(FULL_MASK, val.max_, offset);
        rhs.sum_ = __shfl_down_sync(FULL_MASK, val.sum_, offset);
        val      = combine_stats(val, rhs);
    }
}

__global__ void reduce_block_stats(const float *__restrict__ input, Stat *__restrict__ block_stats, const int N)
{
    const auto tid    = threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;
    auto       id     = threadIdx.x + blockDim.x * blockIdx.x;

    Stat local_val{MAX_IDENTITY, 0.0f};
    Stat batch_val;
    for (; id * 4 < N; id += stride) {
        const auto base_idx  = id * 4;
        const auto remaining = N - base_idx;
        if (remaining >= 4) {
            const auto data    = reinterpret_cast<const float4 *>(input)[id];
            const auto max_1   = fmax(data.x, data.y);
            const auto max_2   = fmax(data.z, data.w);
            const auto vec_max = fmax(max_1, max_2);
            const auto vec_sum = __expf(data.x - vec_max) + __expf(data.y - vec_max) + __expf(data.z - vec_max)
                               + __expf(data.w - vec_max);
            batch_val = {vec_max, vec_sum};
        }
        else {
            float m_batch = MAX_IDENTITY;
            float s_batch = 0.0f;
            for (int k = base_idx; k < N; ++k) {
                float v = input[base_idx];
                float new_max = fmaxf(m_batch, v);
                float scale = __expf(m_batch - new_max);
                s_batch = s_batch * scale + __expf(v - new_max);
                m_batch = new_max;
            }
            batch_val = {m_batch, s_batch};
        }

        local_val = combine_stats(local_val, batch_val);
    }

    local_val = WarpReduceStat(local_val);
    static __shared__ float smax[WARP_SIZE];
    static __shared__ float ssum[WARP_SIZE];
    const auto warp_id = tid / WARP_SIZE;
    const auto lane_id = tid & (WARP_SIZE - 1);
    if (lane_id == 0) {
        smax[warp_id] = local_val.max_;
        ssum[warp_id] = local_val.sum_;
    }
    __syncthreads();

    if (warp_id == 0) {
        Stat warp_stat;
        const auto warp_nums = blockDim.x / WARP_SIZE;
        warp_stat.max_ = (lane_id < warp_nums) ? smax[warp_id] : MAX_IDENTITY;
        warp_stat.sum_ = (lane_id < warp_nums) ? ssum[warp_id] : 0.f;
        warp_stat = WarpReduceStat(warp_stat);
        if (lane_id == 0) {
            block_stats[blockIdx.x] = warp_stat;
        }
    }
}

__global__ void reduce_global_stats(const Stat *__restrict__ stats,
                                    float *__restrict__ global_max,
                                    float *__restrict__ global_sum,
                                    const int num_blocks)
{
    Stat local_val{MAX_IDENTITY, 0.0f};
    for (auto i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        local_val = combine_stats(local_val, stats[i]);
    }

    local_val = WarpReduceStat(local_val);
    static __shared__ float smax[WARP_SIZE];
    static __shared__ float ssum[WARP_SIZE];
    const auto tid = threadIdx.x;
    const auto warp_id = tid / WARP_SIZE;
    const auto lane_id = tid & (WARP_SIZE - 1);
    if (lane_id == 0) {
        smax[warp_id] = local_val.max_;
        ssum[warp_id] = local_val.sum_;
    }
    __syncthreads();

    if (warp_id == 0) {
        Stat warp_stat;
        const auto warp_nums = blockDim.x / WARP_SIZE;
        warp_stat.max_ = (lane_id < warp_nums) ? smax[warp_id] : MAX_IDENTITY;
        warp_stat.sum_ = (lane_id < warp_nums) ? ssum[warp_id] : 0.f;
        warp_stat = WarpReduceStat(warp_stat);
        if (lane_id == 0) {
            *global_max = local_val.max_;
            *global_sum = local_val.sum_;
        }
    }
}


__global__ void softmax_kernel(const float *__restrict__ input,
                               float *__restrict__ output,
                               float *__restrict__ global_max,
                               float *__restrict__ global_sum,
                               const int N)
{
    const auto stride = gridDim.x * blockDim.x;
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    const auto global_max_v = *global_max;
    const auto inv_global_sum_v = 1.0f / (*global_sum);
    for (; idx * 4 < N; idx += stride) {
        const auto base_idx = idx * 4;
        const auto remaining = N - base_idx;
        if (remaining >= 4) {
            const auto data = reinterpret_cast<const float4*>(input)[idx];
            float4 result;
            result.x = __expf(data.x - global_max_v) * inv_global_sum_v;
            result.y = __expf(data.y - global_max_v) * inv_global_sum_v;
            result.z = __expf(data.z - global_max_v) * inv_global_sum_v;
            result.w = __expf(data.w - global_max_v) * inv_global_sum_v;
            reinterpret_cast<float4*>(output)[idx] = result;
        } else {
            for (int j = base_idx; j < N; ++j) {
                output[j] = __expf(input[j] - global_max_v) * inv_global_sum_v;
            }
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int  threadsPerBlock = 256;
    constexpr auto max_blocks      = 1024;
    const int      blocksPerGrid   = std::min(max_blocks, CDIV(N, threadsPerBlock * 4));

    Stat  *stat;
    float *d_sum, *d_max;
    cudaMalloc(&stat, sizeof(Stat) * blocksPerGrid);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));

    reduce_block_stats<<<blocksPerGrid, threadsPerBlock>>>(input, stat, N);

    reduce_global_stats<<<1, 256>>>(stat, d_max, d_sum, blocksPerGrid);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_max, d_sum, N);

    cudaDeviceSynchronize();
}