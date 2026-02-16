#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <cuda/std/utility>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ auto LOAD4(const T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<const int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<const float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<const double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<const short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for LOAD4");
    return nullptr;
  }
}

template <typename T>
__forceinline__ __device__ auto STORE4(T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for STORE4");
    return nullptr;
  }
}

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

[[maybe_unused]] static inline __device__ float atomicMax(float* addr, float value) {
  float old = *addr, assumed;
  if (old >= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

  } while (old != assumed);

  return old;
}

template <typename T>
struct ReduceOp {
  static constexpr char op_id = 0;
};

template <typename T>
struct SumOp : public ReduceOp<T> {
  static constexpr char op_id = 1;

  __forceinline__ __device__ static T apply(const T& a, const T& b) { return a + b; }

  __forceinline__ __device__ static constexpr auto identity() { return T{0}; }
};

template <typename T>
struct MaxOp : public ReduceOp<T> {
  static constexpr char op_id = 2;

  __forceinline__ __device__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, float>) {
      return fmaxf(a, b);
    } else {
      return (a > b) ? a : b;
    }
  }

  __forceinline__ __device__ static constexpr auto identity() {
    if constexpr (std::is_same_v<T, float>) {
      return -INFINITY;
    } else if constexpr (std::is_same_v<T, half>) {
      return -65504;
    } else {
      return std::numeric_limits<T>::lowest();
    }
  }
};

static constexpr unsigned int FULL_MASK = 0xffffffff;

template <typename T, typename Op>
__forceinline__ __device__ T WarpReduceOp(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = Op::apply(value, __shfl_down_sync(FULL_MASK, value, offset));
  }
  return value;
}

template <typename T, typename Op>
inline __device__ T BlockReduceOp(T value) {
  const auto tid = threadIdx.x;
  const auto lane_id = tid & 31;
  const auto warp_id = tid >> 5;

  // warp reduce
  value = WarpReduceOp<T, Op>(value);

  __shared__ T shared_data[32];
  if (lane_id == 0) {
    shared_data[warp_id] = value;
  }
  __syncthreads();

  // block reduce
  if (warp_id == 0) {
    const int nums_warps = blockDim.x >> 5;
    const T lane_value = (lane_id < nums_warps) ? shared_data[lane_id] : Op::identity();
    value = WarpReduceOp<T, Op>(lane_value);
  }
  return value;
}

enum class ReduceOpType {
  SUM = 1,
  MAX = 2,
};

template <typename T>
__forceinline__ __device__ T BlockReduceDynamic(T value, ReduceOpType op_type) {
  switch (op_type) {
    case ReduceOpType::SUM:
      return BlockReduceOp<T, SumOp<T>>(value);
    case ReduceOpType::MAX:
      return BlockReduceOp<T, MaxOp<T>>(value);
    default:
      // Runtime-dispatched op type: keep a runtime fallback instead of
      // compile-time static_assert in a potentially unreachable branch.
      return value;
  }
}

[[maybe_unused]] static constexpr int CFACTOR = 1;
[[maybe_unused]] static constexpr int WARP_SIZE = 32;
[[maybe_unused]] static constexpr int MAX_BLOCKS = 4096;
[[maybe_unused]] static constexpr int THREADS_PER_BLOCK = 256;
[[maybe_unused]] static constexpr float FLOAT_MAX = INFINITY;

/**
Implement a program for a multi-agent flocking simulation (boids). The input consists of:
An array agents containing N agents, where N is the total number of agents
Each agent occupies 4 consecutive 32-bit floating point numbers in the array:[x,y,vx,vy],
where:
(x,y) represents the agent's position in 2D space
(vx,vy) represents the agent's velocity vector
The total array size is 4*N floats, with agent i's data stored at indices[4i,4i+1,4i+2,4i+3]
Simulation Rules
1. For each agent i, identify all neighbors j (where i≠j) within radius r=5.0 using:
✔(xi-xj)²+(yi-yj)²
2. Compute average velocity of neighboring agents:
vavg={|Ni|≥1}∑j∈Nivj if |Ni|>0
vvi if |Ni|=0
3. Update velocity:
vnew=v+α(vavg-v),whereα=0.05
4. Update position:
pnew=p+vnew
Implementation Requirements
·Use only native features (external libraries are not permitted)
·The solve function signature must remain unchanged
·The final result must be stored in the agents _ next array
Example 1:
Input: N = 2
agents = [
0.0, 0.0, 1.0, 0.0, // Agent 0: [x, y, vx, vy]
3.0, 4.0, 0.0, 1.0 // Agent 1: [x, y, vx, vy]
]
Output:
agents _ next = [
1.0, 0.0, 1.0, 0.0, // Agent 0: [x, y, vx, vy]
3.0, 3.0, 0.0, 1.0 // Agent 1: [x, y, vx, vy]
]

Constraints
1 ≤ N ≤ 100,000
Each agent's position and velocity components are 32-bit floats
Performance is measured with N = 10,000
 */

__global__ void kernel(const float* agents, float* __restrict__ agents_next, const int N) {
  static constexpr double radius_pow_2 = 25.0;
  static constexpr double alpha = 0.05;

  extern __shared__ float smem[];
  const int tile_size = blockDim.x;
  float4* tile_agent = reinterpret_cast<float4*>(smem);  // tile_size * sizeof(float) * 4
  const int tid = threadIdx.x;
  const int gid = blockDim.x * blockIdx.x + tid;
  const int num_tiles = CEIL(N, blockDim.x);
  const float4* agent4 = LOAD4(agents);

  const auto agent = (gid < N) ? agent4[gid] : make_float4(FLOAT_MAX, FLOAT_MAX, 0, 0);
  double2 p = make_double2(static_cast<double>(agent.x), static_cast<double>(agent.y));
  double2 v = make_double2(static_cast<double>(agent.z), static_cast<double>(agent.w));
  // Use double accumulators to reduce numerical drift on large neighbor sets.
  double v_avg_x = 0.0;
  double v_avg_y = 0.0;
  int total = 0;

  for (int k = 0; k < num_tiles; ++k) {
    const int base = k * tile_size;
    int idx = base + tid;
    tile_agent[tid] = (idx < N) ? agent4[idx] : make_float4(FLOAT_MAX, FLOAT_MAX, 0, 0);
    __syncthreads();

    for (int i = 0; i < tile_size; ++i) {
      const int agent_id = base + i;
      if (agent_id == gid) continue;
      if (agent_id < N) {
        float4 other = tile_agent[i];
        const double dx = static_cast<double>(agent.x) - static_cast<double>(other.x);
        const double dy = static_cast<double>(agent.y) - static_cast<double>(other.y);
        const double distance = dx * dx + dy * dy;
        if (distance < radius_pow_2) {
          // update v
          v_avg_x += static_cast<double>(other.z);
          v_avg_y += static_cast<double>(other.w);
          total += 1;
        }
      }
    }
    __syncthreads();
  }

  // update v and v_avg
  const double v_avg_x_d = (total == 0) ? v.x : (v_avg_x / static_cast<double>(total));
  const double v_avg_y_d = (total == 0) ? v.y : (v_avg_y / static_cast<double>(total));
  v.x += alpha * (v_avg_x_d - v.x);
  v.y += alpha * (v_avg_y_d - v.y);

  // update position
  p.x += v.x;
  p.y += v.y;

  // write back to agents_next
  float4* agent_next4 = STORE4(agents_next);
  if (gid < N) {
    agent_next4[gid] = make_float4(static_cast<float>(p.x), static_cast<float>(p.y),
                                   static_cast<float>(v.x), static_cast<float>(v.y));
  }
}

extern "C" void solve(const float* agents, float* agents_next, int N) {
  constexpr int nthrs = 256;
  const int nblks = CEIL(N, nthrs);
  constexpr int shared_size = nthrs * sizeof(float) * 4;
  kernel<<<nblks, nthrs, shared_size>>>(agents, agents_next, N);
}