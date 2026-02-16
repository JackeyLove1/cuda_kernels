#include <cuda_runtime.h>

#include <iostream>
#include <vector>

constexpr int BlockSize = 256;

__global__ void distance_kernel(const float* data_x, const float* data_y, float* centroid_x,
                                float* centroid_y, int* belongs, int sample_size, int k) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < sample_size) {
    float px = data_x[id];
    float py = data_y[id];

    int min_idx = 0;
    float min_dis = INFINITY;
    for (int i = 0; i < k; i++) {
      float cx = centroid_x[i];
      float cy = centroid_y[i];

      float dis = (px - cx) * (px - cx) + (py - cy) * (py - cy);

      if (dis < min_dis) {
        min_idx = i;
        min_dis = dis;
      }
    }

    belongs[id] = min_idx;
  }
}

__global__ void center_kernel(const float* data_x, const float* data_y, float* centroid_x,
                              float* centroid_y, int* nums, int* belongs, int sample_size, int k) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= sample_size) {
    return;
  }

  atomicAdd(centroid_x + belongs[id], data_x[id]);
  atomicAdd(centroid_y + belongs[id], data_y[id]);
  atomicAdd(nums + belongs[id], 1);
}

__global__ void mean_kernel(float* centroid_x, float* centroid_y, const float* old_centroid_x,
                            const float* old_centroid_y, int* nums, int k) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= k) return;

  if (nums[id] > 0) {
    centroid_x[id] /= nums[id];
    centroid_y[id] /= nums[id];
  } else {
    // 空簇：保持原 centroid
    centroid_x[id] = old_centroid_x[id];
    centroid_y[id] = old_centroid_y[id];
  }
}

extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
  // int *belongs;
  // cudaMalloc(&belongs, sample_size * sizeof(int));
  int* nums;
  cudaMalloc(&nums, k * sizeof(int));
  float* res_centroid_x;
  cudaMalloc(&res_centroid_x, k * sizeof(float));
  float* res_centroid_y;
  cudaMalloc(&res_centroid_y, k * sizeof(float));

  int num_blocks;

  cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);
  for (int i = 0; i < max_iterations; i++) {
    cudaMemset(res_centroid_x, 0.f, k * sizeof(float));
    cudaMemset(res_centroid_y, 0.f, k * sizeof(float));
    cudaMemset(nums, 0, k * sizeof(int));

    num_blocks = (sample_size + BlockSize - 1) / BlockSize;
    distance_kernel<<<num_blocks, BlockSize>>>(data_x, data_y, final_centroid_x, final_centroid_y,
                                               labels, sample_size, k);
    // cdu::d_print<int>(belongs, sample_size);

    center_kernel<<<num_blocks, BlockSize>>>(data_x, data_y, res_centroid_x, res_centroid_y, nums,
                                             labels, sample_size, k);

    num_blocks = (k + BlockSize - 1) / BlockSize;
    mean_kernel<<<num_blocks, BlockSize>>>(res_centroid_x, res_centroid_y, final_centroid_x,
                                           final_centroid_y, nums, k);
    // cdu::d_print<float>(res_centroid_x, k);

    std::swap(final_centroid_x, res_centroid_x);
    std::swap(final_centroid_y, res_centroid_y);
  }

  cudaFree(nums);
  cudaFree(res_centroid_x);
  cudaFree(res_centroid_y);
}
