#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <random>

int main()
{
    constexpr int N = 1 << 20;
    thrust::device_vector<float> d_in(N);
    // Use host vector to initialize data efficiently
    thrust::host_vector<float> h_in(N);
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i * 1.0f / rand());
    }
    d_in = h_in;

    thrust::device_vector<int> d_flags(N);
    thrust::transform(
        thrust::device, d_in.begin(), d_in.end(), d_flags.begin(),
        [] __device__ (float x) {return x > 0.f ? 1 : 0;}
    );

    thrust::device_vector<float> d_out(N);
    thrust::device_vector<int> d_num_selected(1);

    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    float* in_ptr   = thrust::raw_pointer_cast(d_in.data());
    int*   flg_ptr  = thrust::raw_pointer_cast(d_flags.data());
    float* out_ptr  = thrust::raw_pointer_cast(d_out.data());
    int*   cnt_ptr  = thrust::raw_pointer_cast(d_num_selected.data());

    // 1. Determine temporary storage size
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, in_ptr, flg_ptr, out_ptr, cnt_ptr, N);
    
    // 2. Allocate temporary storage
    cudaMalloc(&d_temp, temp_bytes);

    // 3. Run selection
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, in_ptr, flg_ptr, out_ptr, cnt_ptr, N);
    
    // 4. Free temporary storage
    cudaFree(d_temp);

    std::cout << d_num_selected[0] << std::endl;
    return 0;
}