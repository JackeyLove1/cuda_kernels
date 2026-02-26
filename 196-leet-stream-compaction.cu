#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <vector>
// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
  cudaMemset(out, 0, N * sizeof(float));
  thrust::copy_if(thrust::device, A, A + N, out, [=] __device__(const float x) { return x > 0.f; });
}

#ifdef LOCAL_TEST
static void checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
  }
}

int main() {
  std::vector<float> h_a = {1.0f, -2.0f, 3.0f, 0.0f, -1.0f, 4.0f};
  const int N = static_cast<int>(h_a.size());
  std::vector<float> h_out(N, 0.0f);

  float* d_a = nullptr;
  float* d_out = nullptr;
  checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc(d_a) failed");
  checkCuda(cudaMalloc(&d_out, N * sizeof(float)), "cudaMalloc(d_out) failed");

  checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D failed");

  solve(d_a, N, d_out);
  checkCuda(cudaGetLastError(), "solve launch failed");
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

  checkCuda(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H failed");

  std::cout << "A = [";
  for (int i = 0; i < N; ++i) {
    std::cout << h_a[i] << (i + 1 == N ? "" : ", ");
  }
  std::cout << "]\n";

  std::cout << "out = [";
  for (int i = 0; i < N; ++i) {
    std::cout << h_out[i] << (i + 1 == N ? "" : ", ");
  }
  std::cout << "]\n";

  cudaFree(d_a);
  cudaFree(d_out);
  return 0;
}
#endif
