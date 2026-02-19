#include <cooperative_groups.h>

#include <cuda/pipeline>

#pragma nv_diag_suppress static_var_with_dynamic_init

using pipeline = cuda::pipeline<cuda::thread_scope_block>;

__device__ void produce(pipeline& pipe, int num_stages, int stage, int num_batches, int batch,
                        float* buffer, int buffer_len, float* in, int N) {
  if (batch < num_batches) {
    pipe.producer_acquire();
    /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */
    cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x,
                       in + batch * buffer_len + threadIdx.x,
                       cuda::aligned_size_t<4>(sizeof(float)), pipe);
    pipe.producer_commit();
  }
}

__device__ void consume(pipeline& pipe, int num_stages, int stage, int num_batches, int batch,
                        float* buffer, int buffer_len, float* out, int N) {
  pipe.consumer_wait();
  /* consume buffer(stage) and update out(batch) */
  pipe.consumer_release();
}

__global__ void producer_consumer_pattern(float* in, float* out, int N, int buffer_len) {
  auto block = cooperative_groups::this_thread_block();
  constexpr int warpSize = 32;

  /* Shared memory buffer declared below is of size 2 * buffer_len
     so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  const int num_batches = N / buffer_len;

  // Create a partitioned pipeline with 2 stages where the first warp is the producer and the other
  // warps are consumers.
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int num_stages = 2;
  cuda::std::size_t producer_count = warpSize;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

  // Producer fills the pipeline
  if (block.thread_rank() < producer_count)
    for (int s = 0; s < num_stages; ++s)
      produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);

  // Process the batches
  int stage = 0;
  for (size_t b = 0; b < num_batches; ++b) {
    if (block.thread_rank() < producer_count) {
      // Producers prefetch the next batch
      produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
    } else {
      // Consumers consume the oldest batch
      consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
    }
    stage = (stage + 1) % num_stages;
  }
}