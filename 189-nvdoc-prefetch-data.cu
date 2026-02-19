#include <cooperative_groups.h>

#include <cuda/pipeline>

template <size_t num_stages = 2 /* Pipeline with num_stages stages */>
__global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size,
                                size_t batch_size) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto thread = cooperative_groups::this_thread();
  assert(size == batch_size * grid.size());  // Assume input size fits batch_size * grid_size

  extern __shared__ int shared[];  // num_stages * block.size() * sizeof(int) bytes
  size_t shared_offset[num_stages];
  for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

  auto block_batch = [&](size_t batch) -> int {
    return block.group_index().x * block.size() + grid.size() * batch;
  };

  // Fill the pipeline with the first ``num_stages`` batches.
  for (int s = 0; s < num_stages; ++s) {
    pipeline.producer_acquire();
    cuda::memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s) + tid,
                       cuda::aligned_size_t<4>(sizeof(int)), pipeline);
    pipeline.producer_commit();
  }

  int stage = 0;

  // compute_batch: next batch to process
  // fetch_batch:   next batch to fetch from global memory
  for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size;
       ++compute_batch, ++fetch_batch) {
    // Wait for the first requested stage to complete.
    constexpr size_t pending_batches = num_stages - 1;
    cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
    __syncthreads();  // Not required if each thread works on the data it copied.

    // Compute on the current batch
    compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);

    // Release the current stage.
    pipeline.consumer_release();
    __syncthreads();  // Not required if each thread works on the data it copied.

    // Load future stage ``num_stages`` ahead of current compute batch.
    pipeline.producer_acquire();
    if (fetch_batch < batch_size) {
      cuda::memcpy_async(shared + shared_offset[stage] + tid,
                         global_in + block_batch(fetch_batch) + tid,
                         cuda::aligned_size_t<4>(sizeof(int)), pipeline);
    }
    pipeline.producer_commit();
    stage = (stage + 1) % num_stages;
  }
}