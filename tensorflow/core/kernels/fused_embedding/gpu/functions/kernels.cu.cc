

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"

#include <algorithm>

#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace fused_embedding {

using GPUDevice = Eigen::GpuDevice;

#define LINER_MAPPING_THREADS 128

inline int CalcBlocksLinearMapping(const int problem_size, const int threads) {
  return problem_size % threads == 0 ? (problem_size / threads)
                                     : (problem_size / threads + 1);
}

__global__ void InitFlagsToOneInt4Kernel(int length, int* flags) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (4 * offset + 3 < length) {
    *((int4*)(flags + 4 * offset)) = make_int4(1, 1, 1, 1);
  } else if (4 * offset < length) {
    for (int i = 0; i < length - 4 * offset; i++) {
      flags[4 * offset + i] = 1;
    }
  }
}

void InitFlagsToOneInt4(const GPUDevice& d, int length, int* flags) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(length, threads * 4);
  TF_CHECK_OK(GpuLaunchKernel(InitFlagsToOneInt4Kernel, blocks, threads, 0,
                              d.stream(), length, flags));
}

__global__ void DetectInvalidKernel(const int64_t* values, const int64_t nnz,
                                    int* invalid_id_flag) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    const int64_t value = values[offset];
    if (value < 0) {
      atomicAnd(invalid_id_flag + offset, 0);
    }
  }
}

void DetectInvalid(const GPUDevice& d, const int64_t* values, const int64_t nnz,
                   int* invalid_id_flag) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(nnz, threads);
  TF_CHECK_OK(GpuLaunchKernel(DetectInvalidKernel, blocks, threads, 0,
                              d.stream(), values, nnz, invalid_id_flag));
}

__global__ void FusedMultiFunctionalKernel(
    const IndicePair* indices, const int64_t* values, const int64_t nnz,
    const int64_t batch_size, const bool prune, const int64_t default_id,
    int* row_emptiness_flag, int* invalid_id_flag,
    IndicePair* tmp_indices_buffer, int64_t* values_extended) {
  // This kernel will do many things together
  // 1. The first part of threads will do job 1(DetectRowEmptiness), others will
  // do job2(InitBatchRowsBuffer)
  // 2. Do job3 (set values extended to default id)

  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    // do DetectRowEmptiness
    if (prune) {
      const int64_t value = values[offset];
      if (value < 0) {
        // invalid, set invalid_id_flag
        atomicAnd(invalid_id_flag + offset, 0);
      } else {
        // valid, set row_emptiness_flag
        const int64_t row_in_batch = indices[offset].row_in_batch;
        atomicAnd(row_emptiness_flag + row_in_batch, 0);
      }
    } else {
      // set row_emptiness_flag
      const int64_t row_in_batch = indices[offset].row_in_batch;
      atomicAnd(row_emptiness_flag + row_in_batch, 0);
    }
  } else {
    // do InitBatchRowsBuffer
    const int other_offset = offset - nnz;
    if (other_offset < batch_size) {
      tmp_indices_buffer[other_offset].row_in_batch = other_offset;
      // always set entry id to 0;
      tmp_indices_buffer[other_offset].entry_in_column = 0;
    }
  }

  // set values extended to default id
  if (2 * offset + 1 < nnz + batch_size) {
    longlong2 l2 = make_longlong2(default_id, default_id);
    *((longlong2*)(values_extended + 2 * offset)) = l2;
  } else if (2 * offset < nnz + batch_size) {
    values_extended[2 * offset] = default_id;
  }
}

void FusedMultiFunctional(const GPUDevice& d, const IndicePair* indices,
                          const int64_t* values, const int64_t nnz,
                          const int64_t batch_size, const bool prune,
                          const int64_t default_id, int* row_emptiness_flag,
                          int* invalid_id_flag, IndicePair* tmp_indices_buffer,
                          int64_t* values_extended) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(nnz + batch_size, threads);
  TF_CHECK_OK(GpuLaunchKernel(
      FusedMultiFunctionalKernel, blocks, threads, 0, d.stream(), indices,
      values, nnz, batch_size, prune, default_id, row_emptiness_flag,
      invalid_id_flag, tmp_indices_buffer, values_extended));
}

template <bool prune, bool use_sparse_weights>
__global__ void InitFillEmptyBuffersKernel(
    int64_t batch_size, int64_t nnz, int64_t default_id, float default_weight,
    const int64_t* sp_values, const int64_t* sp_indices, int64_t* sp_values_out,
    int64_t* sp_indices_out, float* sp_weights_values_out, bool* is_row_empty,
    int64_t* tmp_indices) {
  const int global_tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (global_tid < batch_size) {
    // init is_row_empty
    is_row_empty[global_tid] = true;
  } else if (global_tid < 3 * batch_size) {
    // init tmp indices
    const int new_global_tid = global_tid - batch_size;
    // even tid keep batch_id, odd tid keep 0
    const int64_t data = ((new_global_tid + 1) % 2) * (new_global_tid / 2);
    tmp_indices[new_global_tid] = data;
  }

  if (global_tid < (batch_size + nnz)) {
    sp_values_out[global_tid] = default_id;
  }

  // using template here to let compiler decide whether to optimize this section
  // out
  if (use_sparse_weights) {
    if (global_tid < (batch_size + nnz)) {
      sp_weights_values_out[global_tid] = default_weight;
    }
  }

  // using template here to let compiler decide whether to optimize this section
  // out
  if (!prune) {
    if (global_tid < nnz) {
      sp_values_out[global_tid] = sp_values[global_tid];
    }

    if (global_tid < 2 * nnz) {
      sp_indices_out[global_tid] = sp_indices[global_tid];
    }
  }
}

void InitFillEmptyBuffers(const GPUDevice& d, const int64_t batch_size,
                          const int64_t nnz, const int64_t default_id,
                          const float default_weight, const bool prune,
                          const bool use_sparse_weights,
                          const int64_t* sp_values, const int64_t* sp_indices,
                          int64_t* sp_values_out, int64_t* sp_indices_out,
                          float* sp_weights_values_out, bool* is_row_empty,
                          int64_t* tmp_indices) {
  const int threads = 32;
  const int blocks = CalcBlocksLinearMapping(
      std::max(std::max(3 * batch_size, batch_size + nnz), 2 * nnz), threads);

#define LAUNCH_KERNEL(prune, use_sparse_weights)                              \
  TF_CHECK_OK(GpuLaunchKernel(                                                \
      InitFillEmptyBuffersKernel<prune, use_sparse_weights>, blocks, threads, \
      0, d.stream(), batch_size, nnz, default_id, default_weight, sp_values,  \
      sp_indices, sp_values_out, sp_indices_out, sp_weights_values_out,       \
      is_row_empty, tmp_indices));

  if (prune && use_sparse_weights) {
    LAUNCH_KERNEL(true, true);
  } else if (prune && !use_sparse_weights) {
    LAUNCH_KERNEL(true, false);
  } else if (!prune && use_sparse_weights) {
    LAUNCH_KERNEL(false, true);
  } else if (!prune && !use_sparse_weights) {
    LAUNCH_KERNEL(false, false);
  }
#undef LAUNCH_KERNEL
}

template <bool prune, bool prune_sparse_weights>
void __global__ DetectEmptyRowKernel(const int64_t* indices,
                                     const int64_t* sp_values,
                                     const float* sp_weights_values,
                                     const int64_t nnz, bool* is_row_empty) {
  const int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_tid < nnz) {
    const int64_t row_in_batch = indices[2 * global_tid];
    // use template for compiler to optimize
    if (prune) {
      if (prune_sparse_weights) {
        if (sp_values[global_tid] >= 0 && sp_weights_values[global_tid] > 0.0) {
          is_row_empty[row_in_batch] = false;
        }
      } else {
        if (sp_values[global_tid] >= 0) {
          is_row_empty[row_in_batch] = false;
        }
      }
    } else {
      is_row_empty[row_in_batch] = false;
    }
  }
}

void DetectEmptyRow(const GPUDevice& d, const int64_t* indices,
                    const int64_t* sp_values, const float* sp_weights_values,
                    const bool prune, const bool prune_sparse_weights,
                    const int64_t nnz, bool* is_row_empty) {
  const int threads = 32;
  const int blocks = CalcBlocksLinearMapping(nnz, threads);

#define LAUNCH_KERNEL(prune, prune_sparse_weights)                           \
  TF_CHECK_OK(GpuLaunchKernel(                                               \
      DetectEmptyRowKernel<prune, prune_sparse_weights>, blocks, threads, 0, \
      d.stream(), indices, sp_values, sp_weights_values, nnz, is_row_empty));

  if (prune) {
    if (prune_sparse_weights) {
      LAUNCH_KERNEL(true, true);
    } else {
      LAUNCH_KERNEL(true, false);
    }
  } else {
    LAUNCH_KERNEL(false, false);
  }
#undef LAUNCH_KERNEL
}

template <typename T>
__global__ void RangeInitKernel(const int64_t length, T* out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    out[idx] = T(idx);
  }
}

template <typename T>
void RangeInit(const GPUDevice& d, const int64_t length, T* out) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(length, threads);
  TF_CHECK_OK(GpuLaunchKernel(RangeInitKernel<T>, blocks, threads, 0,
                              d.stream(), length, out));
}

template void RangeInit<int64_t>(const GPUDevice& d, const int64_t length,
                                 int64_t* out);

__global__ void SumUpEmbeddingShardSinglePartitionKernel(
    const float* emb_shard, const int64_t* indices_before_unique,
    const int* unique_idxs, const float* sp_weights_values,
    const bool use_sparse_weights, const int nnz, const float max_norm,
    const int emb_vec_size, float* emb_vectors, int* feature_nums) {
  __shared__ float l2_sum[1];

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    const int unique_id = unique_idxs[blockIdx.x];
    float emb_element = emb_shard[unique_id * emb_vec_size + threadIdx.x];
    if (max_norm >= 0.0f) {
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      float l2_norm = sqrtf(l2_sum[0]);
      if (l2_norm > max_norm) {
        emb_element *= max_norm / l2_norm;
      }
    }

    if (use_sparse_weights) {
      atomicAdd(emb_vectors + row_in_batch * emb_vec_size + threadIdx.x,
                emb_element * sp_weights_values[blockIdx.x]);
    } else {
      atomicAdd(emb_vectors + row_in_batch * emb_vec_size + threadIdx.x,
                emb_element);
    }

    if (threadIdx.x == 0) {
      atomicAdd(feature_nums + row_in_batch, 1);
    }
  }
}

void SumUpEmbeddingShardSinglePartition(
    const GPUDevice& d, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const float max_norm, const int emb_vec_size,
    float* emb_vectors, int* feature_nums) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(
      GpuLaunchKernel(SumUpEmbeddingShardSinglePartitionKernel, blocks, threads,
                      0, d.stream(), emb_shard, indices_before_unique,
                      unique_idxs, sp_weights_values, use_sparse_weights, nnz,
                      max_norm, emb_vec_size, emb_vectors, feature_nums));
}

__global__ void SumUpEmbeddingShardMultiPartitionKernel(
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const float max_norm, const int emb_vec_size,
    float* emb_vectors, int* feature_nums) {
  __shared__ float l2_sum[1];

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    const int unique_id = unique_idxs[blockIdx.x];
    const int partition_id = partition_permutation[2 * unique_id];
    const int64_t offset_in_partition =
        partition_permutation[2 * unique_id + 1];

    float emb_element =
        ((const float*)(emb_shard_ptrs[partition_id]))[offset_in_partition *
                                                           emb_vec_size +
                                                       threadIdx.x];
    if (max_norm >= 0.0f) {
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      float l2_norm = sqrtf(l2_sum[0]);
      if (l2_norm > max_norm) {
        emb_element *= max_norm / l2_norm;
      }
    }

    if (use_sparse_weights) {
      atomicAdd(emb_vectors + row_in_batch * emb_vec_size + threadIdx.x,
                emb_element * sp_weights_values[blockIdx.x]);
    } else {
      atomicAdd(emb_vectors + row_in_batch * emb_vec_size + threadIdx.x,
                emb_element);
    }

    if (threadIdx.x == 0) {
      atomicAdd(feature_nums + row_in_batch, 1);
    }
  }
}

void SumUpEmbeddingShardMultiPartition(
    const GPUDevice& d, const void* const* emb_shard_ptrs,
    const int* partition_permutation, const int64_t* indices_before_unique,
    const int* unique_idxs, const float* sp_weights_values,
    const bool use_sparse_weights, const int nnz, const float max_norm,
    const int emb_vec_size, float* emb_vectors, int* feature_nums) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(
      SumUpEmbeddingShardMultiPartitionKernel, blocks, threads, 0, d.stream(),
      emb_shard_ptrs, partition_permutation, indices_before_unique, unique_idxs,
      sp_weights_values, use_sparse_weights, nnz, max_norm, emb_vec_size,
      emb_vectors, feature_nums));
}

template <Combiner combiner>
__global__ void ApplyCombinerKernel(const bool* is_row_empty,
                                    const bool set_empty_row_zero,
                                    int* feature_nums, float* emb_vectors) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int feature_num = feature_nums[blockIdx.x];
  if (set_empty_row_zero) {
    if (is_row_empty[blockIdx.x]) {
      feature_nums[blockIdx.x] = 0;
      emb_vectors[offset] = 0.0f;
      return;
    }
  }
  const float emb_element = emb_vectors[offset];
  emb_vectors[offset] = Combine<combiner>(emb_element, feature_num);
}

template <Combiner combiner>
void ApplyCombiner(const GPUDevice& d, const int batch_size,
                   const int emb_vec_size, const bool* is_row_empty,
                   const bool set_empty_row_zero, int* feature_nums,
                   float* emb_vectors) {
  const int blocks = batch_size;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(ApplyCombinerKernel<combiner>, blocks, threads, 0,
                              d.stream(), is_row_empty, set_empty_row_zero,
                              feature_nums, emb_vectors));
}

template void ApplyCombiner<Sqrtn>(const GPUDevice& d, const int batch_size,
                                   const int emb_vec_size,
                                   const bool* is_row_empty,
                                   const bool set_empty_row_zero,
                                   int* feature_nums, float* emb_vectors);
template void ApplyCombiner<Mean>(const GPUDevice& d, const int batch_size,
                                  const int emb_vec_size,
                                  const bool* is_row_empty,
                                  const bool set_empty_row_zero,
                                  int* feature_nums, float* emb_vectors);
template void ApplyCombiner<Sum>(const GPUDevice& d, const int batch_size,
                                 const int emb_vec_size,
                                 const bool* is_row_empty,
                                 const bool set_empty_row_zero,
                                 int* feature_nums, float* emb_vectors);

template <Combiner combiner>
__global__ void DistributeGradToShardSinglePartitionKernel(
    const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, float* grad_shard) {
  __shared__ float l2_sum[1];
  float l2_norm = -1.0f;

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    if (set_empty_row_zero && is_row_empty[row_in_batch]) {
      return;
    }

    const int unique_id = unique_idxs[blockIdx.x];

    if (max_norm >= 0.0f) {
      const float emb_element =
          emb_shard[unique_id * emb_vec_size + threadIdx.x];
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      l2_norm = sqrtf(l2_sum[0]);
    }

    float grad = top_grad[row_in_batch * emb_vec_size + threadIdx.x];
    const int feature_num = feature_nums[row_in_batch];
    grad = CombineGrad<combiner>(grad, feature_num);
    if (use_sparse_weights) {
      grad = grad * sp_weights_values[blockIdx.x];
    }
    if (max_norm >= 0.0f && l2_norm > max_norm) {
      grad *= max_norm / l2_norm;
    }

    atomicAdd(grad_shard + unique_id * emb_vec_size + threadIdx.x, grad);
  }
}

template <Combiner combiner>
void DistributeGradToShardSinglePartition(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, float* grad_shard) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(
      DistributeGradToShardSinglePartitionKernel<combiner>, blocks, threads, 0,
      d.stream(), top_grad, emb_shard, indices_before_unique, unique_idxs,
      sp_weights_values, use_sparse_weights, nnz, emb_vec_size, max_norm,
      set_empty_row_zero, feature_nums, is_row_empty, grad_shard));
}

template void DistributeGradToShardSinglePartition<Sqrtn>(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, float* grad_shard);

template void DistributeGradToShardSinglePartition<Mean>(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, float* grad_shard);

template void DistributeGradToShardSinglePartition<Sum>(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, float* grad_shard);

template <Combiner combiner>
__global__ void DistributeGradToShardMultiPartitionKernel(
    const float* top_grad, const void* const* emb_shard_ptrs,
    const int* partition_permutation, const int64_t* indices_before_unique,
    const int* unique_idxs, const float* sp_weights_values,
    const bool use_sparse_weights, const int nnz, const int emb_vec_size,
    const float max_norm, const bool set_empty_row_zero,
    const int* feature_nums, const bool* is_row_empty, void** grad_shard_ptrs) {
  __shared__ float l2_sum[1];
  float l2_norm = -1.0f;

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    if (set_empty_row_zero && is_row_empty[row_in_batch]) {
      return;
    }
    const int unique_id = unique_idxs[blockIdx.x];
    const int partition_id = partition_permutation[2 * unique_id];
    const int64_t offset_in_partition =
        partition_permutation[2 * unique_id + 1];

    if (max_norm >= 0.0f) {
      float emb_element =
          ((const float*)(emb_shard_ptrs[partition_id]))[offset_in_partition *
                                                             emb_vec_size +
                                                         threadIdx.x];
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      l2_norm = sqrtf(l2_sum[0]);
    }

    float grad = top_grad[row_in_batch * emb_vec_size + threadIdx.x];
    const int feature_num = feature_nums[row_in_batch];
    grad = CombineGrad<combiner>(grad, feature_num);
    if (use_sparse_weights) {
      grad = grad * sp_weights_values[blockIdx.x];
    }
    if (max_norm >= 0.0f && l2_norm > max_norm) {
      grad *= max_norm / l2_norm;
    }

    atomicAdd(((float*)(grad_shard_ptrs[partition_id])) +
                  offset_in_partition * emb_vec_size + threadIdx.x,
              grad);
  }
}

template <Combiner combiner>
void DistributeGradToShardMultiPartition(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, void** grad_shard_ptrs) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(
      DistributeGradToShardMultiPartitionKernel<combiner>, blocks, threads, 0,
      d.stream(), top_grad, emb_shard_ptrs, partition_permutation,
      indices_before_unique, unique_idxs, sp_weights_values, use_sparse_weights,
      nnz, emb_vec_size, max_norm, set_empty_row_zero, feature_nums,
      is_row_empty, grad_shard_ptrs));
}

template void DistributeGradToShardMultiPartition<Sum>(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, void** grad_shard_ptrs);

template void DistributeGradToShardMultiPartition<Mean>(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, void** grad_shard_ptrs);

template void DistributeGradToShardMultiPartition<Sqrtn>(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, void** grad_shard_ptrs);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA