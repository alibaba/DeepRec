#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_KERNELS_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_KERNELS_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"

namespace tensorflow {

namespace fused_embedding {

using GPUDevice = Eigen::GpuDevice;

enum Combiner { Mean, Sum, Sqrtn };

template <Combiner combiner>
__forceinline__ __device__ float Combine(const float in, const int feature_num);

template <>
__forceinline__ __device__ float Combine<Sqrtn>(const float in,
                                                const int feature_num) {
  return in / sqrtf(feature_num);
}

template <>
__forceinline__ __device__ float Combine<Mean>(const float in,
                                               const int feature_num) {
  return in / feature_num;
}

template <>
__forceinline__ __device__ float Combine<Sum>(const float in,
                                              const int feature_num) {
  return in;
}

template <Combiner combiner>
__forceinline__ __device__ float CombineGrad(const float grad,
                                             const int feature_num);

template <>
__forceinline__ __device__ float CombineGrad<Sqrtn>(const float grad,
                                                    const int feature_num) {
  return grad / sqrtf(feature_num);
}

template <>
__forceinline__ __device__ float CombineGrad<Mean>(const float grad,
                                                   const int feature_num) {
  return grad / feature_num;
}

template <>
__forceinline__ __device__ float CombineGrad<Sum>(const float grad,
                                                  const int feature_num) {
  return grad;
}

void InitFlagsToOneInt4(const GPUDevice& d, int length, int* flags);

void DetectInvalid(const GPUDevice& d, const int64_t* values, const int64_t nnz,
                   int* invalid_id_flag);

void FusedMultiFunctional(const GPUDevice& d, const IndicePair* indices,
                          const int64_t* values, const int64_t nnz,
                          const int64_t batch_size, const bool prune_invalid_id,
                          const int64_t default_id, int* row_emptiness_flag,
                          int* invalid_id_flag, IndicePair* tmp_indices_buffer,
                          int64_t* values_extended);

void InitFillEmptyBuffers(const GPUDevice& d, const int64_t batch_size,
                          const int64_t nnz, const int64_t default_id,
                          const float default_weight, const bool prune,
                          const bool use_sparse_weights,
                          const int64_t* sp_values, const int64_t* sp_indices,
                          int64_t* sp_values_out, int64_t* sp_indices_out,
                          float* sp_weights_values_out, bool* is_row_empty,
                          int64_t* tmp_indices);

void DetectEmptyRow(const GPUDevice& d, const int64_t* indices,
                    const int64_t* sp_values, const float* sp_weights_values,
                    const bool prune, const bool prune_sparse_weights,
                    const int64_t nnz, bool* is_row_empty);

template <typename T>
void RangeInit(const GPUDevice& d, const int64_t length, T* out);

void SumUpEmbeddingShardSinglePartition(
    const GPUDevice& d, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const float max_norm, const int emb_vec_size,
    float* emb_vectors, int* feature_nums);

void SumUpEmbeddingShardMultiPartition(
    const GPUDevice& d, const void* const* emb_shard_ptrs,
    const int* partition_permutation, const int64_t* indices_before_unique,
    const int* unique_idxs, const float* sp_weights_values,
    const bool use_sparse_weights, const int nnz, const float max_norm,
    const int emb_vec_size, float* emb_vectors, int* feature_nums);

template <Combiner combiner>
void ApplyCombiner(const GPUDevice& d, const int batch_size,
                   const int emb_vec_size, const bool* is_row_empty,
                   const bool set_empty_row_zero, int* feature_nums,
                   float* emb_vectors);

template <Combiner combiner>
void DistributeGradToShardSinglePartition(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int* unique_idxs,
    const float* sp_weights_values, const bool use_sparse_weights,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const bool* is_row_empty, float* grad_shard);

template <Combiner combiner>
void DistributeGradToShardMultiPartition(
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

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_KERNELS_CU_H_