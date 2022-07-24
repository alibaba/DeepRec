#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "cub/thread/thread_operators.cuh"
#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class FusedEmbeddingSparsePostLookUpV2GPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePostLookUpV2GPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_sparse_weights", &use_sparse_weights_));
  }

  void Compute(OpKernelContext* ctx) override {
    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    Tensor const* partition_permutation = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("partition_permutation", &partition_permutation));

    Tensor const* dense_shape_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape_tensor));

    Tensor const* is_row_empty = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("is_row_empty", &is_row_empty));

    Tensor const* indices_before_unique = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("indices_before_unique", &indices_before_unique));

    Tensor const* unique_idxs = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("unique_idxs", &unique_idxs));

    Tensor const* sp_weights_values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_weights_values", &sp_weights_values));

    const int emb_vec_size = emb_shards[0].shape().dim_size(1);
    const int batch_size = dense_shape_tensor->flat<int64>().data()[0];
    const int nnz = indices_before_unique->shape().dim_size(0);
    const bool set_empty_row_zero = default_id_ < 0 && fill_empty_row_;

    // = 1. sum up emb values from emb_shards and dump into output = //
    Tensor* emb_vectors_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("emb_vectors",
                                             TensorShape({int64(batch_size),
                                                          int64(emb_vec_size)}),
                                             &emb_vectors_tensor));
    CK_CUDA_THROW_(cudaMemsetAsync(
        emb_vectors_tensor->flat<float>().data(), 0x0,
        sizeof(float) * emb_vectors_tensor->NumElements(), device.stream()));

    Tensor* feature_nums;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("feature_nums",
                                             TensorShape({int64(batch_size)}),
                                             &feature_nums));
    CK_CUDA_THROW_(cudaMemsetAsync(feature_nums->flat<int>().data(), 0x0,
                                   sizeof(int) * feature_nums->NumElements(),
                                   device.stream()));

    Tensor* emb_shard_ptrs;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("emb_shard_ptrs",
                                        TensorShape({int64(num_partitions_)}),
                                        &emb_shard_ptrs));

    if (num_partitions_ == 1) {
      SumUpEmbeddingShardSinglePartition(
          device, data_p_with_type<const float>(emb_shards[0]),
          data_p_with_type<const int64_t>(indices_before_unique),
          data_p_with_type<const int>(unique_idxs),
          data_p_with_type<float>(sp_weights_values), use_sparse_weights_, nnz,
          max_norm_, emb_vec_size, data_p_with_type<float>(emb_vectors_tensor),
          data_p_with_type<int>(feature_nums));
    } else {
      std::vector<void*> emb_shard_ptrs_host;
      emb_shard_ptrs_host.resize(num_partitions_);
      for (int i = 0; i < num_partitions_; i++) {
        emb_shard_ptrs_host[i] = data_p_with_type<void>(emb_shards[i]);
      }

      CK_CUDA_THROW_(cudaMemcpyAsync(data_p_with_type<void>(emb_shard_ptrs),
                                     emb_shard_ptrs_host.data(),
                                     num_partitions_ * sizeof(size_t),
                                     cudaMemcpyHostToDevice, device.stream()));

      SumUpEmbeddingShardMultiPartition(
          device, data_p_with_type<void*>(emb_shard_ptrs),
          data_p_with_type<int>(partition_permutation),
          data_p_with_type<const int64_t>(indices_before_unique),
          data_p_with_type<const int>(unique_idxs),
          data_p_with_type<float>(sp_weights_values), use_sparse_weights_, nnz,
          max_norm_, emb_vec_size, data_p_with_type<float>(emb_vectors_tensor),
          data_p_with_type<int>(feature_nums));
    }

    // ================================================================ //

    // ========================= 2. combiner ========================== //
    if (combiner_ == "sqrtn") {
      ApplyCombiner<Sqrtn>(device, batch_size, emb_vec_size,
                           data_p_with_type<const bool>(is_row_empty),
                           set_empty_row_zero,
                           data_p_with_type<int>(feature_nums),
                           data_p_with_type<float>(emb_vectors_tensor));
    } else if (combiner_ == "mean") {
      ApplyCombiner<Mean>(device, batch_size, emb_vec_size,
                          data_p_with_type<const bool>(is_row_empty),
                          set_empty_row_zero,
                          data_p_with_type<int>(feature_nums),
                          data_p_with_type<float>(emb_vectors_tensor));
    } else {
      ApplyCombiner<Sum>(device, batch_size, emb_vec_size,
                         data_p_with_type<const bool>(is_row_empty),
                         set_empty_row_zero,
                         data_p_with_type<int>(feature_nums),
                         data_p_with_type<float>(emb_vectors_tensor));
    }
    // ================================================================ //
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
  bool fill_empty_row_;
  int64_t default_id_;
  bool use_sparse_weights_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingSparsePostLookUpV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        FusedEmbeddingSparsePostLookUpV2GPU);

class FusedEmbeddingSparsePostLookUpV2GradGPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePostLookUpV2GradGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_sparse_weights", &use_sparse_weights_));
  }

  void Compute(OpKernelContext* ctx) override {
    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    Tensor const* top_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_grad", &top_grad_tensor));

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    Tensor const* emb_shard_ptrs = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_shard_ptrs", &emb_shard_ptrs));

    Tensor const* partition_permutation = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("partition_permutation", &partition_permutation));

    Tensor const* feature_nums = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("feature_nums", &feature_nums));

    Tensor const* indices_before_unique = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("indices_before_unique", &indices_before_unique));

    Tensor const* unique_idxs = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("unique_idxs", &unique_idxs));

    Tensor const* is_row_empty = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("is_row_empty", &is_row_empty));

    Tensor const* sp_weights_values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_weights_values", &sp_weights_values));

    OpOutputList grad_shards;
    OP_REQUIRES_OK(ctx, ctx->output_list("grad_shards", &grad_shards));

    const int batch_size = top_grad_tensor->shape().dim_size(0);
    const int emb_vec_size = emb_shards[0].shape().dim_size(1);
    const int nnz = indices_before_unique->shape().dim_size(0);
    const bool set_empty_row_zero = default_id_ < 0 && fill_empty_row_;

    std::vector<void*> grad_shard_ptrs_host;
    grad_shard_ptrs_host.resize(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      Tensor* grad_out;
      grad_shards.allocate(i, emb_shards[i].shape(), &grad_out);
      grad_shard_ptrs_host[i] = data_p_with_type<void>(grad_out);
      CK_CUDA_THROW_(cudaMemsetAsync(data_p_with_type<void>(grad_out), 0x0,
                                     sizeof(float) * grad_out->NumElements(),
                                     device.stream()));
    }

    if (num_partitions_ == 1) {
      if (combiner_ == "mean") {
        DistributeGradToShardSinglePartition<Mean>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<const float>(emb_shards[0]),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int>(unique_idxs),
            data_p_with_type<const float>(sp_weights_values),
            use_sparse_weights_, nnz, emb_vec_size, max_norm_,
            set_empty_row_zero, data_p_with_type<const int>(feature_nums),
            data_p_with_type<const bool>(is_row_empty),
            data_p_with_type<float>(grad_shards[0]));
      } else if (combiner_ == "sqrt") {
        DistributeGradToShardSinglePartition<Sqrtn>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<const float>(emb_shards[0]),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int>(unique_idxs),
            data_p_with_type<const float>(sp_weights_values),
            use_sparse_weights_, nnz, emb_vec_size, max_norm_,
            set_empty_row_zero, data_p_with_type<const int>(feature_nums),
            data_p_with_type<const bool>(is_row_empty),
            data_p_with_type<float>(grad_shards[0]));
      } else {
        DistributeGradToShardSinglePartition<Sum>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<const float>(emb_shards[0]),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int>(unique_idxs),
            data_p_with_type<const float>(sp_weights_values),
            use_sparse_weights_, nnz, emb_vec_size, max_norm_,
            set_empty_row_zero, data_p_with_type<const int>(feature_nums),
            data_p_with_type<const bool>(is_row_empty),
            data_p_with_type<float>(grad_shards[0]));
      }

    } else {
      Tensor grad_shard_ptrs;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                              DT_UINT64, TensorShape({int64(num_partitions_)}),
                              &grad_shard_ptrs));
      CK_CUDA_THROW_(cudaMemcpyAsync(data_p_with_type<void>(grad_shard_ptrs),
                                     grad_shard_ptrs_host.data(),
                                     num_partitions_ * sizeof(size_t),
                                     cudaMemcpyHostToDevice, device.stream()));

      if (combiner_ == "mean") {
        DistributeGradToShardMultiPartition<Mean>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<void*>(emb_shard_ptrs),
            data_p_with_type<const int>(partition_permutation),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int>(unique_idxs),
            data_p_with_type<const float>(sp_weights_values),
            use_sparse_weights_, nnz, emb_vec_size, max_norm_,
            set_empty_row_zero, data_p_with_type<const int>(feature_nums),
            data_p_with_type<const bool>(is_row_empty),
            data_p_with_type<void*>(grad_shard_ptrs));
      } else if (combiner_ == "sqrt") {
        DistributeGradToShardMultiPartition<Sqrtn>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<void*>(emb_shard_ptrs),
            data_p_with_type<const int>(partition_permutation),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int>(unique_idxs),
            data_p_with_type<const float>(sp_weights_values),
            use_sparse_weights_, nnz, emb_vec_size, max_norm_,
            set_empty_row_zero, data_p_with_type<const int>(feature_nums),
            data_p_with_type<const bool>(is_row_empty),
            data_p_with_type<void*>(grad_shard_ptrs));
      } else {
        DistributeGradToShardMultiPartition<Sum>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<void*>(emb_shard_ptrs),
            data_p_with_type<const int>(partition_permutation),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int>(unique_idxs),
            data_p_with_type<const float>(sp_weights_values),
            use_sparse_weights_, nnz, emb_vec_size, max_norm_,
            set_empty_row_zero, data_p_with_type<const int>(feature_nums),
            data_p_with_type<const bool>(is_row_empty),
            data_p_with_type<void*>(grad_shard_ptrs));
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
  bool fill_empty_row_;
  int64_t default_id_;
  bool use_sparse_weights_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingSparsePostLookUpV2Grad").Device(DEVICE_GPU),
    FusedEmbeddingSparsePostLookUpV2GradGPU);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA