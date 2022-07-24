#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/constant_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"
#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class PruneInvalidAndFillEmptyRowsGPU : public OpKernel {
 public:
  struct PruneInvalidSelectOp {
    template <typename ThurstTupleT>
    __host__ __device__ __forceinline__ bool operator()(
        ThurstTupleT const& tuple) const {
      return thrust::get<0>(tuple) >= 0;
    }
  };

  struct PruneInvalidWithWeightSelectOp {
    template <typename ThurstTupleT>
    __host__ __device__ __forceinline__ bool operator()(
        ThurstTupleT const& tuple) const {
      return thrust::get<0>(tuple) >= 0 && thrust::get<2>(tuple) > 0;
    }
  };

  explicit PruneInvalidAndFillEmptyRowsGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune", &prune_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64(temp_default_id);

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_sparse_weights", &use_sparse_weights_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("prune_sparse_weights", &prune_sparse_weights_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_weight", &default_weight_));

    cudaEventCreateWithFlags(&memcpy_event_, cudaEventDisableTiming);
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    const int64 default_id = default_id_ >= 0 ? default_id_ : 0;

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    // 1. bind & set inputs, vars, outputs and Init buffers.
    Tensor const* sp_values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &sp_values));
    const int64 nnz = sp_values->shape().dim_size(0);

    Tensor const* sp_indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &sp_indices));

    Tensor const* sp_dense_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &sp_dense_shape));
    const int64 batch_size = sp_dense_shape->flat<int64>().data()[0];

    Tensor const* sp_weights_values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_weights_values", &sp_weights_values));

    if (!prune_ && !fill_empty_row_) {
      ctx->set_output("sp_values_out", *sp_values);
      ctx->set_output("sp_indices_out", *sp_indices);
      ctx->set_output("sp_weights_values_out", *sp_weights_values);
      return;
    }

    Tensor sp_values_out;
    Tensor sp_indices_out;
    Tensor sp_weights_values_out;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT64, TensorShape{nnz + batch_size},
                                      &sp_values_out));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT64, TensorShape{nnz + batch_size, 2},
                                &sp_indices_out));

    if (use_sparse_weights_) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_FLOAT, TensorShape{nnz + batch_size},
                                        &sp_weights_values_out));
    }

    Tensor tmp_indices;
    Tensor* is_row_empty;
    Tensor selected_num_d;

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{batch_size, 2},
                                           &tmp_indices));

    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT32, TensorShape{1}, &selected_num_d));

    if (fill_empty_row_) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output("is_row_empty", TensorShape{batch_size},
                                    &is_row_empty));

      InitFillEmptyBuffers(device, batch_size, nnz, default_id, default_weight_,
                           prune_, use_sparse_weights_,
                           data_p_with_type<const int64_t>(sp_values),
                           data_p_with_type<const int64_t>(sp_indices),
                           data_p_with_type<int64_t>(sp_values_out),
                           data_p_with_type<int64_t>(sp_indices_out),
                           data_p_with_type<float>(sp_weights_values_out),
                           data_p_with_type<bool>(is_row_empty),
                           data_p_with_type<int64_t>(tmp_indices));

      DetectEmptyRow(device, data_p_with_type<const int64_t>(sp_indices),
                     data_p_with_type<const int64_t>(sp_values),
                     data_p_with_type<const float>(sp_weights_values),
                     prune_, prune_sparse_weights_, nnz,
                     data_p_with_type<bool>(is_row_empty));

    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output("is_row_empty", TensorShape{1},
                                               &is_row_empty));
    }

    // 2. Allocate cub tmp

    // nnz = number of non zero
    int new_nnz = nnz;
    Tensor cub_temp_storage;
    size_t max_cub_bytes = 0;
    size_t temp_storage_bytes = 0;
    auto triple_input_iter = thrust::make_zip_iterator(
        thrust::make_tuple(data_p_with_type<const int64_t>(sp_values),
                           data_p_with_type<const IndicePair>(sp_indices),
                           data_p_with_type<const float>(sp_weights_values)));

    auto double_input_iter = thrust::make_zip_iterator(
        thrust::make_tuple(data_p_with_type<const int64_t>(sp_values),
                           data_p_with_type<const IndicePair>(sp_indices)));

    auto triple_output_iter = thrust::make_zip_iterator(
        thrust::make_tuple(data_p_with_type<int64_t>(sp_values_out),
                           data_p_with_type<IndicePair>(sp_indices_out),
                           data_p_with_type<float>(sp_weights_values_out)));

    auto double_output_iter = thrust::make_zip_iterator(
        thrust::make_tuple(data_p_with_type<int64_t>(sp_values_out),
                           data_p_with_type<IndicePair>(sp_indices_out)));

    auto with_weight_select_op = PruneInvalidWithWeightSelectOp();
    auto select_op = PruneInvalidSelectOp();

    if (prune_) {
      if (use_sparse_weights_) {
        if (prune_sparse_weights_) {
          cub::DeviceSelect::If(nullptr, temp_storage_bytes, triple_input_iter,
                                triple_output_iter, (int*)nullptr, int(nnz),
                                with_weight_select_op, device.stream());
        } else {
          cub::DeviceSelect::If(nullptr, temp_storage_bytes, triple_input_iter,
                                triple_output_iter, (int*)nullptr, int(nnz),
                                select_op, device.stream());
        }
      } else {
        cub::DeviceSelect::If(nullptr, temp_storage_bytes, double_input_iter,
                              double_output_iter, (int*)nullptr, int(nnz),
                              select_op, device.stream());
      }
      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;
    }

    if (fill_empty_row_) {
      cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes,
                                 (IndicePair*)nullptr, (int*)nullptr,
                                 (IndicePair*)nullptr, (int*)nullptr,
                                 batch_size, device.stream());
      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;
    }

    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DT_INT8, TensorShape({static_cast<int64>(max_cub_bytes)}),
                 &cub_temp_storage));

    // 3. select valid id & empty row indices
    if (prune_) {
      if (use_sparse_weights_) {
        if (prune_sparse_weights_) {
          cub::DeviceSelect::If(data_p_with_type<void>(cub_temp_storage),
                                max_cub_bytes, triple_input_iter,
                                triple_output_iter,
                                data_p_with_type<int>(selected_num_d), nnz,
                                with_weight_select_op, device.stream());
        } else {
          cub::DeviceSelect::If(data_p_with_type<void>(cub_temp_storage),
                                max_cub_bytes, triple_input_iter,
                                triple_output_iter,
                                data_p_with_type<int>(selected_num_d), nnz,
                                select_op, device.stream());
        }

      } else {
        cub::DeviceSelect::If(data_p_with_type<void>(cub_temp_storage),
                              max_cub_bytes, double_input_iter,
                              double_output_iter,
                              data_p_with_type<int>(selected_num_d), nnz,
                              select_op, device.stream());
      }
      int selected_num;
      CK_CUDA_THROW_(cudaMemcpyAsync(
          &selected_num, data_p_with_type<int>(selected_num_d), sizeof(int),
          cudaMemcpyDeviceToHost, device.stream()));
      CK_CUDA_THROW_(cudaEventRecord(memcpy_event_, device.stream()));
      CK_CUDA_THROW_(cudaEventSynchronize(memcpy_event_));
      new_nnz = selected_num;
    }

    if (fill_empty_row_) {
      cub::DeviceSelect::Flagged(
          data_p_with_type<void>(cub_temp_storage), max_cub_bytes,
          data_p_with_type<const IndicePair>(tmp_indices),
          data_p_with_type<bool>(is_row_empty),
          data_p_with_type<IndicePair>(sp_indices_out) + new_nnz,
          data_p_with_type<int>(selected_num_d), batch_size, device.stream());
      int selected_num;
      CK_CUDA_THROW_(cudaMemcpyAsync(
          &selected_num, data_p_with_type<void>(selected_num_d), sizeof(int),
          cudaMemcpyDeviceToHost, device.stream()));
      CK_CUDA_THROW_(cudaEventRecord(memcpy_event_, device.stream()));
      CK_CUDA_THROW_(cudaEventSynchronize(memcpy_event_));
      new_nnz += selected_num;
    }

    Tensor new_sp_values_out = sp_values_out.Slice(0, new_nnz);
    Tensor new_sp_indices_out = sp_indices_out.Slice(0, new_nnz);

    ctx->set_output("sp_values_out", new_sp_values_out);
    ctx->set_output("sp_indices_out", new_sp_indices_out);

    if (use_sparse_weights_) {
      Tensor new_sp_weights_values_out =
          sp_weights_values_out.Slice(0, new_nnz);
      ctx->set_output("sp_weights_values_out", new_sp_weights_values_out);
    } else {
      Tensor* unused;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("sp_weights_values_out",
                                               TensorShape{1}, &unused));
    }
  }

 private:
  bool fill_empty_row_;
  bool prune_;
  int64 default_id_;
  bool use_sparse_weights_;
  bool prune_sparse_weights_;
  float default_weight_;
  cudaEvent_t memcpy_event_;
};

REGISTER_KERNEL_BUILDER(Name("PruneInvalidAndFillEmptyRows")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        PruneInvalidAndFillEmptyRowsGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA‰