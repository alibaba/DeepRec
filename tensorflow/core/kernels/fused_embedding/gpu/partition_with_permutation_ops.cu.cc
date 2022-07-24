#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/partition_select.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/constant_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class PartitionWithPermutationGPU : public OpKernel {
 public:
  explicit PartitionWithPermutationGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("partition_strategy", &partition_strategy_));
    cudaEventCreateWithFlags(&memcpy_event_, cudaEventDisableTiming);
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));

    Tensor const* input = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input));
    const int64 input_size = input->NumElements();

    Tensor* partition_permutation;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("partition_permutation",
                                             TensorShape{input_size, 2},
                                             &partition_permutation));

    if (partition_strategy_ == "div") {
      OpInputList partition_shapes;
      OP_REQUIRES_OK(ctx,
                     ctx->input_list("partition_shapes", &partition_shapes));

      std::vector<int64_t> accu_div_host;
      accu_div_host.resize(num_partitions_);
      for (int i = 0; i < partition_shapes.size(); i++) {
        OP_REQUIRES(ctx, partition_shapes[i].NumElements() == 2,
                    errors::InvalidArgument(
                        "input partition_shapes must all have 2 elements"));
        const int64_t div = partition_shapes[i].flat<int64>().data()[0];
        accu_div_host[i] = i == 0 ? div : accu_div_host[i - 1] + div;
      }

      Tensor accu_div;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT64, TensorShape({static_cast<int64_t>(num_partitions_)}),
              &accu_div));
      CK_CUDA_THROW_(cudaMemcpyAsync(data_p_with_type<int64>(accu_div),
                                     accu_div_host.data(),
                                     num_partitions_ * sizeof(int64_t),
                                     cudaMemcpyHostToDevice, device.stream()));

      if (input_size < 512) {
        PartitionSelectDiv<int64, int, 64>(
            ctx, input, accu_div, num_partitions_, memcpy_event_,
            partitioned_values, partition_permutation);
      } else if (input_size < 1024) {
        PartitionSelectDiv<int64, int, 128>(
            ctx, input, accu_div, num_partitions_, memcpy_event_,
            partitioned_values, partition_permutation);
      } else if (input_size < 2048) {
        PartitionSelectDiv<int64, int, 256>(
            ctx, input, accu_div, num_partitions_, memcpy_event_,
            partitioned_values, partition_permutation);
      } else if (input_size < 4096) {
        PartitionSelectDiv<int64, int, 512>(
            ctx, input, accu_div, num_partitions_, memcpy_event_,
            partitioned_values, partition_permutation);
      } else {
        PartitionSelectDiv<int64, int, 1024>(
            ctx, input, accu_div, num_partitions_, memcpy_event_,
            partitioned_values, partition_permutation);
      }
    } else if (partition_strategy_ == "mod") {
      if (input_size < 512) {
        PartitionSelectMod<int64, int, 64>(ctx, input, num_partitions_,
                                           memcpy_event_, partitioned_values,
                                           partition_permutation);
      } else if (input_size < 1024) {
        PartitionSelectMod<int64, int, 128>(ctx, input, num_partitions_,
                                            memcpy_event_, partitioned_values,
                                            partition_permutation);
      } else if (input_size < 2048) {
        PartitionSelectMod<int64, int, 256>(ctx, input, num_partitions_,
                                            memcpy_event_, partitioned_values,
                                            partition_permutation);
      } else if (input_size < 4096) {
        PartitionSelectMod<int64, int, 512>(ctx, input, num_partitions_,
                                            memcpy_event_, partitioned_values,
                                            partition_permutation);
      } else {
        PartitionSelectMod<int64, int, 1024>(ctx, input, num_partitions_,
                                             memcpy_event_, partitioned_values,
                                             partition_permutation);
      }
    } else if (partition_strategy_ == "mod_ev") {
      if (input_size < 512) {
        PartitionSelectModEV<int64, int, 64>(ctx, input, num_partitions_,
                                             memcpy_event_, partitioned_values,
                                             partition_permutation);
      } else if (input_size < 1024) {
        PartitionSelectModEV<int64, int, 128>(ctx, input, num_partitions_,
                                              memcpy_event_, partitioned_values,
                                              partition_permutation);
      } else if (input_size < 2048) {
        PartitionSelectModEV<int64, int, 256>(ctx, input, num_partitions_,
                                              memcpy_event_, partitioned_values,
                                              partition_permutation);
      } else if (input_size < 4096) {
        PartitionSelectModEV<int64, int, 512>(ctx, input, num_partitions_,
                                              memcpy_event_, partitioned_values,
                                              partition_permutation);
      } else {
        PartitionSelectModEV<int64, int, 1024>(
            ctx, input, num_partitions_, memcpy_event_, partitioned_values,
            partition_permutation);
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string partition_strategy_;
  cudaEvent_t memcpy_event_;
};

REGISTER_KERNEL_BUILDER(Name("PartitionWithPermutation")
                            .Device(DEVICE_GPU)
                            .HostMemory("partition_shapes"),
                        PartitionWithPermutationGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA