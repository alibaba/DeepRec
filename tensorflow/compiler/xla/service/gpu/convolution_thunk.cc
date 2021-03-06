/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/profiler/nvtx_utils.h"

namespace xla {
namespace gpu {

ConvolutionThunk::ConvolutionThunk(
    const HloCustomCallInstruction* cudnn_call,
    std::vector<BufferAllocation::Slice> operand_slices,
    BufferAllocation::Slice result_slice, BufferAllocation::Slice scratch_slice,
    BufferAllocation::Slice tuple_result_slice)
    : Thunk(Kind::kConvolution, cudnn_call),
      cudnn_call_(cudnn_call),
      operand_buffers_(std::move(operand_slices)),
      result_buffer_(result_slice),
      scratch_buffer_(scratch_slice),
      tuple_result_buffer_(tuple_result_slice) {}

Status ConvolutionThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  std::vector<se::DeviceMemoryBase> operand_se_buffers;
  for (const auto& buffer : operand_buffers_) {
    operand_se_buffers.push_back(buffer_allocations.GetDeviceAddress(buffer));
  }

  se::DeviceMemoryBase result_buffer =
      buffer_allocations.GetDeviceAddress(result_buffer_);

  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  tensorflow::nvtx::ScopedRangeIfEnabled<tensorflow::nvtx::CoreDomain>
      nvtx_range(cudnn_call_->metadata().op_type(), [&]() {
        return tensorflow::nvtx::GetThunkExecutionRangeMessage(
            cudnn_call_->GetModule()->name(),
            cudnn_call_->metadata().op_name());
      });
  TF_RETURN_IF_ERROR(RunGpuConv(cudnn_call_, absl::MakeSpan(operand_se_buffers),
                                result_buffer, scratch, params.stream));

  // Write the output tuple.
  const int kNumOutputs = 2;
  auto ptrs = absl::make_unique<void*[]>(kNumOutputs);
  ptrs[0] = result_buffer.opaque();
  ptrs[1] = scratch.opaque();
  se::DeviceMemory<void*> tuple_addr(
      buffer_allocations.GetDeviceAddress(tuple_result_buffer_));
  SafeH2DMemcpy(tuple_addr, std::move(ptrs), kNumOutputs, params.stream,
                params.deferred_host_callbacks);

  if (!params.stream->ok()) {
    return InternalError("ConvolutionThunk::ExecuteOnStream failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
