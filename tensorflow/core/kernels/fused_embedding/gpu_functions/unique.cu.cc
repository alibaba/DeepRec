#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/gpu_functions/unique.cu.h"

#include <cub/cub.cuh>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

// Returns true iff index is at the end of a segment (which is equivalent to the
// beginning of the next segment).
template <typename T, typename TIndex>
struct SegmentIndicatorFunctor {
  const T* __restrict__ sorted_input_ptr_;
  SegmentIndicatorFunctor(const T* sorted_input_ptr)
      : sorted_input_ptr_(sorted_input_ptr) {}
  __device__ bool operator()(const TIndex& i) const {
    return i > 0 && sorted_input_ptr_[i] != sorted_input_ptr_[i - 1];
  }
};

template <typename TIndex>
__global__ void ExtractFirstOccurrenceIndicesKernel(
    int64 input_size, int64 uniq_size,
    const TIndex* __restrict__ sorted_input_inds,
    const TIndex* __restrict__ sorted_input_unique_ids,
    TIndex* __restrict__ unique_input_inds, TIndex* __restrict__ segment_ends) {
  GPU_1D_KERNEL_LOOP(i, input_size) {
    TIndex sorted_input_unique_id = sorted_input_unique_ids[i];
    if (i == 0 || sorted_input_unique_id != sorted_input_unique_ids[i - 1]) {
      unique_input_inds[sorted_input_unique_id] = sorted_input_inds[i];
      if (segment_ends) {
        if (i == 0) {
          // First thread writes the last element.
          segment_ends[uniq_size - 1] = input_size;
        } else {
          segment_ends[sorted_input_unique_id - 1] = i;
        }
      }
    }
  }
}

// Scatters the index of the first occurrence of each unique input value to
// unique_input_inds.
// If segment_ends is not nullptr, it is filled with the end index of each
// unique value's range in the sorted input (the last element is always set
// to input_size).
template <typename TIndex>
Status ExtractFirstOccurrenceIndices(const GPUDevice& d, int64 input_size,
                                     int64 uniq_size,
                                     const TIndex* sorted_input_inds,
                                     const TIndex* sorted_input_unique_ids,
                                     TIndex* unique_input_inds,
                                     TIndex* segment_ends) {
  CHECK_GT(input_size, 0);  // Crash OK
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_size, d, &ExtractFirstOccurrenceIndicesKernel<TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(ExtractFirstOccurrenceIndicesKernel<TIndex>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), input_size, uniq_size, sorted_input_inds,
                         sorted_input_unique_ids, unique_input_inds,
                         segment_ends);
}

template <typename T, typename TIndex>
__global__ void GatherOutputsAndInvertPermutationKernel(
    int64 uniq_size, const T* __restrict__ input,
    const TIndex* __restrict__ sorted_unique_input_inds,
    const TIndex* __restrict__ sorted_unique_perm,
    const TIndex* __restrict__ segment_ends, T* __restrict__ output,
    TIndex* __restrict__ inv_sorted_unique_perm, TIndex* __restrict__ count) {
  GPU_1D_KERNEL_LOOP(i, uniq_size) {
    output[i] = input[sorted_unique_input_inds[i]];
    auto j = sorted_unique_perm[i];
    inv_sorted_unique_perm[j] = i;
    if (count) {
      TIndex beg = j == 0 ? 0 : segment_ends[j - 1];
      TIndex end = segment_ends[j];
      count[i] = end - beg;
    }
  }
}

// Gathers input values using sorted_unique_input_inds, and inverts the
// permutation specified by sorted_unique_perm.
template <typename T, typename TIndex>
Status GatherOutputsAndInvertPermutation(const GPUDevice& d, int64 uniq_size,
                                         const T* input,
                                         const TIndex* sorted_unique_input_inds,
                                         const TIndex* sorted_unique_perm,
                                         const TIndex* segment_ends, T* output,
                                         TIndex* inv_sorted_unique_perm,
                                         TIndex* count) {
  if (uniq_size == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      uniq_size, d, &GatherOutputsAndInvertPermutationKernel<T, TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(GatherOutputsAndInvertPermutationKernel<T, TIndex>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), uniq_size, input, sorted_unique_input_inds,
                         sorted_unique_perm, segment_ends, output,
                         inv_sorted_unique_perm, count);
}

template <typename TIndex>
__global__ void LookupAndScatterUniqueIdsKernel(
    int64 input_size, const TIndex* sorted_input_inds,
    const TIndex* __restrict__ sorted_input_unique_ids,
    const TIndex* __restrict__ inv_sorted_unique_perm,
    TIndex* __restrict__ idx) {
  GPU_1D_KERNEL_LOOP(i, input_size) {
    idx[sorted_input_inds[i]] =
        inv_sorted_unique_perm[sorted_input_unique_ids[i]];
  }
}

// Maps the values of sorted_input_unique_ids and scatters them to idx using
// sorted_input_inds.
template <typename TIndex>
Status LookupAndScatterUniqueIds(const GPUDevice& d, int64 input_size,
                                 const TIndex* sorted_input_inds,
                                 const TIndex* sorted_input_unique_ids,
                                 const TIndex* inv_sorted_unique_perm,
                                 TIndex* idx) {
  CHECK_GT(input_size, 0);  // Crash OK
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_size, d, &LookupAndScatterUniqueIdsKernel<TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(LookupAndScatterUniqueIdsKernel<TIndex>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), input_size, sorted_input_inds,
                         sorted_input_unique_ids, inv_sorted_unique_perm, idx);
}

template <typename T>
__global__ void RangeInitKernel(const T start, const T delta, const T size,
                                T* out) {
  GPU_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}

// Initialize out with range start, start + delta, start + 2 * delta, ...
template <typename T>
Status RangeInit(const Eigen::GpuDevice& d, const T start, const T delta,
                 const T size, T* out) {
  if (size == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  return GpuLaunchKernel(RangeInitKernel<T>, config.block_count,
                         config.thread_per_block, 0, d.stream(), start, delta,
                         size, out);
}

template <typename TIndex>
__global__ void GetIdxOfInputToUniqueKernel(
    const TIndex* count_ptr, const TIndex* segment_ends_ptr,
    const TIndex* sorted_unique_perm_ptr, const TIndex* count_prefix_sum_ptr,
    const TIndex* sorted_input_inds_ptr, TIndex* idx_of_input_to_unique_ptr) {
  const int target_unique_key_idx = blockIdx.x;
  const TIndex unique_key_count = count_ptr[target_unique_key_idx];
  int offset_of_sorted_input_inds =
      segment_ends_ptr[sorted_unique_perm_ptr[target_unique_key_idx]] -
      unique_key_count;
  for (int j = threadIdx.x; j < unique_key_count; j += blockDim.x) {
    int result_offset = count_prefix_sum_ptr[target_unique_key_idx] + j;
    idx_of_input_to_unique_ptr[result_offset] =
        sorted_input_inds_ptr[offset_of_sorted_input_inds + j];
  }
}

template <typename TIndex>
Status GetIdxOfInputToUnique(const Eigen::GpuDevice& d, const int64 uniq_size,
                             const TIndex* count_ptr,
                             const TIndex* segment_ends_ptr,
                             const TIndex* sorted_unique_perm_ptr,
                             const TIndex* count_prefix_sum_ptr,
                             const TIndex* sorted_input_inds_ptr,
                             TIndex* idx_of_input_to_unique_ptr) {
  if (uniq_size == 0) return Status::OK();
  const int64 blocks = uniq_size;
  const int64 threads = 32;
  return GpuLaunchKernel(GetIdxOfInputToUniqueKernel<TIndex>, blocks, threads,
                         0, d.stream(), count_ptr, segment_ends_ptr,
                         sorted_unique_perm_ptr, sorted_unique_perm_ptr,
                         sorted_input_inds_ptr, idx_of_input_to_unique_ptr);
}

// Computes keys_out = sorted(keys_in), and indices_out = argsort(keys_in).
// If keys_out is not required, it can be set to nullptr.
// If indices_in is nullptr, the range of input indices [0, size) will be
// used.
template <typename Tkey, typename Tindex>
Status GpuRadixSort(OpKernelContext* context, int size, const Tkey* keys_in,
                    Tkey* keys_out,            // Optional
                    const Tindex* indices_in,  // Optional
                    Tindex* indices_out, int num_bits = sizeof(Tkey) * 8) {
  if (size == 0) return Status::OK();
  // Allocate temporary inputs/outputs if necessary.
  Tensor tmp_indices_in;
  if (!indices_in) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Tindex>::value, TensorShape({size}), &tmp_indices_in));
    Tindex* mutable_indices_in = tmp_indices_in.flat<Tindex>().data();
    indices_in = mutable_indices_in;
    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    // Initialize indices_in to the input index range.
    TF_RETURN_IF_ERROR(RangeInit(device, Tindex(0), Tindex(1), Tindex(size),
                                 mutable_indices_in));
  }
  Tensor tmp_keys_out;
  if (!keys_out) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Tkey>::value, TensorShape({size}), &tmp_keys_out));
    keys_out = tmp_keys_out.flat<Tkey>().data();
  }
  // Determine temporary device storage requirements.
  Tensor temp_storage;
  size_t temp_storage_bytes = 0;
  const auto& cu_stream = GetGpuStream(context);
  auto err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                             keys_in, keys_out, indices_in,
                                             indices_out, size, /*begin_bit=*/0,
                                             /*end_bit=*/num_bits, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceRadixSort::SortPairs to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  // Allocate temporary storage.
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
      &temp_storage));
  // Sort indices by keys.
  err = cub::DeviceRadixSort::SortPairs(
      temp_storage.flat<int8>().data(), temp_storage_bytes, keys_in, keys_out,
      indices_in, indices_out, size,
      /*begin_bit=*/0, /*end_bit=*/num_bits, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceRadixSort::SortPairs, "
        "temp_storage_bytes: ",
        temp_storage_bytes, "status: ", cudaGetErrorString(err));
  }
  return Status::OK();
}

enum PrefixSumType { inclusive, exclusive };

template <typename InputIteratorT, typename OutputIteratorT>
Status GpuPrefixSum(OpKernelContext* context, int size,
                    PrefixSumType prefix_sum_type, InputIteratorT input,
                    OutputIteratorT output) {
  static_assert(
      !std::is_same<typename std::remove_reference<decltype(*input)>::type,
                    bool>::value,
      "GpuInclusivePrefixSum does not work correct with booleans, please use "
      "TransformInputIterator to explicitly cast to an integer.");
  if (size == 0) return Status::OK();
  const auto& cu_stream = GetGpuStream(context);
  size_t temp_storage_bytes;

  cudaError_t err;
  if (prefix_sum_type == inclusive) {
    err = cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, input,
                                        output, size, cu_stream);
  } else {
    // exclusive
    err = cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, input,
                                        output, size, cu_stream);
  }

  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceScan::(Ex)InclusiveSum to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
      &temp_storage));

  if (prefix_sum_type == inclusive) {
    err = cub::DeviceScan::InclusiveSum(temp_storage.flat<int8>().data(),
                                        temp_storage_bytes, input, output, size,
                                        cu_stream);
  } else {
    err = cub::DeviceScan::ExclusiveSum(temp_storage.flat<int8>().data(),
                                        temp_storage_bytes, input, output, size,
                                        cu_stream);
  }
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceScan::(Ex)InclusiveSum, "
        "temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return Status::OK();
}

// Helper class to allocate scratch memory and keep track of debug info.
// Mostly a thin wrapper around Tensor & allocate_temp.
template <typename Scalar>
class ScratchSpace {
 public:
  ScratchSpace(OpKernelContext* context, int64 size, bool on_host)
      : ScratchSpace(context, TensorShape({size}), "", on_host) {}

  ScratchSpace(OpKernelContext* context, int64 size, const string& debug_info,
               bool on_host)
      : ScratchSpace(context, TensorShape({size}), debug_info, on_host) {}

  ScratchSpace(OpKernelContext* context, const TensorShape& shape,
               const string& debug_info, bool on_host)
      : context_(context), debug_info_(debug_info), on_host_(on_host) {
    AllocatorAttributes alloc_attr;
    if (on_host) {
      // Allocate pinned memory on the host to avoid unnecessary
      // synchronization.
      alloc_attr.set_on_host(true);
      alloc_attr.set_gpu_compatible(true);
    }
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<Scalar>::value, shape,
                                       &scratch_tensor_, alloc_attr));
  }

  virtual ~ScratchSpace() {}

  Scalar* mutable_data() {
    return scratch_tensor_.template flat<Scalar>().data();
  }
  const Scalar* data() const {
    return scratch_tensor_.template flat<Scalar>().data();
  }
  Scalar& operator()(int64 i) {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  const Scalar& operator()(int64 i) const {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  int64 bytes() const { return scratch_tensor_.TotalBytes(); }
  int64 size() const { return scratch_tensor_.NumElements(); }
  const string& debug_info() const { return debug_info_; }

  Tensor& tensor() { return scratch_tensor_; }
  const Tensor& tensor() const { return scratch_tensor_; }

  // Returns true if this ScratchSpace is in host memory.
  bool on_host() const { return on_host_; }

 protected:
  OpKernelContext* context() const { return context_; }

 private:
  OpKernelContext* context_;  // not owned
  const string debug_info_;
  const bool on_host_;
  Tensor scratch_tensor_;
};

template <typename U>
void AllocateTemp(OpKernelContext* context, int64 size, Tensor* tensor,
                  U** tensor_data) {
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 TensorShape({size}), tensor));
  *tensor_data = tensor->flat<U>().data();
}

}  // namespace

namespace fused_embedding {

// Extention to TensorFlow 2.x's UniqueWithCount operator
template <typename T, typename TIndex>
void UniqueWithCountsGPU(OpKernelContext* context, const Tensor* input,
                         Tensor* unique_keys, Tensor* unique_idxs_out,
                         Tensor* unique_counts_out,
                         Tensor* idx_of_input_to_unique_out,
                         Tensor* unique_offsets_out) {
  OP_REQUIRES(context,
              input->NumElements() <= std::numeric_limits<int32>::max(),
              errors::InvalidArgument(
                  "unique does not support input tensors larger than ",
                  std::numeric_limits<int32>::max(), " elements"));

  OP_REQUIRES(context, TensorShapeUtils::IsVector(input->shape()),
              errors::InvalidArgument("unique expects a 1D vector."));

  se::Stream* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

  cudaEvent_t memcpy_event;
  cudaEventCreateWithFlags(&memcpy_event, cudaEventDisableTiming);

  int64 input_size = input->NumElements();
  if (input_size == 0) {
    // Early exit for trivial case.
    Tensor* temp;
    OP_REQUIRES_OK(context, context->allocate_output("unique_idxs",
                                                     TensorShape({0}), &temp));
    OP_REQUIRES_OK(context, context->allocate_output("unique_counts",
                                                     TensorShape({0}), &temp));
    OP_REQUIRES_OK(context, context->allocate_output("idx_of_input_to_unique",
                                                     TensorShape({0}), &temp));
    return;
  }

  // The algorithm implemented here is as follows:
  // input = [3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6]
  // 1) Sort the input to group equal values together in segments.
  //      sorted_input, sorted_input_inds = sort(input)
  // sorted_input:
  //   [1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8, 9]
  // sorted_input_inds:
  //   [4, 17, 0, 2, 9, 3, 5, 14, 16, 1, 10, 18, 8, 15, 19, 11, 7, 12, 13, 6]
  // 2) Identify the boundaries between segments and use prefix sum to
  //    compute the unique ID for each sorted value.
  //      sorted_input_unique_ids = prefix_sum(indicator(sorted_input))
  // indicator(sorted_input):
  //   [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  // sorted_input_unique_ids:
  //   [0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7, 7, 7, 8]
  // 3) Extract the input index of the first occurrence of each unique value.
  //    If counts are required, also extract the end index of each segment.
  //      unique_input_inds[sorted_input_unique_ids] =
  //          sorted_input_inds (@ indicator)
  //      segment_ends[sorted_input_unique_ids[i] - 1] = i (@ indicator)
  // unique_input_inds: [4, 17, 0, 3, 1, 8, 11, 7, 6]
  // segment_ends: [1, 2, 5, 9, 12, 15, 16, 19, 20]
  // 4) Sort the extracted unique input indices to put them in order of
  //    first appearance.
  //      sorted_unique_input_inds, sorted_unique_perm =
  //          sort(unique_input_inds)
  // sorted_unique_input_inds: [0, 1, 3, 4, 6, 7, 8, 11, 17]
  // sorted_unique_perm: [2, 4, 3, 0, 8, 7, 5, 6, 1]
  // 5) Gather the sorted unique input values to produce output, and invert
  //    the second sort permutation to produce an inverse ID mapping. If
  //    counts are required, also take the adjacent difference between
  //    segment_ends indices to produce counts.
  //      output = input[sorted_unique_input_inds]
  //      inv_sorted_unique_perm[sorted_unique_perm[i]] = i
  //      counts = adjacent_difference(segment_ends)
  // output: [3, 5, 4, 1, 9, 8, 6, 7, 2]
  // inv_sorted_unique_perm: [3, 8, 0, 2, 1, 6, 7, 5, 4]
  // counts: [3, 3, 4, 1, 1, 3, 3, 1, 1]
  // 6) calculate prefix sum of counts
  // epsc = exclusive_prefix_sum_of_counts = [3, 6, 10, 11, 12, 15, 18, 19,
  // 20] 7) Use custom kernel to calculate the mapping between output and
  // corresponding indices of the input.
  // pseudo code of this kernel:
  // suii = sorted_unique_input_inds;
  // sup = sorted_unique_perm;
  // epsc = exclusive_prefix_sum_of_counts;
  // GetIdxOfInputToUnique kernel
  //   blocks = len(suii)
  //   threads = 32 or 64 or else
  //   each block i:
  //     target_count = counts[i]
  //     offset = segment_ends[sup[i]] - target_count
  //     for each thread j : (j <= target_count)
  //       idx_of_input_to_unique_ptr[epsc[i] + j] = sorted_input_inds[offset
  //       + j]
  // idx_of_input_to_unique_ptr: [0, 2, 9, 1, 10, 18, 3, 5, 14, 16, 4, 6, 7,
  //                              12, 13, 8, 15, 19, 11, 17]

  /*  REVISE: below not necessary
  // 6) Look up unique IDs via the inverse ID mapping and scatter them
  using
  //    the original sort permutation to produce the indices output.
  //      idx[sorted_input_inds] =
  //          inv_sorted_unique_perm[sorted_input_unique_ids]
  // idx: [0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6]
  */

  Tensor sorted_input_inds;
  TIndex* sorted_input_inds_ptr = nullptr;
  AllocateTemp(context, input_size, &sorted_input_inds, &sorted_input_inds_ptr);
  if (!context->status().ok()) return;

  Tensor sorted_input;
  T* sorted_input_ptr = nullptr;
  AllocateTemp(context, input_size, &sorted_input, &sorted_input_ptr);
  if (!context->status().ok()) return;

  const T* input_ptr = input->flat<T>().data();
  OP_REQUIRES_OK(
      context, GpuRadixSort(context, input_size, /*keys_in=*/input_ptr,
                            /*keys_out=*/sorted_input_ptr,
                            /*indices_in=*/static_cast<const TIndex*>(nullptr),
                            /*indices_out=*/sorted_input_inds_ptr));

  // Create a fancy input iterator to indicate segment boundaries.
  cub::TransformInputIterator<TIndex, SegmentIndicatorFunctor<T, TIndex>,
                              cub::CountingInputIterator<TIndex>>
      segment_indicator_iter(0, {sorted_input_ptr});

  Tensor sorted_input_unique_ids;
  TIndex* sorted_input_unique_ids_ptr = nullptr;
  AllocateTemp(context, input_size, &sorted_input_unique_ids,
               &sorted_input_unique_ids_ptr);
  if (!context->status().ok()) return;

  OP_REQUIRES_OK(context, GpuPrefixSum(context, input_size, inclusive,
                                       segment_indicator_iter,
                                       sorted_input_unique_ids_ptr));

  // Copy the last element of sorted_input_unique_ids back to the host to
  // obtain uniq_size.
  ScratchSpace<TIndex> last_idx_host(context, 1, /*on_host=*/true);
  auto status =
      stream
          ->ThenMemcpy(last_idx_host.mutable_data(),
                       se::DeviceMemoryBase(
                           const_cast<TIndex*>(sorted_input_unique_ids_ptr) +
                               (input_size - 1),
                           sizeof(*last_idx_host.data())),
                       sizeof(*last_idx_host.data()))
          .ok();
  if (!status) {
    context->SetStatus(errors::Internal("Copying device-to-host failed."));
  }

  const GPUDevice& device = context->eigen_gpu_device();
  cudaEventRecord(memcpy_event, device.stream());
  cudaEventSynchronize(memcpy_event);
  int64 uniq_size = (*last_idx_host.data()) + 1;

  se::cuda::ScopedActivateExecutorContext scoped_activation{
      context->op_device_context()->stream()->parent()};

  Tensor unique_input_inds;
  TIndex* unique_input_inds_ptr = nullptr;
  AllocateTemp(context, uniq_size, &unique_input_inds, &unique_input_inds_ptr);
  if (!context->status().ok()) return;

  Tensor segment_ends;
  TIndex* segment_ends_ptr = nullptr;
  AllocateTemp(context, uniq_size, &segment_ends, &segment_ends_ptr);
  if (!context->status().ok()) return;

  OP_REQUIRES_OK(context,
                 ExtractFirstOccurrenceIndices(
                     device, input_size, uniq_size, sorted_input_inds_ptr,
                     sorted_input_unique_ids_ptr, unique_input_inds_ptr,
                     segment_ends_ptr));

  Tensor sorted_unique_input_inds;
  TIndex* sorted_unique_input_inds_ptr = nullptr;
  AllocateTemp(context, uniq_size, &sorted_unique_input_inds,
               &sorted_unique_input_inds_ptr);
  if (!context->status().ok()) return;

  Tensor sorted_unique_perm;
  TIndex* sorted_unique_perm_ptr = nullptr;
  AllocateTemp(context, uniq_size, &sorted_unique_perm,
               &sorted_unique_perm_ptr);
  if (!context->status().ok()) return;

  // Sort by input index so that output is in order of appearance.
  OP_REQUIRES_OK(
      context, GpuRadixSort(context, uniq_size,
                            /*keys_in=*/unique_input_inds_ptr,
                            /*keys_out=*/sorted_unique_input_inds_ptr,
                            /*indices_in=*/static_cast<const TIndex*>(nullptr),
                            /*indices_out=*/sorted_unique_perm_ptr,
                            /*num_bits=*/Log2Ceiling(input_size)));

  // Free temporary tensor that is no longer needed.
  unique_input_inds = Tensor();
  unique_input_inds_ptr = nullptr;

  // output 0 unique_keys
  T* output_ptr = nullptr;
  AllocateTemp(context, uniq_size, unique_keys, &output_ptr);

  Tensor inv_sorted_unique_perm;
  TIndex* inv_sorted_unique_perm_ptr = nullptr;
  AllocateTemp(context, uniq_size, &inv_sorted_unique_perm,
               &inv_sorted_unique_perm_ptr);
  if (!context->status().ok()) return;

  // output 2 unique_counts_out
  OP_REQUIRES_OK(context, context->allocate_output("unique_counts",
                                                   TensorShape({uniq_size}),
                                                   &unique_counts_out));
  TIndex* count_ptr = unique_counts_out->flat<TIndex>().data();

  // Compute output and counts (if necessary).
  OP_REQUIRES_OK(context,
                 GatherOutputsAndInvertPermutation(
                     device, uniq_size, input_ptr, sorted_unique_input_inds_ptr,
                     sorted_unique_perm_ptr, segment_ends_ptr, output_ptr,
                     inv_sorted_unique_perm_ptr, count_ptr));

  // Free temporary tensors that are no longer needed.
  sorted_unique_input_inds = Tensor();
  sorted_unique_input_inds_ptr = nullptr;

  // Compute prefix sum of counts
  // also output 4 unique_offsets_out
  OP_REQUIRES_OK(context, context->allocate_output("unique_offsets",
                                                   TensorShape({uniq_size}),
                                                   &unique_offsets_out));
  TIndex* count_prefix_sum_ptr = unique_offsets_out->flat<TIndex>().data();

  OP_REQUIRES_OK(context, GpuPrefixSum(context, uniq_size, exclusive, count_ptr,
                                       count_prefix_sum_ptr));

  // GetIdxOfInputToUnique kernel to calculate the mapping between output and
  // corresponding indices of the input

  // output 3 idx_of_input_to_unique_out
  OP_REQUIRES_OK(context,
                 context->allocate_output("idx_of_input_to_unique",
                                          TensorShape({uniq_size}),
                                          &idx_of_input_to_unique_out));
  TIndex* idx_of_input_to_unique_ptr =
      idx_of_input_to_unique_out->flat<TIndex>().data();

  OP_REQUIRES_OK(
      context,
      GetIdxOfInputToUnique(device, uniq_size, count_ptr, segment_ends_ptr,
                            sorted_unique_perm_ptr, count_prefix_sum_ptr,
                            sorted_input_inds_ptr, idx_of_input_to_unique_ptr));

  // output 1 unique_idxs_out
  OP_REQUIRES_OK(
      context, context->allocate_output(
                   "unique_idxs", TensorShape({input_size}), &unique_idxs_out));
  TIndex* idx_ptr = unique_idxs_out->flat<TIndex>().data();

  // Compute indices output.
  OP_REQUIRES_OK(context, LookupAndScatterUniqueIds(
                              device, input_size, sorted_input_inds_ptr,
                              sorted_input_unique_ids_ptr,
                              inv_sorted_unique_perm_ptr, idx_ptr));
}

template void UniqueWithCountsGPU<int64, int64>(
    OpKernelContext* context, const Tensor* input, Tensor* unique_keys,
    Tensor* unique_idxs_out, Tensor* unique_counts_out,
    Tensor* idx_of_input_to_unique_out, Tensor* unique_offsets_out);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA