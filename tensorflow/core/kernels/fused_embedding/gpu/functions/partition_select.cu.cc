#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/partition_select.cu.h"

#include <cub/cub.cuh>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace fused_embedding {

// A macro helper to declare SelectScanKernel, because they just have only
// a little bit differences. Need to define macros before using this
#define DeclareSelectScanKernel                                              \
  template <typename T, int WarpWorkload>                                    \
  __global__ void SelectScanKernelName(                                      \
      const T* keys, SelectScanKernelArgs const int64 num_partitions,        \
      const int64 length, const int64 predicates_length,                     \
      const int64 counters_length, unsigned int* predicates,                 \
      unsigned int* accumulate_counters) {                                   \
    __shared__ unsigned int warp_cnt_sum[1];                                 \
    int lnid = threadIdx.x % 32;                                             \
    if (lnid == 0) warp_cnt_sum[0] = 0; /* init shared mem of sum */         \
    const int warp_iteration = WarpWorkload / 32;                            \
    const int select_scan_warps = length % WarpWorkload == 0                 \
                                      ? (length / WarpWorkload)              \
                                      : (length / WarpWorkload + 1);         \
    const int partition_id = blockIdx.x / select_scan_warps;                 \
    int warp_id_in_partition = blockIdx.x % select_scan_warps;               \
    unsigned int mask;                                                       \
    unsigned int cnt;                                                        \
    SelectScanKernelLoadCodeBlock;                                           \
    _Pragma("unroll") for (int i = 0; i < warp_iteration; i++) {             \
      int selected;                                                          \
      int load_offset = warp_id_in_partition * WarpWorkload + i * 32 + lnid; \
      if (load_offset < length) {                                            \
        T key = keys[load_offset];                                           \
        SelectScanKernelEvalCodeBlock;                                       \
      } else {                                                               \
        selected = 0;                                                        \
      }                                                                      \
      mask = __ballot_sync(0xffffffff, selected);                            \
      if (lnid == 0)                                                         \
        predicates[partition_id * predicates_length +                        \
                   warp_id_in_partition * (WarpWorkload / 32) + i] = mask;   \
      cnt = __popc(mask);                                                    \
      if (lnid == 0) atomicAdd(warp_cnt_sum, cnt);                           \
    }                                                                        \
    /* use different threads in warp to accumulate to different location of  \
     * accumulate_counters, this will do a prefix sum on global mem */       \
    const int counters_slots_need_accumulate =                               \
        select_scan_warps - warp_id_in_partition;                            \
    const unsigned int warp_cnt = warp_cnt_sum[0];                           \
    for (int i = 0; i < counters_slots_need_accumulate; i += 32) {           \
      if (i + lnid < counters_slots_need_accumulate) {                       \
        atomicAdd(accumulate_counters + partition_id * counters_length +     \
                      warp_id_in_partition + lnid,                           \
                  warp_cnt);                                                 \
      }                                                                      \
    }                                                                        \
  }

// A macro helper to declare SelectKernel, because they just have only
// a little bit differences. Need to define macros before using this
#define DeclareSelectKernel                                                    \
  template <typename T, typename TIndex, int WarpWorkload>                     \
  __global__ void SelectKernelName(                                            \
      const T* keys, SelectKernelArgs unsigned int* predicates,                \
      unsigned int* accumulate_counters, const int64 num_partitions,           \
      const int64 length, const int64 predicates_length,                       \
      const int64 counters_length, void** output_ptrs, TIndex* permutation) {  \
    const int lnid = threadIdx.x % 32;                                         \
    const int warp_iteration = WarpWorkload / 32;                              \
    const int select_scan_warps = length % WarpWorkload == 0                   \
                                      ? (length / WarpWorkload)                \
                                      : (length / WarpWorkload + 1);           \
    const int partition_id = blockIdx.x / select_scan_warps;                   \
    int warp_id_in_partition = blockIdx.x % select_scan_warps;                 \
    unsigned int predmask = 0;                                                 \
    unsigned int cnt = 0;                                                      \
    SelectKernelLoadCodeBlock;                                                 \
    T* keys_output_ptr = (T*)(output_ptrs[partition_id]);                      \
    if (lnid < warp_iteration) {                                               \
      predmask =                                                               \
          predicates[partition_id * predicates_length +                        \
                     (warp_id_in_partition * (WarpWorkload / 32)) + lnid];     \
      cnt = __popc(predmask);                                                  \
    }                                                                          \
    _Pragma("unroll") for (int offset = 1; offset < warp_iteration;            \
                           offset <<= 1) {                                     \
      /* prefix sum */                                                         \
      int n = __shfl_up_sync(0xffffffff, cnt, offset);                         \
      if (lnid >= offset) cnt += n;                                            \
    }                                                                          \
    unsigned int global_index = 0;                                             \
    if (warp_id_in_partition > 0)                                              \
      global_index = accumulate_counters[partition_id * counters_length +      \
                                         warp_id_in_partition - 1];            \
    _Pragma("unroll") for (int i = 0; i < warp_iteration; i++) {               \
      unsigned int mask = __shfl_sync(0xffffffff, predmask, i);                \
      unsigned int inner_warp_index = 0;                                       \
      if (i > 0) inner_warp_index = __shfl_sync(0xffffffff, cnt, i - 1);       \
      if (mask & (1 << lnid)) {                                                \
        int load_offset = warp_id_in_partition * WarpWorkload + i * 32 + lnid; \
        T key = keys[load_offset]; /* Will not cause out of boundry access,    \
                                     because mask will be 0 for this thread*/  \
        SelectKernelRecalcKeyCodeBlock;                                        \
        int output_offset =                                                    \
            global_index + inner_warp_index +                                  \
            __popc(mask & (((unsigned int)(1) << lnid) - (unsigned int)(1)));  \
        keys_output_ptr[output_offset] = new_key;                              \
        permutation[2 * load_offset] = partition_id;                           \
        permutation[2 * load_offset + 1] = (TIndex)output_offset;              \
      }                                                                        \
    }                                                                          \
  }

// A macro helper to declare DefinePartitionSelect. Need to define
// macros before using this
#define DeclareSelect                                                          \
  template <typename T, typename TIndex, int WarpWorkload>                     \
  void SelectName(OpKernelContext* ctx, const Tensor* keys,                    \
                  SelectArgs const int64 num_partitions,                       \
                  cudaEvent_t memcpy_event, OpOutputList& selected_keys,       \
                  Tensor* permutation) {                                       \
    OP_REQUIRES(ctx, keys->dims() == 1,                                        \
                errors::InvalidArgument("Tensor keys must ranks 1"));          \
    OP_REQUIRES(                                                               \
        ctx,                                                                   \
        WarpWorkload >= 32 && WarpWorkload <= 1024 &&                          \
            (WarpWorkload && !(WarpWorkload & (WarpWorkload - 1))),            \
        errors::InvalidArgument(                                               \
            "WarpWorkload must be larger than warp size and less than 1024 "   \
            "32 and is exponential of 2, 32, 64, 128, i.e."));                 \
    const GPUDevice& device = ctx->eigen_gpu_device();                         \
    const int64 length = keys->NumElements();                                  \
    const int64 warp_iteration = WarpWorkload / 32;                            \
    Tensor predicates;                                                         \
    Tensor accumulate_counters;                                                \
    const int64 select_scan_warps = length % WarpWorkload == 0                 \
                                        ? (length / WarpWorkload)              \
                                        : (length / WarpWorkload + 1);         \
    const int64 counters_length = select_scan_warps;                           \
    const int64 predicates_length = select_scan_warps * warp_iteration;        \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(DT_UINT32,                                     \
                                TensorShape{counters_length * num_partitions}, \
                                &accumulate_counters));                        \
    CK_CUDA_THROW_(cudaMemsetAsync(                                            \
        data_p_with_type<void>(accumulate_counters), 0,                        \
        accumulate_counters.NumElements() * sizeof(unsigned int),              \
        device.stream()));                                                     \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(                                               \
                 DT_UINT32, TensorShape{predicates_length * num_partitions},   \
                 &predicates));                                                \
                                                                               \
    {                                                                          \
      const int64 threads = 32;                                                \
      const int64 blocks = select_scan_warps * num_partitions;                 \
      OP_REQUIRES_OK(                                                          \
          ctx,                                                                 \
          GpuLaunchKernel(                                                     \
              SelectScanKernelName<T, WarpWorkload>, blocks, threads, 0,       \
              device.stream(), data_p_with_type<T>(keys),                      \
              SelectScanPassArgs num_partitions, length, predicates_length,    \
              counters_length, data_p_with_type<unsigned int>(predicates),     \
              data_p_with_type<unsigned int>(accumulate_counters)));           \
    }                                                                          \
    std::vector<unsigned int> selected_nums_host;                              \
    selected_nums_host.resize(num_partitions);                                 \
    /* copy the last element(which is the sum of previous) with stride */      \
    CK_CUDA_THROW_(cudaMemcpy2DAsync(                                          \
        selected_nums_host.data(), 1 * sizeof(unsigned int),                   \
        data_p_with_type<unsigned int>(accumulate_counters) +                  \
            counters_length - 1,                                               \
        counters_length * sizeof(unsigned int), 1 * sizeof(unsigned int),      \
        num_partitions, cudaMemcpyDeviceToHost, device.stream()));             \
    CK_CUDA_THROW_(cudaEventRecord(memcpy_event, device.stream()));            \
    CK_CUDA_THROW_(cudaEventSynchronize(memcpy_event));                        \
                                                                               \
    std::vector<void*> output_ptrs_host;                                       \
    output_ptrs_host.resize(num_partitions);                                   \
    for (int i = 0; i < num_partitions; i++) {                                 \
      Tensor* tmp_out;                                                         \
      OP_REQUIRES_OK(                                                          \
          ctx, selected_keys.allocate(                                         \
                   i, TensorShape({int64(selected_nums_host[i])}), &tmp_out)); \
      output_ptrs_host[i] = data_p_with_type<void>(tmp_out);                   \
    }                                                                          \
    Tensor output_ptrs;                                                        \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(DT_UINT64, TensorShape{int64(num_partitions)}, \
                                &output_ptrs));                                \
    CK_CUDA_THROW_(cudaMemcpyAsync(data_p_with_type<void>(output_ptrs),        \
                                   output_ptrs_host.data(),                    \
                                   num_partitions * sizeof(size_t),            \
                                   cudaMemcpyHostToDevice, device.stream()));  \
    {                                                                          \
      const int64 threads = 32;                                                \
      const int64 blocks = select_scan_warps * num_partitions;                 \
      OP_REQUIRES_OK(                                                          \
          ctx, GpuLaunchKernel(                                                \
                   SelectKernelName<T, TIndex, WarpWorkload>, blocks, threads, \
                   0, device.stream(), data_p_with_type<T>(keys),              \
                   SelectPassArgs data_p_with_type<unsigned int>(predicates),  \
                   data_p_with_type<unsigned int>(accumulate_counters),        \
                   num_partitions, length, predicates_length, counters_length, \
                   data_p_with_type<void*>(output_ptrs),                       \
                   data_p_with_type<TIndex>(permutation)));                    \
    }                                                                          \
  }

// =============== Div Selection =============== //
#define SelectName PartitionSelectDiv
#define SelectArgs const Tensor &accu_div,
#define SelectScanPassArgs data_p_with_type<int64>(accu_div),
#define SelectPassArgs SelectScanPassArgs

#define SelectScanKernelName SelectScanDivKernel
#define SelectScanKernelArgs const int64 *accu_div,
#define SelectScanKernelLoadCodeBlock                                    \
  int64 lower_bound = partition_id > 0 ? accu_div[partition_id - 1] : 0; \
  int64 upper_bound = accu_div[partition_id];
#define SelectScanKernelEvalCodeBlock \
  selected = int(key >= lower_bound && key < upper_bound);

#define SelectKernelName SelectDivKernel
#define SelectKernelArgs SelectScanKernelArgs

#define SelectKernelLoadCodeBlock \
  int64 lower_bound = partition_id > 0 ? accu_div[partition_id - 1] : 0;
#define SelectKernelRecalcKeyCodeBlock T new_key = key - lower_bound;

DeclareSelectScanKernel;
DeclareSelectKernel;
DeclareSelect;

template void PartitionSelectDiv<int64, int, 64>(
    OpKernelContext* ctx, const Tensor* keys, const Tensor& accu_div,
    const int64 num_partitions, cudaEvent_t memcpy_event,
    OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectDiv<int64, int, 128>(
    OpKernelContext* ctx, const Tensor* keys, const Tensor& accu_div,
    const int64 num_partitions, cudaEvent_t memcpy_event,
    OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectDiv<int64, int, 256>(
    OpKernelContext* ctx, const Tensor* keys, const Tensor& accu_div,
    const int64 num_partitions, cudaEvent_t memcpy_event,
    OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectDiv<int64, int, 512>(
    OpKernelContext* ctx, const Tensor* keys, const Tensor& accu_div,
    const int64 num_partitions, cudaEvent_t memcpy_event,
    OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectDiv<int64, int, 1024>(
    OpKernelContext* ctx, const Tensor* keys, const Tensor& accu_div,
    const int64 num_partitions, cudaEvent_t memcpy_event,
    OpOutputList& selected_keys, Tensor* permutation);

// =============== Mod Selection =============== //

#define SelectName PartitionSelectMod
#define SelectArgs
#define SelectScanPassArgs
#define SelectPassArgs SelectScanPassArgs

#define SelectScanKernelName SelectScanModKernel
#define SelectScanKernelArgs
#define SelectScanKernelLoadCodeBlock
#define SelectScanKernelEvalCodeBlock \
  selected = int(key % num_partitions == partition_id);

#define SelectKernelName SelectModKernel
#define SelectKernelArgs SelectScanKernelArgs

#define SelectKernelLoadCodeBlock
#define SelectKernelRecalcKeyCodeBlock T new_key = key / num_partitions;

DeclareSelectScanKernel;
DeclareSelectKernel;
DeclareSelect;

template void PartitionSelectMod<int64, int, 64>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectMod<int64, int, 128>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectMod<int64, int, 256>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectMod<int64, int, 512>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectMod<int64, int, 1024>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

// =============== Mod EV Selection =============== //

#define SelectName PartitionSelectModEV
#define SelectArgs
#define SelectScanPassArgs
#define SelectPassArgs SelectScanPassArgs

#define SelectScanKernelName SelectScanModEVKernel
#define SelectScanKernelArgs
#define SelectScanKernelLoadCodeBlock
#define SelectScanKernelEvalCodeBlock \
  selected = int(key % 1000 % num_partitions == partition_id);

#define SelectKernelName SelectModEVKernel
#define SelectKernelArgs SelectScanKernelArgs

#define SelectKernelLoadCodeBlock
#define SelectKernelRecalcKeyCodeBlock T new_key = key;

DeclareSelectScanKernel;
DeclareSelectKernel;
DeclareSelect;

template void PartitionSelectModEV<int64, int, 64>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectModEV<int64, int, 128>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectModEV<int64, int, 256>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectModEV<int64, int, 512>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

template void PartitionSelectModEV<int64, int, 1024>(
    OpKernelContext* ctx, const Tensor* keys, const int64 num_partitions,
    cudaEvent_t memcpy_event, OpOutputList& selected_keys, Tensor* permutation);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA