#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_PARTITION_SELECT_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_PARTITION_SELECT_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace fused_embedding {
template <typename T, typename TIndex, int WarpWorkload>
void PartitionSelectDiv(OpKernelContext* ctx, const Tensor* keys,
                        const Tensor& accu_div, const int64 num_partitions,
                        cudaEvent_t memcpy_event, OpOutputList& selected_keys,
                        Tensor* permutation);

template <typename T, typename TIndex, int WarpWorkload>
void PartitionSelectMod(OpKernelContext* ctx, const Tensor* keys,
                        const int64 num_partitions, cudaEvent_t memcpy_event,
                        OpOutputList& selected_keys, Tensor* permutation);

template <typename T, typename TIndex, int WarpWorkload>
void PartitionSelectModEV(OpKernelContext* ctx, const Tensor* keys,
                          const int64 num_partitions, cudaEvent_t memcpy_event,
                          OpOutputList& selected_keys, Tensor* permutation);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_PARTITION_SELECT_CU_H_