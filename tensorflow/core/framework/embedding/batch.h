#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BATCH_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BATCH_

#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV

namespace tensorflow {

template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len,
                          int limit, V** default_value, bool* init_flags);

template<class V>
__global__ void BatchUnpack(V** dev_value_address, V* memcpy_buffer_gpu,
                            int value_len, int limit);

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, V* g, float lr,
                                      int embedding_dim, long long int limit,
                                      bool* init_flags, V* default_value);

template<class V>
__global__ void CopyEmbedding(V** batch, V* batch_data_space, int total_dims, int limit);

}  // namespace tensorflow

#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BATCH_
