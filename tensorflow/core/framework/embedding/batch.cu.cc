#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV

#include "tensorflow/core/framework/embedding/batch.h"

namespace tensorflow {
template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len,
                          int limit, V** default_value, bool* init_flags) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    if (init_flags[item_id]) {
      *(batch[item_id] + item_pos) = *(default_value[item_id] + item_pos);
    }
    val_base[i] = *(batch[item_id] + item_pos);
  }
}

template __global__ void BatchCopy<int>(int**, int*, int, int, int**, bool*);
template __global__ void BatchCopy<float>(float**, float*, int, int, float**, bool*);
template __global__ void BatchCopy<double>(double**, double*, int, int, double**, bool*);
template __global__ void BatchCopy<long long>(long long**, long long*, int, int, long long**, bool*);

template<class V>
__global__ void BatchUnpack(V** dev_value_address, V* memcpy_buffer_gpu,
                            int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    *(dev_value_address[item_id] + item_pos) = memcpy_buffer_gpu[i];
  }
}

template __global__ void BatchUnpack<int>(int**, int*, int, int);
template __global__ void BatchUnpack<float>(float**, float*, int, int);
template __global__ void BatchUnpack<double>(double**, double*, int, int);
template __global__ void BatchUnpack<long long>(long long**, long long*, int, int);

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, V* g, float lr,
                                      int embedding_dim, long long int limit,
                                      bool* init_flags, V* default_value) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    if (init_flags[item_id]) {
      *(a[item_id] + item_pos) = default_value[item_pos];
    }
    *(a[item_id] + item_pos) += g[i] * g[i];
    *(v[item_id] + item_pos) -= lr * g[i] * rsqrt(*(a[item_id] + item_pos));
  }
}

template __global__ void SparseApplyAdagradGPU<float>(float**, float**, float*, float, int, long long int, bool*, float*);
template __global__ void SparseApplyAdagradGPU<double>(double**, double**, double*, float, int, long long int, bool*, double*);

template<class V>
__global__ void CopyEmbedding(V** batch, V* batch_data_space, int total_dims, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < limit  * total_dims) {
    batch_data_space[i] = *(batch[i / total_dims] + i % total_dims);
  }
}

template __global__ void CopyEmbedding<int>(int**, int*, int, int);
template __global__ void CopyEmbedding<float>(float**, float*, int, int);
template __global__ void CopyEmbedding<double>(double**, double*, int, int);
template __global__ void CopyEmbedding<long long>(long long**, long long*, int, int);

}  // namespace tensorflow
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
