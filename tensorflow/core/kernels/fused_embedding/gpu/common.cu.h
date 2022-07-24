#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_COMMON_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_COMMON_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

#define CK_CUDA_THROW_(x)                                                      \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + " \n");        \
    }                                                                          \
  } while (0)

namespace tensorflow {

namespace fused_embedding {

template <typename T>
inline T* data_p_with_type(Tensor& t) {
  return reinterpret_cast<T*>(t.data());
}

template <typename T>
inline T* data_p_with_type(const Tensor& t) {
  return reinterpret_cast<T*>(t.data());
}

template <typename T>
inline T* data_p_with_type(Tensor* t) {
  return reinterpret_cast<T*>(t->data());
}

template <typename T>
inline T* data_p_with_type(const Tensor* t) {
  return reinterpret_cast<T*>(t->data());
}

struct IndicePair {
  int64_t row_in_batch;
  int64_t entry_in_column;
};

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_COMMON_CU_H_