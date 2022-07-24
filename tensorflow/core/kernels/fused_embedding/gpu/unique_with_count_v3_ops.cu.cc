#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/constant_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"
#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/hash_functions.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

// Overload CUDA atomic for other 64bit unsinged/signed integer type
__forceinline__ __device__ long atomicAdd(long* address, long val) {
  return (long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicAdd(long long* address,
                                               long long val) {
  return (long long)atomicAdd((unsigned long long*)address,
                              (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long* address,
                                                   unsigned long val) {
  return (unsigned long)atomicAdd((unsigned long long*)address,
                                  (unsigned long long)val);
}

__forceinline__ __device__ long atomicCAS(long* address, long compare,
                                          long val) {
  return (long)atomicCAS((unsigned long long*)address,
                         (unsigned long long)compare, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicCAS(long long* address,
                                               long long compare,
                                               long long val) {
  return (long long)atomicCAS((unsigned long long*)address,
                              (unsigned long long)compare,
                              (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicCAS(unsigned long* address,
                                                   unsigned long compare,
                                                   unsigned long val) {
  return (unsigned long)atomicCAS((unsigned long long*)address,
                                  (unsigned long long)compare,
                                  (unsigned long long)val);
}

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace gpu_unique_with_counts {

const static int block_size = 64;

template <typename KeyType, typename CounterType>
__global__ void InitKernel(KeyType* keys, CounterType* vals,
                           CounterType* counts, CounterType* counter,
                           const size_t capacity, const KeyType empty_key,
                           const CounterType empty_val,
                           const CounterType empty_counts,
                           const CounterType init_counter_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    // Simply store every element a unused <K, V> pair
    keys[idx] = empty_key;
    vals[idx] = empty_val;
    counts[idx] = empty_counts;
  }
  if (idx == 0) {
    counter[idx] = init_counter_val;
  }
}

template <typename KeyType, typename CounterType>
void Init(const GPUDevice& d, KeyType* keys, CounterType* vals,
          CounterType* counts, CounterType* counter, const size_t capacity,
          const KeyType empty_key, const CounterType empty_val,
          const CounterType empty_counts, const CounterType init_counter_val) {
  const int threads = block_size;
  const int blocks = (capacity - 1) / block_size + 1;
  TF_CHECK_OK(GpuLaunchKernel(InitKernel<KeyType, CounterType>, blocks, threads,
                              0, d.stream(), keys, vals, counts, counter,
                              capacity, empty_key, empty_val, empty_counts,
                              init_counter_val));
}

template <typename KeyType>
__global__ void GetSizeKernel(const KeyType* keys, const size_t capacity,
                              size_t* d_size, const KeyType empty_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  /* Whether the slot mapping to the current thread is empty? do nothing :
   * Atomically add to counter */
  if (idx < capacity) {
    if (keys[idx] != empty_key) {
      atomicAdd(&block_acc, 1);
    }
  }
  __syncthreads();

  /* Atomically reduce block counter to global conuter */
  if (threadIdx.x == 0) {
    atomicAdd(d_size, block_acc);
  }
}

template <typename KeyType>
void GetSize(const GPUDevice& d, const KeyType* keys, const size_t capacity,
             size_t* d_size, const KeyType empty_key) {
  const int threads = block_size;
  const int blocks = (capacity - 1) / block_size + 1;
  TF_CHECK_OK(GpuLaunchKernel(GetSizeKernel<KeyType>, blocks, threads, 0,
                              d.stream(), keys, capacity, d_size, empty_key));
}

template <typename KeyType, typename CounterType,
          typename hasher = MurmurHash3_32<KeyType>>
__global__ void GetInsertKernel(const KeyType* d_key, CounterType* d_val,
                                const size_t len, KeyType* keys,
                                CounterType* vals, CounterType* counts,
                                const size_t capacity,
                                CounterType* d_global_counter,
                                KeyType empty_key,
                                const CounterType empty_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    KeyType target_key = d_key[idx];
    size_t hash_index = hasher::hash(target_key) % capacity;
    size_t counter = 0;
    while (true) {
      // Have searched all the slot in the hashtable, but all slots in the
      // hashtable are occupied by other keys
      if (counter >= capacity) {
        assert(false && "error: unique op fails: hashtable is full");
      }
      // Try to set the key for the current slot to target key
      const KeyType old_key =
          atomicCAS(keys + hash_index, empty_key, target_key);
      volatile CounterType& target_val_pos = vals[hash_index];
      if (empty_key == old_key) {
        CounterType result_val;
        result_val = atomicAdd(d_global_counter, 1);
        d_val[idx] = result_val;
        target_val_pos = result_val;
        atomicAdd(counts + hash_index, 1);
        break;
      } else if (target_key == old_key) {
        while (target_val_pos == empty_val)
          ;
        d_val[idx] = target_val_pos;
        atomicAdd(counts + hash_index, 1);
        break;
      }
      counter++;
      hash_index = (hash_index + 1) % capacity;
    }
  }
}

template <typename KeyType, typename CounterType,
          typename hasher = MurmurHash3_32<KeyType>>
void GetInsert(const GPUDevice& d, const KeyType* d_key, CounterType* d_val,
               const size_t input_size, KeyType* keys, CounterType* vals,
               CounterType* counts, const size_t capacity,
               CounterType* d_global_counter, KeyType empty_key,
               const CounterType empty_val) {
  const int threads = block_size;
  const int blocks = (input_size - 1) / block_size + 1;
  TF_CHECK_OK(GpuLaunchKernel(GetInsertKernel<KeyType, CounterType, hasher>,
                              blocks, threads, 0, d.stream(), d_key, d_val,
                              input_size, keys, vals, counts, capacity,
                              d_global_counter, empty_key, empty_val));
}

template <typename KeyType, typename CounterType>
__global__ void DumpKernel(KeyType* d_key, CounterType* d_counts,
                           const KeyType* keys, const CounterType* vals,
                           const CounterType* counts, const size_t offset,
                           const size_t capacity, size_t* d_dump_counter,
                           const KeyType empty_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  KeyType read_key;
  CounterType read_val;
  CounterType read_count;
  bool valid_slot = false;
  // Each thread gather the key and value from slot assigned to them.
  if (idx < capacity) {
    read_key = keys[offset + idx];
    if (read_key != empty_key) {
      valid_slot = true;
      atomicAdd(&block_acc, 1);
      read_val = vals[offset + idx];
      read_count = counts[offset + idx];
    }
  }
  __syncthreads();

  // Each block accumulate the dump count to global counter
  if (threadIdx.x == 0) {
    atomicAdd(d_dump_counter, block_acc);
  }

  // Each thread store one slot's data back to global memory, d_dump_counter
  // is how many slots in total dumped.
  if (valid_slot) {
    d_key[read_val] = read_key;
    d_counts[read_val] = read_count;
  }
}

template <typename KeyType, typename CounterType>
void Dump(const GPUDevice& d, KeyType* d_key, CounterType* d_counts,
          const KeyType* keys, const CounterType* vals,
          const CounterType* counts, const size_t offset, const size_t capacity,
          size_t* d_dump_counter, const KeyType empty_key) {
  const int threads = block_size;
  const int blocks = (capacity - 1) / block_size + 1;
  TF_CHECK_OK(GpuLaunchKernel(DumpKernel<KeyType, CounterType>, blocks, threads,
                              0, d.stream(), d_key, d_counts, keys, vals,
                              counts, offset, capacity, d_dump_counter,
                              empty_key));
}

}  // namespace gpu_unique_with_counts

template <typename KeyType, typename CounterType>
class UniqueWithCountsV3 : public OpKernel {
 public:
  explicit UniqueWithCountsV3(OpKernelConstruction* ctx) : OpKernel(ctx) {
    cudaEventCreateWithFlags(&memcpy_event_, cudaEventDisableTiming);
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace gpu_unique_with_counts;
    using fused_embedding::data_p_with_type;

    auto device = ctx->eigen_device<GPUDevice>();

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    Tensor const* input = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input));

    const size_t input_size = input->NumElements();
    const KeyType empty_key = std::numeric_limits<KeyType>::max();
    const CounterType empty_val = std::numeric_limits<CounterType>::max();
    const CounterType empty_counts = 0;
    const CounterType init_counter_val = 0;
    const float load_factor = 1.3;
    const size_t capacity = input_size * load_factor;

    Tensor keys_storage;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<KeyType>::value,
                                TensorShape({int64(capacity)}), &keys_storage));

    Tensor vals_storage;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<CounterType>::value,
                                TensorShape({int64(capacity)}), &vals_storage));

    Tensor counts_storage;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<CounterType>::value,
                                           TensorShape({int64(capacity)}),
                                           &counts_storage));

    Tensor counter;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<CounterType>::value,
                                           TensorShape({int64(1)}), &counter));

    Tensor dump_counter;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({int64(1)}),
                                           &dump_counter));

    Tensor* unique_idxs;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("unique_idxs",
                                             TensorShape({int64(input_size)}),
                                             &unique_idxs));

    Init(device, data_p_with_type<KeyType>(keys_storage),
         data_p_with_type<CounterType>(vals_storage),
         data_p_with_type<CounterType>(counts_storage),
         data_p_with_type<CounterType>(counter), capacity, empty_key, empty_val,
         empty_counts, init_counter_val);

    GetInsert(device, data_p_with_type<const KeyType>(input),
              data_p_with_type<CounterType>(unique_idxs), input_size,
              data_p_with_type<KeyType>(keys_storage),
              data_p_with_type<CounterType>(vals_storage),
              data_p_with_type<CounterType>(counts_storage), capacity,
              data_p_with_type<CounterType>(counter), empty_key, empty_val);

    CounterType uniq_size;
    CK_CUDA_THROW_(cudaMemcpyAsync(
        &uniq_size, data_p_with_type<CounterType>(counter), sizeof(CounterType),
        cudaMemcpyDeviceToHost, device.stream()));
    CK_CUDA_THROW_(cudaEventRecord(memcpy_event_, device.stream()));
    CK_CUDA_THROW_(cudaEventSynchronize(memcpy_event_));

    Tensor* unique_keys;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "unique_keys", TensorShape({int64(uniq_size)}), &unique_keys));

    Tensor* unique_counts;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("unique_counts",
                                             TensorShape({int64(uniq_size)}),
                                             &unique_counts));

    Dump(device, data_p_with_type<KeyType>(unique_keys),
         data_p_with_type<CounterType>(unique_counts),
         data_p_with_type<const KeyType>(keys_storage),
         data_p_with_type<const CounterType>(vals_storage),
         data_p_with_type<const CounterType>(counts_storage), 0, capacity,
         data_p_with_type<size_t>(dump_counter), empty_key);
  }

 private:
  cudaEvent_t memcpy_event_;
};

REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("KeyType")
                            .TypeConstraint<int32>("CounterType"),
                        UniqueWithCountsV3<int, int>);

REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("KeyType")
                            .TypeConstraint<int64>("CounterType"),
                        UniqueWithCountsV3<int, long long>);

REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("KeyType")
                            .TypeConstraint<int32>("CounterType"),
                        UniqueWithCountsV3<long long, int>);

REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("KeyType")
                            .TypeConstraint<int64>("CounterType"),
                        UniqueWithCountsV3<long long, long long>);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA