/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if GOOGLE_CUDA
#if TF_ENABLE_GPU_EV

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <unordered_map>
#include "third_party/cuco_hash_table/cuco/dynamic_map.cuh"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/kv_variable_ops_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace cg = cooperative_groups;

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class gpu_hash_map_tf_allocator {
public:
  Allocator* alloc_;
  using value_type = T;

  gpu_hash_map_tf_allocator(Allocator* alloc) : alloc_(alloc) {}

  gpu_hash_map_tf_allocator(const gpu_hash_map_tf_allocator& a) noexcept : alloc_(a.alloc_) {}

  template <typename U>
  gpu_hash_map_tf_allocator(const gpu_hash_map_tf_allocator<U>& a) noexcept : alloc_(a.alloc_) {}

  gpu_hash_map_tf_allocator& operator=(const gpu_hash_map_tf_allocator& a) noexcept {
    return *this;
  }

  gpu_hash_map_tf_allocator& operator=(gpu_hash_map_tf_allocator&& a) {
    alloc_ = a.alloc_;
    return *this;
  }

  ~gpu_hash_map_tf_allocator() noexcept {}

  value_type* allocate(size_t size) const {
    void* ptr = alloc_->AllocateRaw(Allocator::kAllocatorAlignment, size * sizeof(value_type), AllocationAttributes());
    return (value_type*)ptr;
  }

  void deallocate(value_type* ptr, size_t) const {
    alloc_->DeallocateRaw(ptr);
  }
};

template <typename T, typename U>
bool operator==(gpu_hash_map_tf_allocator<T> const&, gpu_hash_map_tf_allocator<U> const&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(gpu_hash_map_tf_allocator<T> const& lhs, gpu_hash_map_tf_allocator<U> const& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename KeyType, typename ValueType, typename CUCOAllocator = gpu_hash_map_tf_allocator<uint8_t>>
class DynamicHashTable {
public:
  cuco::dynamic_map<KeyType, ValueType, cuda::thread_scope_device, CUCOAllocator> map_;

  DynamicHashTable(size_t initial_capacity, KeyType empty_key_sentinel, ValueType empty_value_sentinel, CUCOAllocator alloc)
      : map_(initial_capacity, empty_key_sentinel, empty_value_sentinel, alloc) {
  }
  ~DynamicHashTable() {}
};

template <typename K, typename V>
GPUHashTable<K, V>::GPUHashTable(K empty_key_sentinel, Allocator* alloc, size_t initial_capacity) : initial_bank_size(initial_capacity) {
  hash_table = new DynamicHashTable<K, int32>(initial_capacity, empty_key_sentinel, -1, gpu_hash_map_tf_allocator<uint8_t>(alloc));
  cudaMallocManaged(&start_idx, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>));
  *start_idx = 0;
}

template <typename K, typename V>
GPUHashTable<K, V>::~GPUHashTable() {
  delete hash_table;
  cudaFree(start_idx);
}

template <typename K, typename V>
int32 GPUHashTable<K, V>::Size() {
  return hash_table->map_.get_size();
}

template class GPUHashTable<int32, float>;
template class GPUHashTable<int32, double>;
template class GPUHashTable<int64, float>;
template class GPUHashTable<int64, double>;

namespace functor {
using atomicT = cuda::atomic<std::size_t, cuda::thread_scope_device>;

template <uint32_t block_size,
          uint32_t tile_size,
          typename Key,
          typename mutableViewT,
          typename ViewT,
          typename Hash     = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_lookup_and_insert_key_kernel(const Key* key_first,
                                                int32* value_first,
                                                int32 num_items,
                                                mutableViewT* submap_mutable_views,
                                                ViewT* submap_views,
                                                uint32_t num_submaps,
                                                atomicT* num_successes,
                                                atomicT* start_idx,
                                                int32 submap_idx,
                                                Hash hash = Hash{},
                                                KeyEqual key_equal = KeyEqual{}) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();
  int32 tmp;

  while(key_idx < num_items) {
    auto key = *(key_first + key_idx);
    int32 found_value = empty_value_sentinel;

    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(tile, key, hash, key_equal);
      if (found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }
    if (found_value == empty_value_sentinel) {
      if (tile.thread_rank() == 0) {
        tmp = start_idx->fetch_add(1);
      }
      found_value = tile.shfl(tmp, 0);
      auto insert_pair = cuco::pair_type<Key, int32>{key, found_value};
      if (submap_mutable_views[submap_idx].insert(tile, insert_pair, hash, key_equal) &&
          tile.thread_rank() == 0) {
        thread_num_successes++;
      }
    }

    if (tile.thread_rank() == 0) {
      *(value_first + key_idx) = found_value;
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }

  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) { *num_successes += block_num_successes; }
}

template <typename Key, typename V>
struct KvLookupInsertKey<GPUDevice, Key, V> {
  void operator()(const Key* key_first,
                  int32* value_first,
                  int32 num_items,
                  GPUHashTable<Key, V>* hash_table,
                  atomicT* start_idx,
                  cudaStream_t stream) {
    using mutableViewT = typename cuco::dynamic_map<Key, int32, cuda::thread_scope_device, gpu_hash_map_tf_allocator<uint8_t>>::mutable_view_type;
    using ViewT = typename cuco::dynamic_map<Key, int32, cuda::thread_scope_device, gpu_hash_map_tf_allocator<uint8_t>>::view_type;
    auto& map = hash_table->hash_table->map_;
    map.reserve(map.get_size() + num_items);
    uint32_t submap_idx = 0;
    std::size_t num_to_insert = num_items;

    while (num_to_insert > 0) {
      std::size_t capacity_remaining =
          map.get_max_load_factor() * map.get_submaps()[submap_idx]->get_capacity() - map.get_submaps()[submap_idx]->get_size();
      if (capacity_remaining >= map.get_min_insert_size()) {
        *(map.get_num_successes()) = 0;
        int device_id;
        CUCO_CUDA_TRY(cudaGetDevice(&device_id));
        CUCO_CUDA_TRY(cudaMemPrefetchAsync(map.get_num_successes(), sizeof(atomicT), device_id));

        auto n = std::min(capacity_remaining, num_to_insert);
        auto const block_size = 128;
        auto const stride = 1;
        auto const tile_size = 4;
        auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);
        TF_CHECK_OK(GpuLaunchKernel(kv_lookup_and_insert_key_kernel<block_size, tile_size, Key, mutableViewT, ViewT, cuco::detail::MurmurHash3_32<Key>, thrust::equal_to<Key>>,
                                    grid_size, block_size, 0, stream,
                                    key_first, value_first, n,
                                    map.get_submap_mutable_views().data().get(), map.get_submap_views().data().get(), map.get_submaps().size(),
                                    map.get_num_successes(), start_idx, submap_idx,
                                    cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
        CUCO_CUDA_TRY(cudaDeviceSynchronize());
        std::size_t h_num_successes = map.get_num_successes()->load(cuda::std::memory_order_relaxed);
        map.update_submap_sizes(submap_idx, h_num_successes);
        key_first += n;
        value_first += n;
        num_to_insert -= n;
      }
      submap_idx++;
    }
  }
};

template <typename Key, typename Value>
__global__ void kv_lookup_or_create_emb_kernel(const Key* key_first,
                                               Value* val,
                                               Value* default_v,
                                               int64 dim,
                                               bool is_use_default_value_tensor,
                                               int32* item_idxs,
                                               int32 slot_idx,
                                               Value** d_banks,
                                               bool** d_flags,
                                               int32 slot_num,
                                               int32 default_v_num,
                                               int32 bank_size) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto slot_offset = bank_idx * slot_num + slot_idx;
  bool stored = d_flags[slot_offset][offset_in_bank];
  __syncthreads();
  if (stored == false) {
    d_flags[slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      int32 default_v_idx;
      if (is_use_default_value_tensor) {
        default_v_idx = item_idx % default_v_num;
      } else {
        default_v_idx = *(key_first + item_idx) % default_v_num;
      }
      d_banks[slot_offset][offset_in_bank * dim + id] = default_v[default_v_idx * dim + id];
    }
  }
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    val[item_idx * dim + id] = d_banks[slot_offset][offset_in_bank * dim + id];
  }
}

template <typename Key, typename Value>
struct KvLookupCreateEmb<GPUDevice, Key, Value> {
  void operator()(const Key* key_first,
                  Value* val,
                  Value* default_v,
                  int64 dim,
                  int32* item_idxs,
                  int32 num_items,
                  int32 slot_idx,
                  int32 default_v_num,
                  bool is_use_default_value_tensor,
                  Value** d_banks,
                  bool** d_flags,
                  int32 slot_num,
                  int32 bank_size,
                  cudaStream_t stream) {
  auto const block_size = 256;
  auto const grid_size = num_items;
  TF_CHECK_OK(GpuLaunchKernel(kv_lookup_or_create_emb_kernel<Key, Value>,
                              grid_size, block_size, 0, stream,
                              key_first, val, default_v, dim, is_use_default_value_tensor,
                              item_idxs, slot_idx,
                              d_banks, d_flags,
                              slot_num, default_v_num, bank_size));
}
};

template <typename Key, typename Value>
__global__ void kv_update_emb_kernel(const Key* key_first,
                                               Value* default_v,
                                               int64 dim,
                                               int32* item_idxs,
                                               int32 slot_idx,
                                               Value** d_banks,
                                               bool** d_flags,
                                               int32 slot_num,
                                               int32 default_v_num,
                                               int32 bank_size) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto slot_offset = bank_idx * slot_num + slot_idx;
  bool stored = d_flags[slot_offset][offset_in_bank];
  __syncthreads();
  if (stored == false) {
    d_flags[slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      int32 default_v_idx;
      default_v_idx = item_idx % default_v_num;
      d_banks[slot_offset][offset_in_bank * dim + id] = default_v[default_v_idx * dim + id];
    }
  }
}

template <typename Key, typename Value>
struct KvUpdateEmb<GPUDevice, Key, Value> {
  void operator()(const Key* key_first,
                  Value* default_v,
                  int64 dim,
                  int32* item_idxs,
                  int32 num_items,
                  int32 slot_idx,
                  int32 default_v_num,
                  Value** d_banks,
                  bool** d_flags,
                  int32 slot_num,
                  int32 bank_size,
                  cudaStream_t stream) {
  auto const block_size = 256;
  auto const grid_size = num_items;
  TF_CHECK_OK(GpuLaunchKernel(kv_update_emb_kernel<Key, Value>,
                              grid_size, block_size, 0, stream,
                              key_first, default_v, dim,
                              item_idxs, slot_idx,
                              d_banks, d_flags,
                              slot_num, default_v_num, bank_size));
}
};

template <typename Key,
          typename ViewT,
          typename Hash     = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_get_key_snapshot_kernel(Key* key,
                                           int32* item_idxs,
                                           int32 slot_idx,
                                           int32 primary_slot_idx,
                                           bool** d_flags,
                                           int32 bank_num,
                                           int32 slot_num,
                                           int32 bank_size,
                                           ViewT* submap_views,
                                           uint32_t num_submaps,
                                           int32 ev_size,
                                           Hash hash = Hash{},
                                           KeyEqual key_equal = KeyEqual{}) {
  int n = 0;
  for(auto i = 0; i < num_submaps; ++i) {
    auto submap_view_size = submap_views[i].get_capacity();
    for(auto j = 0; j < submap_view_size; ++j) {
      auto found = submap_views[i].get_slot(j, hash, key_equal);
      if (found != submap_views[i].end()) {
        int32 item_pos = found->second;
        auto bank_idx = item_pos / bank_size;
        auto offset_in_bank = item_pos % bank_size;
        auto slot_offset = bank_idx * slot_num + slot_idx;
        auto pri_slot_offset = bank_idx * slot_num + primary_slot_idx;
        if (d_flags[slot_offset][offset_in_bank] && d_flags[pri_slot_offset][offset_in_bank]) {
          *(key + n) = found->first;
          *(item_idxs + n) = found->second;
          ++n;
        }
      }
    }
  }
  for (auto i = n; i < ev_size; ++i) {
    *(key + n) = submap_views[0].get_empty_key_sentinel();
  }
}

template <typename Key, typename V>
struct KvKeyGetSnapshot<GPUDevice, Key, V> {
  void operator()(Key* key_first,
                  int32* value_first,
                  int32 slot_idx,
                  int32 primary_slot_idx,
                  bool** d_flags,
                  int32 bank_num,
                  int32 slot_num,
                  int32 bank_size,
                  GPUHashTable<Key, V>* hash_table,
                  int32 ev_size,
                  cudaStream_t stream) {
    using ViewT = typename cuco::dynamic_map<Key, int32, cuda::thread_scope_device, gpu_hash_map_tf_allocator<uint8_t>>::view_type;
    auto& map = hash_table->hash_table->map_;

    auto const block_size = 1;
    auto const grid_size = 1;
    TF_CHECK_OK(GpuLaunchKernel(kv_get_key_snapshot_kernel<Key, ViewT, cuco::detail::MurmurHash3_32<Key>, thrust::equal_to<Key>>,
                                grid_size, block_size, 0, stream,
                                key_first, value_first, slot_idx, primary_slot_idx,
                                d_flags, bank_num, slot_num, bank_size,
                                map.get_submap_views().data().get(), map.get_submaps().size(),
                                ev_size, cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
  }
};

template <typename Key, typename Value>
__global__ void kv_emb_get_snapshot_kernel(Key* key,
                                           Value* val,
                                           Key empty_key_sentinel,
                                           int64 dim,
                                           int32* item_idxs,
                                           int32 slot_idx,
                                           Value** d_banks,
                                           int32 bank_num,
                                           int32 slot_num,
                                           int32 bank_size) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto slot_offset = bank_idx * slot_num + slot_idx;
  if (key[item_idx] != empty_key_sentinel) {
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      val[item_idx * dim + id] = d_banks[slot_offset][offset_in_bank * dim + id];
    }
  }
}

template <typename Key, typename Value>
struct KvEmbGetSnapshot<GPUDevice, Key, Value> {
  void operator()(Key* key,
                  Value* val,
                  Key empty_key_sentinel,
                  int64 dim,
                  int32* item_idxs,
                  int32 num_items,
                  int32 slot_idx,
                  Value** d_banks,
                  int32 bank_num,
                  int32 slot_num,
                  int32 bank_size,
                  cudaStream_t stream) {
  auto const block_size = 256;
  auto const grid_size = num_items;
  TF_CHECK_OK(GpuLaunchKernel(kv_emb_get_snapshot_kernel<Key, Value>,
                              grid_size, block_size, 0, stream,
                              key, val, empty_key_sentinel, dim,
                              item_idxs, slot_idx, d_banks,
                              bank_num, slot_num, bank_size));
}
};

} // namespace functor

template struct functor::KvLookupInsertKey<GPUDevice, int32, float>;
template struct functor::KvLookupInsertKey<GPUDevice, int32, double>;
template struct functor::KvLookupInsertKey<GPUDevice, int64, float>;
template struct functor::KvLookupInsertKey<GPUDevice, int64, double>;

template struct functor::KvLookupCreateEmb<GPUDevice, int32, float>;
template struct functor::KvLookupCreateEmb<GPUDevice, int32, double>;
template struct functor::KvLookupCreateEmb<GPUDevice, int64, float>;
template struct functor::KvLookupCreateEmb<GPUDevice, int64, double>;

template struct functor::KvKeyGetSnapshot<GPUDevice, int32, float>;
template struct functor::KvKeyGetSnapshot<GPUDevice, int32, double>;
template struct functor::KvKeyGetSnapshot<GPUDevice, int64, float>;
template struct functor::KvKeyGetSnapshot<GPUDevice, int64, double>;

template struct functor::KvEmbGetSnapshot<GPUDevice, int32, float>;
template struct functor::KvEmbGetSnapshot<GPUDevice, int32, double>;
template struct functor::KvEmbGetSnapshot<GPUDevice, int64, float>;
template struct functor::KvEmbGetSnapshot<GPUDevice, int64, double>;

template struct functor::KvUpdateEmb<GPUDevice, int32, float>;
template struct functor::KvUpdateEmb<GPUDevice, int32, double>;
template struct functor::KvUpdateEmb<GPUDevice, int64, float>;
template struct functor::KvUpdateEmb<GPUDevice, int64, double>;

}  // namespace tensorflow

#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
