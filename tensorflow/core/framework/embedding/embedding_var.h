/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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


#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/framework/embedding/embedding_filter.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/multilevel_embedding.h"
#include "tensorflow/core/framework/typed_allocator.h"

namespace tensorflow {

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name,
               embedding::StorageManager<K, V>* storage_manager,
               EmbeddingConfig emb_cfg = EmbeddingConfig(),
               Allocator* alloc = nullptr):
      name_(name),
      storage_manager_(storage_manager),
      default_value_(nullptr),
      value_len_(0),
      alloc_(alloc),
      emb_config_(emb_cfg) {
        if (IsMultiLevel() || emb_config_.record_freq) {
          add_freq_fn_ = [](ValuePtr<V>* value_ptr, int freq, int64 filter_freq) {
            value_ptr->AddFreq(freq);
          };
        } else if (emb_config_.is_counter_filter()) {
          add_freq_fn_ = [](ValuePtr<V>* value_ptr, int freq, int64 filter_freq) {
            if (value_ptr->GetFreq() < filter_freq)
              value_ptr->AddFreq(freq);
          };
        } else {
          add_freq_fn_ = [](ValuePtr<V>* value_ptr, int freq, int64 filter_freq) {};
        }
        if (emb_config_.steps_to_live != 0 || emb_config_.record_version) {
          update_version_fn_ = [](ValuePtr<V>* value_ptr, int64 gs) {
            value_ptr->SetStep(gs);
          };
        } else {
          update_version_fn_ = [](ValuePtr<V>* value_ptr, int64 gs) {};
        }
      }

  Status Init(const Tensor& default_tensor, int64 default_value_dim) {
    filter_ = FilterFactory::CreateFilter<K, V, EmbeddingVar<K, V>>(emb_config_, this, storage_manager_);

    if (storage_manager_ == nullptr) {
      return errors::InvalidArgument("Invalid ht_type to construct EmbeddingVar");
    } else {
      if (embedding::StorageType::HBM_DRAM == storage_manager_->GetStorageType()) {
#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
        emb_config_.default_value_dim = default_value_dim;
        value_len_ = default_tensor.NumElements() / emb_config_.default_value_dim;
        default_value_ = TypedAllocator::Allocate<V>(alloc_, default_tensor.NumElements(), AllocationAttributes());
        auto default_tensor_flat = default_tensor.flat<V>();
        buffer1 = nullptr;
        buffer2 = nullptr;
        buffer3 = nullptr;
        buffer1_size = 0;
        buffer2_size = 0;
        buffer3_size = 0;
        cudaMemcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes(), cudaMemcpyHostToDevice);
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
      } else {
        alloc_ = cpu_allocator();
        emb_config_.default_value_dim = default_value_dim;
        value_len_ = default_tensor.NumElements()/emb_config_.default_value_dim;
        default_value_ = TypedAllocator::Allocate<V>(alloc_, default_tensor.NumElements(), AllocationAttributes());
        auto default_tensor_flat = default_tensor.flat<V>();
        memcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes());
      }
      if (LayoutType::NORMAL_CONTIGUOUS == storage_manager_->GetLayoutType()) {
        storage_manager_->SetAllocLen(value_len_, emb_config_.slot_num + 1);
      }

      return Status::OK();
    }
  }

  void SetInitialized() {
    is_initialized_ = true;
  }

  bool IsInitialized() const {
    return is_initialized_;
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, bool* is_filter) {
    return filter_->LookupOrCreateKey(key, value_ptr, is_filter);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr) {
    Status s = storage_manager_->GetOrCreate(key, value_ptr, emb_config_.total_num(storage_manager_->GetAllocLen()));
    TF_CHECK_OK(s);
    return s;
  }

  void UpdateVersion(ValuePtr<V>* value_ptr, int64 gs) {
    update_version_fn_(value_ptr, gs);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, int64 update_version, bool &need_copyback) {
    Status s = storage_manager_->GetOrCreate(key, value_ptr, emb_config_.total_num(storage_manager_->GetAllocLen()), need_copyback);
    TF_CHECK_OK(s);
    if (emb_config_.is_primary() && emb_config_.steps_to_live != 0 && update_version != -1) {
      (*value_ptr)->SetStep(update_version);
    }
    return s;
  }

  void BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    TF_CHECK_OK(storage_manager_->BatchCommit(keys, value_ptrs));
  }

  int64 GetVersion(K key) {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(LookupOrCreateKey(key, &value_ptr));
    return value_ptr->GetStep();
  }

  int64 GetFreq(K key) {
    return filter_->GetFreq(key);
  }

  void LookupOrCreate(K key, V* val, V* default_v, int count = 1)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    ValuePtr<V>* value_ptr = nullptr;
    filter_->LookupOrCreate(key, val, default_value_ptr, &value_ptr, count);
    add_freq_fn_(value_ptr, count, emb_config_.filter_freq);
  }

  void LookupWithFreqBatch(K* keys, bool *init_flags, bool *copyback_flags, V** memcpy_address, int start, int limit) {
    ValuePtr<V>* value_ptr = nullptr;
    for (int i = start; i < limit; i++) {
      TF_CHECK_OK(LookupOrCreateKey(keys[i], &value_ptr, -1, copyback_flags[i]));
      if (!copyback_flags[i]) {
        memcpy_address[i] = LookupOrCreateEmb(value_ptr, init_flags[i]);
      } else {
        //memcpy_address[i] = LookupOrCreateEmb(value_ptr, init_flags[i]);
        memcpy_address[i] = value_ptr->GetValue(0,0);
      }
      value_ptr->AddFreq();
    }
  }

  void BatchInitEmb(int64 size, V** memcpy_address, V* default_value, bool* init_flags, int64 value_len) {
    filter_->BatchInitEmb(size, memcpy_address, default_value, init_flags, value_len);
  }

#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
  void CreateGPUBatch(V* val_base, V** default_values, int64 size, int64 slice_elems, bool* init_flags, V** memcpy_address) {
    for (int i = 0;i < size;i++) {
      default_values[i] = (default_values[i] == nullptr) ? default_value_ : default_values[i];
    }
    filter_->CreateGPUBatch(val_base, default_values, size, slice_elems, value_len_, init_flags, memcpy_address);
  }

  void CopyBackToGPU(K* keys, int64 size, bool* copyback_flags, V** memcpy_address) {
    size_t value_len = emb_config_.total_num(storage_manager_->GetAllocLen());
    V* memcpy_buffer_gpu;
    V** dev_value_address, **value_address;
    int total = 0;
    for (int i = 0; i < size;i++) {
      if (copyback_flags[i]) {
        total++;
      }
    }
    int *copyback_cursor = new int[total]();
    ValuePtr<V>** gpu_value_ptrs = new ValuePtr<V>* [total];
    cudaMalloc(&memcpy_buffer_gpu, total * value_len * sizeof(V));

    storage_manager_->CopyBackToGPU(total, keys, size, copyback_flags, memcpy_address, value_len, copyback_cursor, gpu_value_ptrs, memcpy_buffer_gpu);

    value_address = (V**)malloc(sizeof(V*) * total);
    cudaMalloc(&dev_value_address, sizeof(V*) * total);

    for (int i = 0;i < total;i++) {
      bool init;
      memcpy_address[copyback_cursor[i]] = LookupOrCreateEmb(gpu_value_ptrs[i], init);
      value_address[i] = memcpy_address[copyback_cursor[i]];
    }

    cudaMemcpy(dev_value_address, value_address, sizeof(V*) * total, cudaMemcpyHostToDevice);
    int block_dim = 128;
    void* args[] = { (void*)&dev_value_address, (void*)&memcpy_buffer_gpu, (void*)&value_len, (void*)&total};

    cudaLaunchKernel((void *)BatchUnpack<V>, (total + block_dim - 1) / block_dim * value_len, block_dim, args, 0, NULL);
    cudaDeviceSynchronize();

    cudaFree(dev_value_address);
    cudaFree(memcpy_buffer_gpu);
    delete []copyback_cursor;
    delete []gpu_value_ptrs;
  }
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, const V* default_v) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, default_v,
        emb_config_.emb_index, storage_manager_->GetOffset(emb_config_.emb_index));
  }

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, bool &need_initialize) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, nullptr,
        emb_config_.emb_index, storage_manager_->GetOffset(emb_config_.emb_index), need_initialize);
  }

  V* LookupPrimaryEmb(ValuePtr<V>* value_ptr) {
    V* primary_val = value_ptr->GetValue(emb_config_.primary_emb_index,
                                 storage_manager_->GetOffset(emb_config_.primary_emb_index));
    return primary_val;
  }

  typename TTypes<V>::Flat flat(ValuePtr<V>* value_ptr) {
    V* val = LookupOrCreateEmb(value_ptr, default_value_);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  void Commit(const K id, ValuePtr<V>* value_ptr) {
    TF_CHECK_OK(storage_manager_->Commit(id, value_ptr));
  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return storage_manager_->Size();
  }

  int64 CacheSize() const {
    return storage_manager_->CacheSize();
  }

  int64 MinFreq() {
    return emb_config_.filter_freq;
  }

  int64 StepsToLive() const {
    return emb_config_.steps_to_live;
  }

  float GetL2WeightThreshold() {
    return emb_config_.l2_weight_threshold;
  }

  bool IsMultiLevel() {
    return storage_manager_->IsMultiLevel();
  }

  bool IsRecordFreq() {
    return emb_config_.record_freq;
  }

  bool IsRecordVersion() {
    return emb_config_.record_version;
  }

  bool IsHBMDRAM() {
    return embedding::StorageType::HBM_DRAM == storage_manager_->GetStorageType();
  }

  std::string DebugString() const {
    return emb_config_.DebugString();
  }

  EmbeddingFilter<K, V, EmbeddingVar<K, V>>* GetFilter() const {
    return filter_;
  }

  Status Import(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num,
                bool is_filter) {
    return filter_->Import(restore_buff, key_num, bucket_num, partition_id, partition_num, is_filter);
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list,
                    std::vector<int64>* version_list, std::vector<int64>* freq_list,
                    embedding::Iterator** it) {
    return storage_manager_->GetSnapshot(key_list, value_list, version_list,
                                         freq_list, emb_config_, filter_, it);
  }

  Status Destroy() {
    return storage_manager_->Destroy();
  }

  mutex* mu() {
    return &mu_;
  }

  embedding::StorageManager<K, V>* storage_manager() {
    return storage_manager_;
  }

  Status Shrink() {
    return storage_manager_->Shrink(emb_config_, value_len_);
  }

  Status Shrink(int64 gs) {
    if (emb_config_.steps_to_live > 0) {
      return storage_manager_->Shrink(gs, emb_config_.steps_to_live);
    }
  }

  V* GetDefaultValuePtr() {
    return default_value_;
  }

  int64 GetDefaultValueDim() {
    return emb_config_.default_value_dim;
  }

  V* GetDefaultValue(int64 key) {
    return default_value_ + (key % emb_config_.default_value_dim) * value_len_;
  }

  void SetSlotNum(int64 slot_num) {
    emb_config_.slot_num = slot_num;
  }

  int64 GetSlotNum() {
    return emb_config_.slot_num;
  }

  embedding::BatchCache<K>* Cache() {
    return storage_manager_->Cache();
  }

  int64 GetEmbeddingIndex() {
    return emb_config_.emb_index;
  }

  Allocator* GetAllocator() {
    return alloc_;
  }

  V** GetBuffer1(int64 size) {
    if (buffer1_size >= size) {
      return buffer1;
    } else{
      if (buffer1_size != 0) {
        alloc_->DeallocateRaw(buffer1);
      }
      buffer1 = (V**)alloc_->AllocateRaw(0, size * sizeof(V*));
      buffer1_size = size;
      return buffer1;
    }
  }

  V** GetBuffer2(int64 size) {
    if (buffer2_size >= size) {
      return buffer2;
    } else {
      if (buffer2_size != 0) {
        alloc_->DeallocateRaw(buffer2);
      }
      buffer2 =(V**)alloc_->AllocateRaw(0, size * sizeof(V*));
      buffer2_size = size;
      return buffer2;
    }
  }

  V** GetBuffer3(int64 size) {
    if (buffer3_size >= size) {
      return buffer3;
    } else {
      if (buffer3_size != 0) {
        alloc_->DeallocateRaw(buffer3);
      }
      buffer3 = (V**)alloc_->AllocateRaw(0, size * sizeof(V*));
      buffer3_size = size;
      return buffer3;
    }
  }

 private:
  std::string name_;
  bool is_initialized_ = false;

  mutex mu_;

  V* default_value_;
  V **buffer1, **buffer2, **buffer3;
  int64 buffer1_size, buffer2_size, buffer3_size;
  int64 value_len_;
  Allocator* alloc_;
  embedding::StorageManager<K, V>* storage_manager_;
  EmbeddingConfig emb_config_;
  EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter_;
  std::function<void(ValuePtr<V>*, int, int64)> add_freq_fn_;
  std::function<void(ValuePtr<V>*, int64)> update_version_fn_;

  ~EmbeddingVar() override {
    // When dynamic dimension embedding is used, there will be more than one primary slot
    if (emb_config_.is_primary() && emb_config_.primary_emb_index == 0) {
      Destroy();
      delete storage_manager_;
    }
    if (embedding::StorageType::HBM_DRAM == storage_manager_->GetStorageType()) {
      buffer1_size = 0;
      buffer2_size = 0;
      buffer3_size = 0;
      if (buffer1 != nullptr) {
        alloc_->DeallocateRaw(buffer1);
        buffer1 = nullptr;
      }
      if (buffer2 != nullptr) {
        alloc_->DeallocateRaw(buffer2);
        buffer2 = nullptr;
      }
      if (buffer3 != nullptr) {
        alloc_->DeallocateRaw(buffer3);
        buffer3 = nullptr;
      }
    }
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_


