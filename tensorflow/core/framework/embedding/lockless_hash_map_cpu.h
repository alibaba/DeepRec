/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LOCKLESS_HASH_MAP_CPU_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LOCKLESS_HASH_MAP_CPU_H_

#include "sparsehash/dense_hash_map_lockless"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"


namespace tensorflow {

namespace embedding {

template <class K, class V>
class LocklessHashMapCPU : public KVInterface<K, V> {
 public:
  LocklessHashMapCPU() {
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(EMPTY_KEY_, nullptr);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(DELETED_KEY_);
  }

  ~LocklessHashMapCPU() {
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY_) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    } else {
      *value_ptr = iter.second;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    auto iter = hash_map_.insert_lockless(
        std::move(std::pair<K, ValuePtr<V>*>(key, const_cast<ValuePtr<V>*>(value_ptr))));
    // insert fail, exist key
    if ((*(iter.first)).second != value_ptr){
      return errors::AlreadyExists(
          "already exists Key: ", key, " in LocklessHashMap.");
    } else {
      return Status::OK();
    }
  }

  // Other Method
  int64 Size() const {
    return hash_map_.size_lockless();
  }

  // Remove KV
  Status Remove(K key) {
    if (hash_map_.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    }
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    ValuePtr<V>* cpu_value_ptr = new NormalContiguousValuePtr<V>(cpu_allocator(), total_dims_);
    cudaMemcpy((char *)cpu_value_ptr->GetPtr() + sizeof(FixedLengthHeader),
               *(char **)((char*)value_ptr->GetPtr() + sizeof(FixedLengthHeader)),
               total_dims_ * sizeof(V),
               cudaMemcpyDeviceToHost);
    memcpy((char *)cpu_value_ptr->GetPtr(), (char *)value_ptr->GetPtr(), sizeof(FixedLengthHeader));
    Insert(key, cpu_value_ptr);
    return Status::OK();
  }

  Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    int batch_size = keys.size();
    V** value_address = (V**)malloc(sizeof(V * ) * batch_size);
    V** dev_value_address;
    V* batch_data_place;
    V* dev_batch_data_place;
    Allocator *allocator = ev_allocator();
    dev_value_address = (V**)allocator->AllocateRaw(0, batch_size * sizeof(V *));
    dev_batch_data_place = (V*)allocator->AllocateRaw(0, sizeof(V) * batch_size * total_dims_);
    batch_data_place = (V *)malloc(sizeof(V) * batch_size * total_dims_);

    // Copy GPU addresses V*
    for(int i = 0;i < batch_size;++i) {
      value_address[i] = *(V **)((char *)value_ptrs[i]->GetPtr() + sizeof(FixedLengthHeader));
    }
    cudaMemcpy(dev_value_address, value_address, sizeof(V *) * batch_size, cudaMemcpyHostToDevice);

    // Launch Kernel,Copy data to continuous place
    int block_dim = 128;
    void* args[] = { (void*)&dev_value_address, (void*)&dev_batch_data_place, (void*)&total_dims_, (void*)&batch_size};

    cudaDeviceSynchronize();

    cudaMemcpy(batch_data_place, dev_batch_data_place, sizeof(V) * batch_size * total_dims_, cudaMemcpyDeviceToHost);

    // Copy data to ValuePtrs in memory;Insert it into hashmap
    for(int i = 0;i < batch_size;++i) {
      ValuePtr<V>* cpu_value_ptr = new NormalContiguousValuePtr<V>(cpu_allocator(), total_dims_);
      memcpy((char *)cpu_value_ptr->GetPtr() + sizeof(FixedLengthHeader), &batch_data_place[i * total_dims_], total_dims_ * sizeof(V));
      memcpy((char *)cpu_value_ptr->GetPtr(), (char *)value_ptrs[i]->GetPtr(), sizeof(FixedLengthHeader));
      Insert(keys[i], cpu_value_ptr);
    }

    allocator->DeallocateRaw(dev_value_address);
    allocator->DeallocateRaw(dev_batch_data_place);

    delete []value_address;
    return Status::OK();
  }

  /*
  Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    int batch_size = keys.size();
    for (int i = 0; i < batch_size; i++) {
      Commit(keys[i],value_ptrs[i]);
    }
    return Status::OK();
  }
  */

  Status GetSnapshot(std::vector<K>* key_list, std::vector<ValuePtr<V>* >* value_ptr_list) {
    std::pair<const K, ValuePtr<V>*> *hash_map_dump;
    int64 bucket_count;
    std::pair<std::pair<const K, ValuePtr<V>*>*, long unsigned int> it = hash_map_.GetSnapshot();
    hash_map_dump = it.first;
    bucket_count = it.second;
    for (int64 j = 0; j < bucket_count; j++) {
      if (hash_map_dump[j].first != EMPTY_KEY_ && hash_map_dump[j].first != DELETED_KEY_) {
        key_list->push_back(hash_map_dump[j].first);
        value_ptr_list->push_back(hash_map_dump[j].second);
      }
    }
    free(hash_map_dump);
    return Status::OK();
  }

  std::string DebugString() const {
    LOG(INFO) << "map info size:" << Size();
    LOG(INFO) << "map info bucket_count:" << hash_map_.bucket_count();
    LOG(INFO) << "map info load_factor:" << hash_map_.load_factor();
    LOG(INFO) << "map info max_load_factor:" << hash_map_.max_load_factor();
    LOG(INFO) << "map info min_load_factor:" << hash_map_.min_load_factor();
    return "";
  }

 private:
  typedef google::dense_hash_map_lockless<K, ValuePtr<V>* > LockLessHashMap;
  static const int EMPTY_KEY_ = -1;
  static const int DELETED_KEY_ = -2;
  LockLessHashMap hash_map_;
  int total_dims_;
};
}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LOCKLESS_HASH_MAP_CPU_H_
