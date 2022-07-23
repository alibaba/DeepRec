#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTILEVEL_EMBEDDING_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTILEVEL_EMBEDDING_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/dense_hash_map.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/ssd_hashkv.h"
#include "tensorflow/core/framework/embedding/lockless_hash_map.h"
#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
#include "tensorflow/core/framework/embedding/lockless_hash_map_cpu.h"
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

namespace embedding {

struct StorageConfig {
  StorageConfig() : type(StorageType::INVALID), path(""), layout_type(LayoutType::NORMAL) {
    size = {1<<30,1<<30,1<<30,1<<30};
  }
  StorageConfig(StorageType t,
                const std::string& p,
                const std::vector<int64>& s,
                const std::string& layout) : type(t), path(p) {
    if ("normal" == layout) {
      layout_type = LayoutType::NORMAL;
    } else if ("light" == layout) {
      layout_type = LayoutType::LIGHT;
    } else if ("normal_contiguous" == layout){
      layout_type = LayoutType::NORMAL_CONTIGUOUS;
    } else {
      LOG(WARNING) << "Unknown layout: " << layout << ", use LayoutType::NORMAL by default.";
      layout_type = LayoutType::NORMAL;
    }
    size = s;
  }
  StorageType type;
  LayoutType layout_type;
  std::string path;
  std::vector<int64> size;
};

template <class K, class V>
class StorageManager {
 public:
  StorageManager(const string& name,
                 StorageConfig sc,
                 size_t cap = -1)
: hash_table_count_(0),
  name_(name),
  sc_(sc),
  cache_(nullptr),
  cache_capacity_(cap),
  eviction_thread_(nullptr),
  total_dims_(0),
  alloc_len_(0),
  is_multi_level_(false) {}

  ~StorageManager() {
    for (auto kv: kvs_) {
      delete kv.first;
    }
    delete cache_;
  }

  Status Init(Allocator* alloc_ = nullptr) {
    switch (sc_.layout_type) {
      case LayoutType::NORMAL:
        new_value_ptr_fn_ = [] (Allocator* alloc, size_t size) { return new NormalValuePtr<V>(alloc, size); };
        break;
      case LayoutType::LIGHT:
        new_value_ptr_fn_ = [] (Allocator* alloc, size_t size) { return new LightValuePtr<V>(alloc, size); };
        break;
      case LayoutType::NORMAL_CONTIGUOUS:
        new_value_ptr_fn_ = [] (Allocator* alloc, size_t size) { return new NormalContiguousValuePtr<V>(alloc, size); };
        break;
      default:
        new_value_ptr_fn_ = [] (Allocator* alloc, size_t size) { return new NormalValuePtr<V>(alloc, size); };
        break;
    }
    switch (sc_.type) {
      Allocator* alloc_ssd;
      case StorageType::DRAM:
        VLOG(1) << "StorageManager::DRAM: " << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(), cpu_allocator()));
        break;
      case StorageType::PMEM_MEMKIND:
        VLOG(1) << "StorageManager::PMEM_MEMKIND: " << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(), pmem_allocator()));
        break;
      case StorageType::PMEM_LIBPMEM:
        VLOG(1) << "StorageManager::PMEM_LIBPMEM: " << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(),
                                      experimental_pmem_allocator(sc_.path, sc_.size[0])));
        break;
      case StorageType::LEVELDB:
        VLOG(1) << "StorageManager::LEVELDB: " << name_;
        kvs_.push_back(std::make_pair(new LevelDBKV<K, V>(sc_.path), ev_allocator()));
        break;
      case StorageType::DRAM_PMEM:
        VLOG(1) << "StorageManager::DRAM_PMEM: " << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(), ev_allocator()));
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(),
                                      experimental_pmem_allocator(sc_.path, sc_.size[1])));
        break;
      case StorageType::DRAM_LEVELDB:
        VLOG(1) << "StorageManager::DRAM_LEVELDB: " << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(), cpu_allocator()));
        kvs_.push_back(std::make_pair(new LevelDBKV<K, V>(sc_.path), cpu_allocator()));
        break;
      case StorageType::SSDHASH:
        VLOG(1) << "StorageManager::SSDHASH: " << name_;
        alloc_ssd = cpu_allocator();
        kvs_.emplace_back(std::make_pair(new SSDHashKV<K, V>(sc_.path, alloc_ssd), alloc_ssd));
        break;
      case StorageType::DRAM_SSDHASH:
        VLOG(1) << "StorageManager::DRAM_SSDHASH: " << name_;
        alloc_ssd = cpu_allocator();
        kvs_.emplace_back(std::make_pair(new LocklessHashMap<K, V>(), alloc_ssd));
        kvs_.emplace_back(std::make_pair(new SSDHashKV<K, V>(sc_.path, alloc_ssd), alloc_ssd));
        break;
      case StorageType::HBM_DRAM:
#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
        new_value_ptr_fn_ = [] (Allocator* allocator, size_t size) { return new NormalGPUValuePtr<V>(allocator, size); };
        LOG(INFO) << "StorageManager::HBM_DRAM: " << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(), alloc_));
        kvs_.push_back(std::make_pair(new LocklessHashMapCPU<K, V>(), cpu_allocator()));
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
        break;
      default:
        VLOG(1) << "StorageManager::default" << name_;
        kvs_.push_back(std::make_pair(new LocklessHashMap<K, V>(), cpu_allocator()));
        break;
    }

    if (sc_.type == embedding::PMEM_MEMKIND || sc_.type == embedding::PMEM_LIBPMEM ||
        sc_.type == embedding::DRAM_PMEM || sc_.type == embedding::DRAM_SSDHASH ||
        sc_.type == embedding::HBM_DRAM || sc_.type == embedding::DRAM_LEVELDB) {
      is_multi_level_ = true;
    }

    hash_table_count_ = kvs_.size();
    if (hash_table_count_ > 1) {
      cache_ = new LRUCache<K>();
      eviction_thread_ = Env::Default()->StartThread(ThreadOptions(), "EV_Eviction",
                                                     [this]() { BatchEviction(); });
      thread_pool_.reset(new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                               "MultiLevel_Embedding_Cache", 2,
                                               /*low_latency_hint=*/false));
    }
    // DebugString();
    CHECK(2 >= hash_table_count_) << "Not support multi-level(>2) embedding.";

    return Status::OK();
  }

  void SetAllocLen(int64 value_len, int slot_num){
    while (flag_.test_and_set(std::memory_order_acquire));
    //The start address of every slot should be aligned to 16 bytes, otherwise a coredump will happen in the ApplyOp.
    alloc_len_ = (value_len * sizeof(V) % 16 == 0) ? value_len : value_len + (16 - (sizeof(V) * value_len) % 16) / sizeof(V);
    int64 temp = alloc_len_ * slot_num;
    if (temp > total_dims_) {
      total_dims_ = temp;
      if (sc_.type == StorageType::LEVELDB || sc_.type == StorageType::SSDHASH) {
        kvs_[0].first->SetTotalDims(total_dims_);
      } else if (sc_.type == StorageType::DRAM_LEVELDB || sc_.type == StorageType::DRAM_SSDHASH) {
        kvs_[1].first->SetTotalDims(total_dims_);
      } else if (sc_.type == StorageType::HBM_DRAM) {
        kvs_[1].first->SetTotalDims(total_dims_);
      }
      if (hash_table_count_ > 1) {
        cache_capacity_ = sc_.size[0] / (total_dims_ * sizeof(V));
        done_ = true;
        LOG(INFO) << "Cache cache_capacity: " << cache_capacity_;
      }
    }
    flag_.clear(std::memory_order_release);
  }

  int64 GetAllocLen(){
    return alloc_len_;
  }

  int64 GetOffset(int64 index) {
    return alloc_len_ * index;
  }

  int64 GetTotalDims() {
    return total_dims_;
  }

  LayoutType GetLayoutType() {
    return sc_.layout_type;
  }

  embedding::StorageType GetStorageType() {
    return sc_.type;
  }

  std::string GetStoragePath() {
    return sc_.path;
  }

  int64 GertStorageSize() {
    return sc_.size;
  }

  bool IsMultiLevel() {
    return is_multi_level_;
  }

  std::string DebugString() const{
    return strings::StrCat("Level Number: ", hash_table_count_,
                          " alloc_len: ", alloc_len_,
                          " total_dims: ", total_dims_,
                          " Storage Type: ", sc_.type,
                          " Storage Path: ", sc_.path,
                          " Storage Capacity: ", sc_.size);
  }



  void Schedule(std::function<void()> fn) {
    if (hash_table_count_ > 1) {
      thread_pool_->Schedule(std::move(fn));
    }
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr, size_t size) {
    bool found = false;
    int level = 0;
    for (; level < hash_table_count_; ++level) {
      Status s = kvs_[level].first->Lookup(key, value_ptr);
      if (s.ok()) {
        found = true;
        break;
      }
    }
    if (!found) {
      *value_ptr = new_value_ptr_fn_(kvs_[0].second, size);
    }

    if (sc_.type == StorageType::HBM_DRAM && level && found) {
#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
      ValuePtr<V>* gpu_value_ptr = new_value_ptr_fn_(kvs_[0].second, size);
      V* cpu_data_address = (*value_ptr)->GetValue(0, 0);
      V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
      cudaMemcpy(gpu_data_address, cpu_data_address, size * sizeof(V), cudaMemcpyHostToDevice);
      *value_ptr = gpu_value_ptr;
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
    }
    if (level || !found) {
      Status s = kvs_[0].first->Insert(key, *value_ptr);
      if (s.ok()) {
        // Insert Success
        return s;
      } else {
        // Insert Failed, key already exist
        (*value_ptr)->Destroy(kvs_[0].second);
        delete *value_ptr;
        s = kvs_[0].first->Lookup(key, value_ptr);
        return s;
      }
    }
    return Status::OK();
  }

Status GetOrCreate(K key, ValuePtr<V>** value_ptr, size_t size, bool &need_copyback) {
    bool found = false;
    int level = 0;
    need_copyback = false;

    for (; level < hash_table_count_; ++level) {
      Status s = kvs_[level].first->Lookup(key, value_ptr);
      if (s.ok()) {
        found = true;
        break;
      }
    }
    if (!found) {
      *value_ptr = new_value_ptr_fn_(kvs_[0].second, size);
    }

    if (sc_.type == StorageType::HBM_DRAM && level && found) {
      need_copyback = true;
    }
    if ( (level || !found ) && !need_copyback) {
      Status s = kvs_[0].first->Insert(key, *value_ptr);
      if (s.ok()) {
        // Insert Success
        return s;
      } else {
        // Insert Failed, key already exist
        (*value_ptr)->Destroy(kvs_[0].second);
        delete *value_ptr;
        s = kvs_[0].first->Lookup(key, value_ptr);
        return s;
      }
    }
    return Status::OK();
  }

#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
  void CopyBackToGPU(int total, K* keys, int64 size, bool* copyback_flags, V** memcpy_address, size_t value_len, int *copyback_cursor, ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu){
    V *memcpy_buffer_cpu;
    memcpy_buffer_cpu = (V*)malloc(total * value_len * sizeof(V));
    int j = 0;
    for (int i = 0; i < size;i++) {
      if (copyback_flags[i]) {
        ValuePtr<V>* gpu_value_ptr = new_value_ptr_fn_(kvs_[0].second, size);
        //Copy Header Info
        memcpy((char *)gpu_value_ptr->GetPtr(), (char *)memcpy_address[i] - sizeof(FixedLengthHeader), sizeof(FixedLengthHeader));
        V* cpu_data_address = memcpy_address[i];
        V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
        //cudaMemcpy(gpu_data_address, cpu_data_address, value_len * sizeof(V), cudaMemcpyHostToDevice);
        memcpy(memcpy_buffer_cpu + j * value_len, cpu_data_address, value_len * sizeof(V));
        copyback_cursor[j] = i;
        gpu_value_ptrs[j] = gpu_value_ptr;
        j++;
        kvs_[0].first->Insert(keys[i], gpu_value_ptr);
      }
    }

    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu, total * value_len * sizeof(V), cudaMemcpyHostToDevice);
  }
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

  Status Remove(K key) {
    for (auto kv : kvs_) {
      kv.first->Remove(key);
    }
    return Status::OK();
  }

  int64 Size() const {
    int64 total_size = 0;
    for (auto kv : kvs_) {
      total_size += kv.first->Size();
    }
    return total_size;
  }

  int64 CacheSize() const {
    return cache_capacity_;
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>* >* value_ptr_list) {
    for (auto kv : kvs_) {
      TF_CHECK_OK(kv.first->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list,
                    std::vector<int64>* version_list, std::vector<int64>* freq_list,
                    const EmbeddingConfig& emb_config, EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter,
                    embedding::Iterator** it) {
    for (auto kv : kvs_) {
      std::vector<ValuePtr<V>* > value_ptr_list;
      std::vector<K> key_list_tmp;
      TF_CHECK_OK(kv.first->GetSnapshot(&key_list_tmp, &value_ptr_list));
      if (key_list_tmp.empty()) {
        *it = kv.first->GetIterator();
        continue;
      }
      for (int64 i = 0; i < key_list_tmp.size(); ++i) {
        V* val = value_ptr_list[i]->GetValue(emb_config.emb_index, GetOffset(emb_config.emb_index));
        V* primary_val = value_ptr_list[i]->GetValue(emb_config.primary_emb_index, GetOffset(emb_config.primary_emb_index));
        key_list->push_back(key_list_tmp[i]);
        if (emb_config.filter_freq != 0 || is_multi_level_
            || emb_config.record_freq) {
            int64 dump_freq = filter->GetFreq(key_list_tmp[i], value_ptr_list[i]);
            freq_list->push_back(dump_freq);
        }
        if (emb_config.steps_to_live != 0 || emb_config.record_version) {
            int64 dump_version = value_ptr_list[i]->GetStep();
            version_list->push_back(dump_version);
        }
        if (val != nullptr && primary_val != nullptr) {
          value_list->push_back(val);
        } else if (val == nullptr && primary_val != nullptr) {
          // only forward, no backward
          value_list->push_back(reinterpret_cast<V*>(-1));
        } else {
          // feature filtered
          value_list->push_back(nullptr);
        }
        // storage_manager_->FreeValuePtr(value_ptr_list[i]);
      }
    }
    return key_list->size();
  }

  Status Shrink(const EmbeddingConfig& emb_config, int64 value_len) {
    mutex_lock l(mu_);
    for (auto kv : kvs_) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>* > value_ptr_list;
      TF_CHECK_OK(kv.first->GetSnapshot(&key_list, &value_ptr_list));
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
      for (int64 i = 0; i < key_list.size(); ++i) {
        V* val = value_ptr_list[i]->GetValue(emb_config.primary_emb_index, GetOffset(emb_config.primary_emb_index));;
        if (val != nullptr) {
          V l2_weight = 0.0;
          for (int64 j = 0; j < value_len; j++) {
              l2_weight += val[j] * val[j];
          }
          l2_weight *= 0.5;
          if (l2_weight < emb_config.l2_weight_threshold) {
            to_deleted.push_back(std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
          }
        }
      }
      for (const auto it : to_deleted) {
        // TODO memory recycle
        (it.second)->Destroy(kv.second);
        delete it.second;
        kv.first->Remove(it.first);
      }
    }
    return Status::OK();
  }

  Status Shrink(int64 gs, int64 steps_to_live) {
    mutex_lock l(mu_);
    for (auto kv : kvs_) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>* > value_ptr_list;
      TF_CHECK_OK(kv.first->GetSnapshot(&key_list, &value_ptr_list));
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
      for (int64 i = 0; i < key_list.size(); ++i) {
        int64 version = value_ptr_list[i]->GetStep();
        if (version == -1) {
          value_ptr_list[i]->SetStep(gs);
        } else {
          if (gs - version > steps_to_live) {
            to_deleted.emplace_back(std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
          }
        }
      }
      for (const auto it : to_deleted) {
        // TODO memory recycle
        (it.second)->Destroy(kv.second);
        delete it.second;
        kv.first->Remove(it.first);
      }
    }
    return Status::OK();
  }

  Status Destroy() {
    if (eviction_thread_) {
      mutex_lock l(mu_);
      shutdown_ = true;
    }
    delete eviction_thread_;
    mutex_lock l(mu_);
    std::vector<K> key_list;
    std::vector<ValuePtr<V>* > value_ptr_list;
    kvs_[0].first->GetSnapshot(&key_list, &value_ptr_list);
    for (auto value_ptr : value_ptr_list) {
      value_ptr->Destroy(kvs_[0].second);
      delete value_ptr;
    }
    return Status::OK();
  }

  Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    for (auto kv : kvs_) {
      TF_CHECK_OK(kv.first->BatchCommit(keys, value_ptrs));
    }
    return Status::OK();
  }

  BatchCache<K>* Cache() {
    return cache_;
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    TF_CHECK_OK(kvs_[0].first->Commit(key, value_ptr));
    return Status::OK();
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) {
    for (auto kv : kvs_) {
      kv.first->FreeValuePtr(value_ptr);
    }
  }

  mutex* get_mutex() { return &mu_; }



 private:
  void BatchEviction() {
    Env* env = Env::Default();
    const int EvictionSize = 10000;
    if (cache_capacity_ == -1) {
      while (true) {
        mutex_lock l(mu_);
        if (done_) {
          break;
        }
      }
    }
    K evic_ids[EvictionSize];
    while (true) {
      mutex_lock l(mu_);
      if (shutdown_) {
        break;
      }
      // add WaitForMilliseconds() for sleep if necessary
      for (int i = 0; i < value_ptr_out_of_date_.size(); i++) {
        value_ptr_out_of_date_[i]->Destroy(kvs_[0].second);
        delete value_ptr_out_of_date_[i];
      }
      value_ptr_out_of_date_.clear();
      int cache_count = cache_->size();
      if (cache_count > cache_capacity_) {
        // eviction
        int k_size = cache_count - cache_capacity_;
        k_size = std::min(k_size, EvictionSize);
        size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
        ValuePtr<V>* value_ptr;
        if (sc_.type == StorageType::HBM_DRAM) {
          std::vector<K> keys;
          std::vector<ValuePtr<V>*> value_ptrs;
          LOG(INFO) << "Cache_count: " << cache_count;
          timespec start, end;

          clock_gettime(CLOCK_MONOTONIC, &start);
          for (int64 i = 0; i < true_size; ++i) {
            if (kvs_[0].first->Lookup(evic_ids[i], &value_ptr).ok()) {
              TF_CHECK_OK(kvs_[0].first->Remove(evic_ids[i]));
              keys.push_back(evic_ids[i]);
              value_ptrs.push_back(value_ptr);
            }
          }

          BatchCommit(keys, value_ptrs);

          for (int64 i = 0; i < true_size; ++i) {
            // value_ptrs[i]->Destroy(kvs_[0].second);
            // delete value_ptrs[i];
          }
          clock_gettime(CLOCK_MONOTONIC, &end);
          LOG(INFO) << "Total Evict Time: " << ((double)(end.tv_sec - start.tv_sec) * 1000000000 + end.tv_nsec - start.tv_nsec) / 1000000 << "ms";
        } else {
          for (int64 i = 0; i < true_size; ++i) {
            if (kvs_[0].first->Lookup(evic_ids[i], &value_ptr).ok()) {
              TF_CHECK_OK(kvs_[1].first->Commit(evic_ids[i], value_ptr));
              TF_CHECK_OK(kvs_[0].first->Remove(evic_ids[i]));
              value_ptr_out_of_date_.emplace_back(value_ptr);
            } else {
              // bypass
            }
          }
        }
      }
    }
  }

 private:
  int32 hash_table_count_;
  std::string name_;
  std::vector<std::pair<KVInterface<K, V>*, Allocator*>> kvs_;
  std::vector<ValuePtr<V>*> value_ptr_out_of_date_;
  std::function<ValuePtr<V>*(Allocator*, size_t)> new_value_ptr_fn_;
  StorageConfig sc_;
  bool is_multi_level_;

  int64 alloc_len_;
  int64 total_dims_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  Thread* eviction_thread_;
  BatchCache<K>* cache_;
  int64 cache_capacity_;
  mutex mu_;
  volatile bool shutdown_ GUARDED_BY(mu_) = false;

  volatile bool done_ = false;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

};

} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTILEVEL_EMBEDDING_H_
