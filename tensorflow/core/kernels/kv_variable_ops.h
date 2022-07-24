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

#ifndef TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_
#define TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/multilevel_embedding.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#if TF_ENABLE_GPU_EV
#include "tensorflow/core/kernels/kv_variable_ops_gpu.h"
#endif // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {
  const int kSavedPartitionNum = 1000;
}

template<class T>
class EVKeyDumpIterator: public  DumpIterator<T> {
 public:
  EVKeyDumpIterator(std::vector<T>& key_list):key_list_(key_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < key_list_.size();
  }

  T Next() {
    return key_list_[keys_idx_++];
  }

 private:
  int64 keys_idx_;
  std::vector<T>& key_list_;
};

template<class K, class T>
class EVValueDumpIterator: public  DumpIterator<T> {
 public:
  EVValueDumpIterator(EmbeddingVar<K, T>*& ev,
      std::vector<T* >& valueptr_list)
        : ev_(ev),
          valueptr_list_(valueptr_list) {
    keys_idx_ = 0;
    col_idx_ = 0;
  }

  bool HasNext() const {
    if (keys_idx_ < valueptr_list_.size()) {
      if (keys_idx_ < valueptr_list_.size() - 1)
        return true;
      else
        return col_idx_ < ev_->ValueLen();
    } else
      return false;
  }

  T Next() {
    if (col_idx_ >= ev_->ValueLen()) {
      keys_idx_++;
      col_idx_ = 0;
    }
    Eigen::array<Eigen::DenseIndex, 1> dims({ev_->ValueLen()});
    typename TTypes<T>::Flat value_flat =
      typename TTypes<T>::Flat(valueptr_list_[keys_idx_], dims);
    return value_flat(col_idx_++);
  }

 private:
  EmbeddingVar<K, T>* ev_;
  std::vector<T* >& valueptr_list_;
  int64 keys_idx_;
  int64 col_idx_;
};

template<class T>
class EVVersionDumpIterator: public  DumpIterator<T> {
 public:
  EVVersionDumpIterator(std::vector<T >& version_list)
      : version_list_(version_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < version_list_.size();
  }

  T Next() {
    return version_list_[keys_idx_++];
  }

 private:
  std::vector<T>& version_list_;
  int64 keys_idx_;
};

template<class T>
class EVFreqDumpIterator: public  DumpIterator<T> {
 public:
  EVFreqDumpIterator(std::vector<T>& freq_list) : freq_list_(freq_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < freq_list_.size();
  }

  T Next() {
    return freq_list_[keys_idx_++];
  }

 private:
  std::vector<T>& freq_list_;
  int64 keys_idx_;
};

template<class T>
class EVOffsetDumpIterator: public  DumpIterator<T> {
 public:
  EVOffsetDumpIterator(std::vector<T>& offset_list)
      : offset_list_(offset_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < offset_list_.size();
  }

  T Next() {
    return offset_list_[keys_idx_++];
  }

 private:
  std::vector<T>& offset_list_;
  int64 keys_idx_;
};

template <class K, class V>
Status GetInputEmbeddingVar(OpKernelContext* ctx, int input,
                            EmbeddingVar<K, V>** var) {
  if (LookupResource(ctx, HandleFromInput(ctx, input), var).ok()) {
    return Status::OK();
  } else {
    return errors::Internal("Invalid versioned variable reference.");
  }
}

template <class K, class V>
Status DumpEmbeddingValues(EmbeddingVar<K, V>* ev,
    const string& tensor_key, BundleWriter* writer,
    Tensor* part_offset_tensor) {
  std::vector<K> tot_key_list;
  std::vector<V* > tot_valueptr_list;
  std::vector<int64> tot_version_list;
  std::vector<int64> tot_freq_list;
  std::vector<K> tot_key_filter_list;
  std::vector<int64> tot_freq_filter_list;
  std::vector<int64> tot_version_filter_list;
  embedding::Iterator* it = nullptr;
  mutex_lock l(*ev->storage_manager()->get_mutex());
  int64 total_size = ev->GetSnapshot(&tot_key_list,
      &tot_valueptr_list, &tot_version_list, &tot_freq_list, &it);
  VLOG(1) << "EV:" << tensor_key << ", save size:" << total_size;
  int64 iterator_size = 0;
  if (it != nullptr) {
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      ++iterator_size;
    }
  }

  std::vector<std::vector<K> > key_list_parts;
  std::vector<std::vector<V* > > valueptr_list_parts;
  std::vector<std::vector<int64 > > version_list_parts;
  std::vector<std::vector<int64 > > freq_list_parts;

  std::vector<std::vector<K> > key_filter_list_parts;
  std::vector<std::vector<int64 > > version_filter_list_parts;
  std::vector<std::vector<int64 > > freq_filter_list_parts;

  std::vector<K> partitioned_tot_key_list;
  std::vector<V* > partitioned_tot_valueptr_list;
  std::vector<int64> partitioned_tot_version_list;
  std::vector<int64> partitioned_tot_freq_list;
  std::vector<K> partitioned_tot_key_filter_list;
  std::vector<int64> partitioned_tot_version_filter_list;
  std::vector<int64> partitioned_tot_freq_filter_list;
  std::vector<int64> part_filter_offset;

  key_list_parts.resize(kSavedPartitionNum);
  valueptr_list_parts.resize(kSavedPartitionNum);
  version_list_parts.resize(kSavedPartitionNum);
  freq_list_parts.resize(kSavedPartitionNum);
  key_filter_list_parts.resize(kSavedPartitionNum);
  version_filter_list_parts.resize(kSavedPartitionNum);
  freq_filter_list_parts.resize(kSavedPartitionNum);
  part_filter_offset.resize(kSavedPartitionNum + 1);
  //partitioned_tot_key_list.resize(tot_key_list.size());
  //partitioned_tot_valueptr_list.resize(tot_valueptr_list.size());

  // save the ev with kSavedPartitionNum piece of tensor
  // so that we can dynamically load ev with changed partition number
  int64 filter_freq = ev->MinFreq();
  for (size_t i = 0; i < tot_key_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        if (tot_valueptr_list[i] == reinterpret_cast<V*>(-1)) {
            // only forward, no backward, bypass
        } else if (tot_valueptr_list[i] == nullptr) {
          key_filter_list_parts[partid].push_back(tot_key_list[i]);
        } else {
          key_list_parts[partid].push_back(tot_key_list[i]);
          valueptr_list_parts[partid].push_back(tot_valueptr_list[i]);
        }
        break;
      }
    }
  }

  for (size_t i = 0; i < tot_version_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        if (tot_valueptr_list[i] == nullptr) {
          version_filter_list_parts[partid].push_back(tot_version_list[i]);
        } else {
          version_list_parts[partid].push_back(tot_version_list[i]);
        }
        break;
      }
    }
  }
  
  for (size_t i = 0; i < tot_freq_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        if (tot_valueptr_list[i] == nullptr) {
          freq_filter_list_parts[partid].push_back(tot_freq_list[i]);
        } else {
          freq_list_parts[partid].push_back(tot_freq_list[i]);
        }
        break;
      }
    }
  }
  // LOG(INFO) << "EV:" << tensor_key << ", key_list_parts:" << key_list_parts.size();

  auto part_offset_flat = part_offset_tensor->flat<int32>();
  part_offset_flat(0) = 0;
  part_filter_offset[0] = 0;
  int ptsize = 0;
  for (int partid = 0; partid < kSavedPartitionNum; partid++) {
    std::vector<K>& key_list = key_list_parts[partid];
    std::vector<V* >& valueptr_list = valueptr_list_parts[partid];
    std::vector<int64>& version_list = version_list_parts[partid];
    std::vector<int64>& freq_list = freq_list_parts[partid];
    std::vector<K>& key_filter_list = key_filter_list_parts[partid];
    std::vector<int64>& version_filter_list =
      version_filter_list_parts[partid];
    std::vector<int64>& freq_filter_list = freq_filter_list_parts[partid];

    ptsize += key_list.size();
    for (int inpid = 0; inpid < key_list.size(); inpid++) {
      partitioned_tot_key_list.push_back(key_list[inpid]);
      partitioned_tot_valueptr_list.push_back(valueptr_list[inpid]);
    }
    for (int inpid = 0; inpid < version_list.size(); inpid++) {
      partitioned_tot_version_list.push_back(version_list[inpid]);
    }
    for (int inpid = 0; inpid < freq_list.size(); inpid++) {
      partitioned_tot_freq_list.push_back(freq_list[inpid]);
    }
    for (int inpid = 0; inpid < key_filter_list.size(); inpid++) {
      partitioned_tot_key_filter_list.push_back(key_filter_list[inpid]);
    }
    for (int inpid = 0; inpid < version_filter_list.size(); inpid++) {
      partitioned_tot_version_filter_list.push_back(version_filter_list[inpid]);
    }
    for (int inpid = 0; inpid < freq_filter_list.size(); inpid++) {
      partitioned_tot_freq_filter_list.push_back(freq_filter_list[inpid]);
    }

    part_offset_flat(partid + 1) = part_offset_flat(partid) + key_list.size();
    part_filter_offset[partid + 1] = part_filter_offset[partid] + key_filter_list.size();
  }
  // TODO: DB iterator not support partition_offset
  writer->Add(tensor_key + "-partition_offset", *part_offset_tensor);
  for(int i = 0; i <  kSavedPartitionNum + 1; i++) {
    part_offset_flat(i) = part_filter_offset[i];
  }
  writer->Add(tensor_key + "-partition_filter_offset", *part_offset_tensor);

  VLOG(1) << "EV before partition:" << tensor_key << ", keysize:" <<  tot_key_list.size()
          << ", valueptr size:" << tot_valueptr_list.size();
  VLOG(1) << "EV after partition:" << tensor_key << ", ptsize:" << ptsize
          << ", keysize:"<<  partitioned_tot_key_list.size()
          <<", valueptr size:" << partitioned_tot_valueptr_list.size();

  size_t bytes_limit = 8 << 20;
  char* dump_buffer = (char*)malloc(sizeof(char) * bytes_limit);
  Status st;

  EVKeyDumpIterator<K> ev_key_dump_iter(partitioned_tot_key_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-keys", writer, dump_buffer,
                                 bytes_limit, &ev_key_dump_iter,
                                 TensorShape({partitioned_tot_key_list.size() + iterator_size}),
                                 it);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVValueDumpIterator<K, V> ev_value_dump_iter(ev, partitioned_tot_valueptr_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-values", writer, dump_buffer,
      bytes_limit, &ev_value_dump_iter,
      TensorShape({partitioned_tot_key_list.size() + iterator_size, ev->ValueLen()}),
      it, ev->storage_manager()->GetOffset(ev->GetEmbeddingIndex()));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVVersionDumpIterator<int64> ev_version_dump_iter(partitioned_tot_version_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-versions", writer, dump_buffer,
      bytes_limit, &ev_version_dump_iter,
      TensorShape({partitioned_tot_version_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVFreqDumpIterator<int64> ev_freq_dump_iter(partitioned_tot_freq_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-freqs", writer, dump_buffer,
      bytes_limit, &ev_freq_dump_iter,
      TensorShape({partitioned_tot_freq_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVKeyDumpIterator<K> ev_key_filter_dump_iter(partitioned_tot_key_filter_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-keys_filtered",
      writer, dump_buffer, bytes_limit, &ev_key_filter_dump_iter,
      TensorShape({partitioned_tot_key_filter_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVVersionDumpIterator<int64> ev_version_filter_dump_iter(
      partitioned_tot_version_filter_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-versions_filtered",
      writer, dump_buffer, bytes_limit, &ev_version_filter_dump_iter,
      TensorShape({partitioned_tot_version_filter_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVFreqDumpIterator<int64> ev_freq_filter_dump_iter(
      partitioned_tot_freq_filter_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-freqs_filtered",
      writer, dump_buffer, bytes_limit, &ev_freq_filter_dump_iter,
      TensorShape({partitioned_tot_freq_filter_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  free(dump_buffer);

  if (it != nullptr) {
    delete it;
  }
  return Status::OK();
}

namespace {
const static string part_str = "part_";
}

template<typename K, typename V>
Status DynamicRestoreValue(EmbeddingVar<K, V>* ev, BundleReader* reader,
    std::string name_string, int orig_partnum,
    int64 partition_id = 0, int64 partition_num = 1) {
  string curr_partid_str = std::to_string(partition_id);
  bool filter_flag = true;
  bool restore_filter_flag = true;
  for (int i = 0; i < orig_partnum; i++) {
    string part_id = std::to_string(i);
    string pre_subname =
      name_string.substr(0, name_string.find("part_"));
    string post_subname =
      name_string.substr(name_string.find("part_")
          + part_str.size() + curr_partid_str.size());
    string tensor_name =
      pre_subname + part_str + part_id + post_subname;

    string tensor_key = tensor_name + "-keys";
    string tensor_value = tensor_name + "-values";
    string tensor_version = tensor_name + "-versions";
    string tensor_freq = tensor_name + "-freqs";
    
    TensorShape key_shape, value_shape, version_shape, freq_shape;
    Status st = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupTensorShape(tensor_value, &value_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupTensorShape(tensor_version, &version_shape);
    if (!st.ok()) {
      return st;
    }

    st = reader->LookupTensorShape(tensor_freq, &freq_shape);
    if (!st.ok()) {
      if (st.code() == error::NOT_FOUND) {
        freq_shape = version_shape;
      }else {
        return st;
      }
    }

    reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupHeader(tensor_value,
        sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupHeader(tensor_version,
        sizeof(int64) * version_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupHeader(tensor_freq,
        sizeof(int64) * freq_shape.dim_size(0));
    if (!st.ok()) {
      if (st.code() == error::NOT_FOUND) {
        filter_flag = false;
      }else {
        return st;
      }
    }

    size_t buffer_size = 8 << 20;
    RestoreBuffer restore_buff;
    restore_buff.key_buffer = new char[buffer_size];
    restore_buff.value_buffer = new char[buffer_size];
    restore_buff.version_buffer = new char[buffer_size];
    restore_buff.freq_buffer = new char[buffer_size];

    size_t key_bytes_read = 0;
    size_t value_bytes_read = 0;
    size_t version_bytes_read = 0;
    size_t freq_bytes_read = 0;
    int64 tot_key_num = key_shape.dim_size(0);
    size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);

    while(tot_key_num > 0) {
      size_t read_key_num = std::min(std::min(buffer_size / sizeof(K),
            buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
      read_key_num = std::min((int64)read_key_num, tot_key_num);
      reader->LookupSegment(tensor_key, read_key_num * sizeof(K),
          restore_buff.key_buffer, key_bytes_read);
      reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes,
          restore_buff.value_buffer, value_bytes_read);
      reader->LookupSegment(tensor_version, read_key_num * sizeof(int64),
          restore_buff.version_buffer, version_bytes_read);
      if (version_bytes_read == 0) {
        memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
      }
      if (filter_flag) {
        reader->LookupSegment(tensor_freq, (read_key_num + 1)* sizeof(int64),
            restore_buff.freq_buffer, freq_bytes_read);
      }else {
        int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
        freq_tmp[0] = 0;
        for (int64 i = 1; i < read_key_num + 1; i++) {
          freq_tmp[i] = ev->MinFreq();
        }
      }

      if (key_bytes_read > 0) {
        read_key_num = key_bytes_read / sizeof(K);
        VLOG(2) << "repartition, read_key_num:" << read_key_num;
        st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
            partition_id, partition_num, false);
        if (!st.ok()) {
          return st;
        }
        tot_key_num -= read_key_num;
      }
    }
  }
  return Status::OK();
}

template<typename K, typename V>
Status EVRestoreNoPartition(EmbeddingVar<K, V>* ev, BundleReader* reader,
    std::string tensor_key, std::string tensor_value,
    std::string tensor_version, std::string tensor_freq) {
  TensorShape key_shape;
  TensorShape value_shape;
  TensorShape version_shape;
  TensorShape freq_shape;
  TensorShape key_filter_shape;
  TensorShape version_filter_shape;
  TensorShape freq_filter_shape;

  Status st;
  reader->LookupTensorShape(tensor_key, &key_shape);
  reader->LookupTensorShape(tensor_value, &value_shape);
  reader->LookupTensorShape(tensor_version, &version_shape);
  st = reader->LookupTensorShape(tensor_freq, &freq_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      freq_shape = version_shape;
    }else {
      return st;
    }
  }
  st = reader->LookupTensorShape(tensor_key + "_filtered", &key_filter_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      key_filter_shape = key_shape;
    }else {
      return st;
    }
  }
  st = reader->LookupTensorShape(tensor_version + "_filtered",
      &version_filter_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      version_filter_shape = version_shape;
    }else {
      return st;
    }
  }
  st = reader->LookupTensorShape(tensor_freq + "_filtered",
      &freq_filter_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      freq_filter_shape = freq_shape;
    }else {
      return st;
    }
  }

  bool filter_flag = true;
  bool restore_filter_flag = true;
  st = reader->LookupHeader(tensor_key,
      sizeof(K) * key_shape.dim_size(0));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_value,
      sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_version,
      sizeof(int64) * version_shape.dim_size(0));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_freq,
      sizeof(int64) * freq_shape.dim_size(0));
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      filter_flag = false;
    }else {
      return st;
    }
  }
  st = reader->LookupHeader(tensor_key + "_filtered",
      sizeof(K) * key_filter_shape.dim_size(0));
  if (!st.ok()){
    if (st.code() == error::NOT_FOUND){
      restore_filter_flag=false;
    } else {
      return st;
    }
  }
  st = reader->LookupHeader(tensor_version + "_filtered",
      sizeof(K) * version_filter_shape.dim_size(0));
  if (!st.ok() && st.code() != error::NOT_FOUND){
    return st;
  }
  st = reader->LookupHeader(tensor_freq + "_filtered",
      sizeof(K) * freq_filter_shape.dim_size(0));
  if (!st.ok() && st.code() != error::NOT_FOUND){
    return st;
  }

  size_t buffer_size = 8 << 20;
  RestoreBuffer restore_buff;
  restore_buff.key_buffer = new char[buffer_size];
  restore_buff.value_buffer = new char[buffer_size];
  restore_buff.version_buffer = new char[buffer_size];
  restore_buff.freq_buffer = new char[buffer_size];

  size_t key_bytes_read = 0;
  size_t value_bytes_read = 0;
  size_t version_bytes_read = 0;
  size_t freq_bytes_read = 0;
  size_t key_filter_bytes_read = 0;
  size_t version_filter_bytes_read = 0;
  size_t freq_filter_bytes_read = 0;

  int64 tot_key_num = key_shape.dim_size(0);
  size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
  std::string key_str = "|";
  while(tot_key_num > 0) {
    size_t read_key_num = std::min(
        std::min(buffer_size / sizeof(K),
          buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
    read_key_num = std::min((int64)read_key_num, tot_key_num);
    reader->LookupSegment(tensor_key, read_key_num * sizeof(K),
        restore_buff.key_buffer, key_bytes_read);
    reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes,
        restore_buff.value_buffer, value_bytes_read);
    reader->LookupSegment(tensor_version, read_key_num * sizeof(int64),
        restore_buff.version_buffer, version_bytes_read);
    if (version_bytes_read == 0) {
        memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
    }
    if (filter_flag) {
      reader->LookupSegment(tensor_freq, read_key_num * sizeof(int64),
          restore_buff.freq_buffer, freq_bytes_read);
    } else {
      int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
      freq_tmp[0] = 0;
      for (int64 i = 1; i < read_key_num + 1; i++) {
        freq_tmp[i] = ev->MinFreq();
      }
    }
    if (key_bytes_read > 0) {
      read_key_num = key_bytes_read / sizeof(K);
      VLOG(2) << "restore, read_key_num:" << read_key_num;

      st = ev->Import(restore_buff, read_key_num, 1, 0, 1, false);
      if (!st.ok())
        return st;
      tot_key_num -= read_key_num;
    }
  }
  
  if (restore_filter_flag) {
    int64 tot_key_filter_num = key_filter_shape.dim_size(0);
    while (tot_key_filter_num > 0) {
      size_t read_key_num = std::min(buffer_size / sizeof(K),
          buffer_size / sizeof(int64));
      read_key_num = std::min((int64)read_key_num, tot_key_filter_num);
      reader->LookupSegment(tensor_key + "_filtered",
          read_key_num * sizeof(K), restore_buff.key_buffer,
          key_filter_bytes_read);
      reader->LookupSegment(tensor_version + "_filtered",
          read_key_num * sizeof(int64), restore_buff.version_buffer,
          version_filter_bytes_read);
      reader->LookupSegment(tensor_freq + "_filtered",
          read_key_num * sizeof(int64), restore_buff.freq_buffer,
          freq_filter_bytes_read);
      if (key_filter_bytes_read > 0) {
        read_key_num = key_filter_bytes_read / sizeof(K);
        VLOG(2) << "restore, read_key_num:" << read_key_num;

        st = ev->Import(restore_buff, read_key_num, 1, 0, 1, true);
        if (!st.ok())
          return st;
        tot_key_filter_num -= read_key_num;
      }
    }
  }
  
  return Status::OK();
}

inline bool IsOldCheckpoint(const std::string& name_string,
    const std::string& curr_partid_str, BundleReader* reader,
    const std::string& part_offset_tensor_suffix) {
  // then check whether checkpoint is in old form
  bool is_oldform = false;

  string part_id = std::to_string(0);
  string pre_subname =
    name_string.substr(0, name_string.find(part_str));
  string post_subname = name_string.substr(
      name_string.find(part_str) + part_str.size() + curr_partid_str.size());
  string tensor_name = pre_subname + part_str + part_id + post_subname;

  TensorShape part_offset_shape;
  DataType part_offset_type;
  Status form_st = reader->LookupDtypeAndShape(
      tensor_name + part_offset_tensor_suffix,
      &part_offset_type, &part_offset_shape);
  if (!form_st.ok()) {
    is_oldform = true;
  }
  return is_oldform;
}

template<typename K, typename V>
Status EVRestoreOldFromCheckpoint(EmbeddingVar<K, V>* ev,
    const std::string& name_string, const std::string& curr_partid_str,
    const std::string& key_suffix, int partition_id,
    BundleReader* reader, int partition_num) {
  // first get original partition number
  int orig_partnum = 0;
  for (;  ; orig_partnum++) {
    string part_id = std::to_string(orig_partnum);
    string pre_subname = name_string.substr(0, name_string.find(part_str));
    string post_subname = name_string.substr(name_string.find(part_str)
        + part_str.size() + curr_partid_str.size());
    string tensor_name = pre_subname + part_str + part_id + post_subname;

    string tensor_key = tensor_name + key_suffix;
    TensorShape key_shape;
    Status st = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!st.ok()) {
      break;
    }
  }

  VLOG(1) << "old form, EV name:" << name_string
          << ", partition_id:" << curr_partid_str
          << ", old partition_num:" << orig_partnum
          << ", new partition num:" << partition_num;
  Status s = DynamicRestoreValue(ev, reader, name_string,
      orig_partnum, partition_id, partition_num);
  if (!s.ok()) {
    LOG(FATAL) <<  "EV restoring fail:" << s.ToString();
  }
}

template<typename K, typename V>
Status EVRestoreDynamically(EmbeddingVar<K, V>* ev,
    const std::string& name_string, int partition_id,
    int partition_num, OpKernelContext* context,
    BundleReader* reader, const std::string& part_offset_tensor_suffix,
    const std::string& key_suffix, const std::string& value_suffix,
    const std::string& version_suffix, const std::string& freq_suffix) {

  // first check whether there is partition
  if (name_string.find(part_str) == std::string::npos) {
    Status s = EVRestoreNoPartition(
        ev, reader, name_string + key_suffix,
        name_string + value_suffix, name_string + version_suffix,
        name_string + freq_suffix);
    if (!s.ok()) {
      LOG(FATAL) <<  "EV restoring fail:" << s.ToString();
    }
    return s;
  }

  const string& curr_partid_str = std::to_string(partition_id);
  auto is_oldform = IsOldCheckpoint(name_string, curr_partid_str,
      reader, part_offset_tensor_suffix);

  if (is_oldform) {
    EVRestoreOldFromCheckpoint(ev, name_string, curr_partid_str, key_suffix,
        partition_id, reader, partition_num);
  } else {
    // first find out which sub parts we should load
    bool filter_flag = true;
    bool restore_filter_flag = true;
    std::vector<int> loaded_parts;
    for (int i = 0; i < kSavedPartitionNum; i++) {
      if (i % partition_num == partition_id) {
        loaded_parts.push_back(i);
      }
    }

    // then we use primary partition number to compose with
    // sub partition number
    VLOG(1) << "new form:" << name_string
            << ", partition_id:" << partition_id
            << ", partition_num:" << partition_num;

    int orig_partnum = 0;
    size_t buffer_size = 8 << 20;
    RestoreBuffer restore_buff;
    restore_buff.key_buffer = new char[buffer_size];
    restore_buff.value_buffer = new char[buffer_size];
    restore_buff.version_buffer = new char[buffer_size];
    restore_buff.freq_buffer = new char[buffer_size];

    for (;  ; orig_partnum++) {
      string part_id = std::to_string(orig_partnum);
      string pre_subname = name_string.substr(0, name_string.find(part_str));
      string post_subname = name_string.substr(name_string.find(part_str)
          + part_str.size() + curr_partid_str.size());
      string tensor_name = pre_subname + part_str + part_id + post_subname;

      // first check whether is  old ckpt form
      string tensor_key = tensor_name + key_suffix;
      string tensor_value = tensor_name + value_suffix;
      string tensor_version = tensor_name + version_suffix;
      string tensor_freq = tensor_name + freq_suffix;
      TensorShape key_shape, value_shape, version_shape, freq_shape;
      TensorShape key_filter_shape, version_filter_shape, freq_filter_shape;
      Status st = reader->LookupTensorShape(tensor_key, &key_shape);
      if (!st.ok()) {
        VLOG(1) << "ev part " << tensor_key
                << " not exist, reach the end of restoring";
        break;
      }
      st = reader->LookupTensorShape(tensor_value, &value_shape);
      if (!st.ok()) {
        break;
      }
      st = reader->LookupTensorShape(tensor_version, &version_shape);
      if (!st.ok()) {
        break;
      }
      st = reader->LookupTensorShape(tensor_freq, &freq_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          freq_shape = version_shape;
        } else {
          return st;
        }
      }
      st = reader->LookupTensorShape(tensor_key + "_filtered",
          &key_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          key_filter_shape = key_shape;
        } else {
          return st;
        }
      }
      st = reader->LookupTensorShape(tensor_version + "_filtered",
          &version_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          version_filter_shape = version_shape;
        } else {
          return st;
        }
      }
      st = reader->LookupTensorShape(tensor_freq + "_filtered",
          &freq_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          freq_filter_shape = freq_shape;
        }else {
          return st;
        }
      }

      reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
      if (!st.ok()) {
        break;
      }
      st = reader->LookupHeader(tensor_value,
          sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
      if (!st.ok()) {
        break;
      }
      st = reader->LookupHeader(tensor_version,
          sizeof(int64) * version_shape.dim_size(0));
      if (!st.ok()) {
        break;
      }
      st = reader->LookupHeader(tensor_freq,
          sizeof(int64) * freq_shape.dim_size(0));
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          filter_flag = false;
        }else {
          return st;
        }
      }
      st = reader->LookupHeader(tensor_key + "_filtered",
          sizeof(K) * key_filter_shape.dim_size(0));
      if (!st.ok()){
        if (st.code() == error::NOT_FOUND){
          restore_filter_flag=false;
        }else {
          return st;
        }
      }
      st = reader->LookupHeader(tensor_version + "_filtered",
          sizeof(K) * version_filter_shape.dim_size(0));
      if (!st.ok() && st.code() != error::NOT_FOUND){
        return st;
      }
      st = reader->LookupHeader(tensor_freq + "_filtered",
          sizeof(K) * freq_filter_shape.dim_size(0));
      if (!st.ok() && st.code() != error::NOT_FOUND){
        return st;
      }

      TensorShape part_offset_shape, part_filter_offset_shape;
      DataType part_offset_type, part_filter_offset_type;
      string offset_tensor_name = tensor_name + part_offset_tensor_suffix;
      string offset_filter_tensor_name =
          tensor_name + "-partition_filter_offset";
      st = reader->LookupDtypeAndShape(offset_tensor_name,
          &part_offset_type, &part_offset_shape);
      if (!st.ok()) {
          LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      st = reader->LookupDtypeAndShape(offset_filter_tensor_name,
          &part_filter_offset_type, &part_filter_offset_shape);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      Tensor part_offset_tensor;
      st = context->allocate_temp(part_offset_type,
          part_offset_shape, &part_offset_tensor);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      Tensor part_filter_offset_tensor;
      st = context->allocate_temp(part_filter_offset_type,
          part_filter_offset_shape, &part_filter_offset_tensor);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      st = reader->Lookup(offset_tensor_name, &part_offset_tensor);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      auto part_offset_flat = part_offset_tensor.flat<int32>();
      st = reader->Lookup(offset_filter_tensor_name, &part_filter_offset_tensor);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      auto part_filter_offset_flat = part_filter_offset_tensor.flat<int32>();

      for (size_t i = 0; i < loaded_parts.size(); i++) {
        int subpart_id = loaded_parts[i];
        int subpart_offset = part_offset_flat(subpart_id);

        size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
        int64 tot_key_num = part_offset_flat(subpart_id + 1) - subpart_offset;
        int64 key_part_offset = subpart_offset * sizeof(K);
        int64 value_part_offset = subpart_offset *  value_unit_bytes;
        int64 version_part_offset = subpart_offset * sizeof(int64);
        int64 freq_part_offset = subpart_offset * sizeof(int64);

        VLOG(1) << "dynamically load ev : " << name_string
                << ", subpartid:" << loaded_parts[i]
                << ", subpart_offset:" << subpart_offset
                << ", partition_id:" << partition_id
                << ", partition_num:" << partition_num
                << ", keynum:" << tot_key_num;

        int64 tot_key_bytes_read(0);
        int64 tot_value_bytes_read(0);
        int64 tot_version_bytes_read(0);
        int64 tot_freq_bytes_read(0);
        size_t key_bytes_read = 0;
        size_t value_bytes_read = 0;
        size_t version_bytes_read = 0;
        size_t freq_bytes_read = 0;
        while(tot_key_num > 0) {
          size_t read_key_num = std::min(std::min(buffer_size / sizeof(K),
                buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
          read_key_num = std::min((int64)read_key_num, tot_key_num);
          reader->LookupSegmentOffset(tensor_key,
              key_part_offset + tot_key_bytes_read, read_key_num * sizeof(K),
              restore_buff.key_buffer, key_bytes_read);

          reader->LookupSegmentOffset(tensor_value,
              value_part_offset + tot_value_bytes_read,
              read_key_num * value_unit_bytes, restore_buff.value_buffer,
              value_bytes_read);

          reader->LookupSegmentOffset(tensor_version,
              version_part_offset + tot_version_bytes_read,
              read_key_num * sizeof(int64), restore_buff.version_buffer,
              version_bytes_read);
          if (version_bytes_read == 0) {
             memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
          }
          if (filter_flag) {
            reader->LookupSegmentOffset(tensor_freq,
                freq_part_offset + tot_freq_bytes_read,
                read_key_num * sizeof(int64), restore_buff.freq_buffer,
                freq_bytes_read);
          } else {
            int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
            for (int64 i = 0; i < read_key_num; i++) {
              freq_tmp[i] = ev->MinFreq();
            }
          }
          if (key_bytes_read > 0) {
            read_key_num = key_bytes_read / sizeof(K);
            VLOG(2) << "restore, read_key_num:" << read_key_num;
            st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
                partition_id, partition_num, false);
            if (!st.ok()) {
              LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
            }
          }
          tot_key_num -= read_key_num;
          tot_key_bytes_read += key_bytes_read;
          tot_value_bytes_read += value_bytes_read;
          tot_version_bytes_read += version_bytes_read;
          tot_freq_bytes_read += freq_bytes_read;
        }

        if (restore_filter_flag) {
          int subpart_filter_offset = part_filter_offset_flat(subpart_id);
          int64 key_filter_part_offset = subpart_filter_offset * sizeof(K);
          int64 version_filter_part_offset = subpart_filter_offset * sizeof(int64);
          int64 freq_filter_part_offset = subpart_filter_offset * sizeof(int64);
          int64 tot_key_filter_num =
            part_filter_offset_flat(subpart_id + 1) - subpart_filter_offset;
          int64 tot_key_filter_bytes_read(0), tot_version_filter_bytes_read(0),
                tot_freq_filter_bytes_read(0);
          size_t key_filter_bytes_read = 0;
          size_t version_filter_bytes_read = 0;
          size_t freq_filter_bytes_read = 0;
          while (tot_key_filter_num > 0) {
            size_t read_key_num =
              std::min(buffer_size / sizeof(K), buffer_size / sizeof(int64));
            read_key_num = std::min((int64)read_key_num, tot_key_filter_num);
            reader->LookupSegmentOffset(tensor_key + "_filtered",
                key_filter_part_offset + key_filter_bytes_read,
                read_key_num * sizeof(K), restore_buff.key_buffer,
                key_filter_bytes_read);
            reader->LookupSegmentOffset(tensor_version + "_filtered",
                version_filter_part_offset + version_filter_bytes_read,
                read_key_num * sizeof(int64), restore_buff.version_buffer,
                version_filter_bytes_read);
            reader->LookupSegmentOffset(tensor_freq + "_filtered",
                freq_filter_part_offset + freq_filter_bytes_read,
                read_key_num * sizeof(int64), restore_buff.freq_buffer,
                freq_filter_bytes_read);
            if (key_filter_bytes_read > 0) {
              read_key_num = key_filter_bytes_read / sizeof(K);
              VLOG(2) << "restore, read_key_num:" << read_key_num;
              st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
                  partition_id, partition_num, true);
              if (!st.ok())
               return st;
              tot_key_filter_num -= read_key_num;
            }
          }
        }
      }
    }
  }
  return Status::OK();
}
#if GOOGLE_CUDA
#if TF_ENABLE_GPU_EV
template<typename K, typename V>
Status EVRestoreNoPartitionGPU(EmbeddingVarGPU<K, V>* ev, BundleReader* reader,
    std::string tensor_key, std::string tensor_value, OpKernelContext* context,
    std::string tensor_version, std::string tensor_freq) {
  TensorShape key_shape;
  TensorShape value_shape;

  Status st;
  reader->LookupTensorShape(tensor_key, &key_shape);
  reader->LookupTensorShape(tensor_value, &value_shape);
  const cudaStream_t& stream = context->eigen_device<GPUDevice>().stream();
  bool filter_flag = true;
  bool restore_filter_flag = true;
  st = reader->LookupHeader(tensor_key,
      sizeof(K) * key_shape.dim_size(0));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_value,
      sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
  if (!st.ok())
    return st;

  size_t buffer_size = 8 << 20;
  RestoreBuffer restore_buff;
  restore_buff.key_buffer = new char[buffer_size];
  restore_buff.value_buffer = new char[buffer_size];

  size_t key_bytes_read = 0;
  size_t value_bytes_read = 0;

  int64 tot_key_num = key_shape.dim_size(0);
  size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
  std::string key_str = "|";
  while(tot_key_num > 0) {
    size_t read_key_num = std::min(
        std::min(buffer_size / sizeof(K),
          buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
    read_key_num = std::min((int64)read_key_num, tot_key_num);
    reader->LookupSegment(tensor_key, read_key_num * sizeof(K),
        restore_buff.key_buffer, key_bytes_read);
    reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes,
        restore_buff.value_buffer, value_bytes_read);

    if (key_bytes_read > 0) {
      read_key_num = key_bytes_read / sizeof(K);
      VLOG(2) << "restore, read_key_num:" << read_key_num;

      st = ev->Import(restore_buff, read_key_num, 1, 0, 1, false, stream);
      if (!st.ok())
        return st;
      tot_key_num -= read_key_num;
    }
  }
  
  return Status::OK();
}

template<typename K, typename V>
Status EVRestoreDynamicallyGPU(EmbeddingVarGPU<K, V>* ev,
    const std::string& name_string, int partition_id,
    int partition_num, OpKernelContext* context,
    BundleReader* reader, const std::string& part_offset_tensor_suffix,
    const std::string& key_suffix, const std::string& value_suffix,
    const std::string& version_suffix, const std::string& freq_suffix) {

  // first check whether there is partition
  if (name_string.find(part_str) == std::string::npos) {
    Status s = EVRestoreNoPartitionGPU(
        ev, reader, name_string + key_suffix,
        name_string + value_suffix, context, name_string + version_suffix,
        name_string + freq_suffix);
    if (!s.ok()) {
      LOG(FATAL) <<  "EV restoring fail:" << s.ToString();
    }
    return s;
  }

  const string& curr_partid_str = std::to_string(partition_id);
  // first find out which sub parts we should load
  std::vector<int> loaded_parts;
  for (int i = 0; i < kSavedPartitionNum; i++) {
    if (i % partition_num == partition_id) {
      loaded_parts.push_back(i);
    }
  }

  // then we use primary partition number to compose with
  // sub partition number
  VLOG(1) << "new form:" << name_string
          << ", partition_id:" << partition_id
          << ", partition_num:" << partition_num;

  int orig_partnum = 0;
  size_t buffer_size = 8 << 20;
  RestoreBuffer restore_buff;
  restore_buff.key_buffer = new char[buffer_size];
  restore_buff.value_buffer = new char[buffer_size];
  const cudaStream_t& stream = context->eigen_device<GPUDevice>().stream();


  for (;  ; orig_partnum++) {
    string part_id = std::to_string(orig_partnum);
    string pre_subname = name_string.substr(0, name_string.find(part_str));
    string post_subname = name_string.substr(name_string.find(part_str)
        + part_str.size() + curr_partid_str.size());
    string tensor_name = pre_subname + part_str + part_id + post_subname;

    // first check whether is  old ckpt form
    string tensor_key = tensor_name + key_suffix;
    string tensor_value = tensor_name + value_suffix;
    TensorShape key_shape, value_shape, version_shape, freq_shape;
    Status st = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!st.ok()) {
      VLOG(1) << "ev part " << tensor_key
              << " not exist, reach the end of restoring";
      break;
    }
    st = reader->LookupTensorShape(tensor_value, &value_shape);
    if (!st.ok()) {
      break;
    }

    reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
    if (!st.ok()) {
      break;
    }
    st = reader->LookupHeader(tensor_value,
        sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
    if (!st.ok()) {
      break;
    }

    TensorShape part_offset_shape;
    DataType part_offset_type;
    string offset_tensor_name = tensor_name + part_offset_tensor_suffix;
    st = reader->LookupDtypeAndShape(offset_tensor_name,
        &part_offset_type, &part_offset_shape);
    if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
    }
    Tensor part_offset_tensor;
    st = context->allocate_temp(part_offset_type,
        part_offset_shape, &part_offset_tensor);
    if (!st.ok()) {
      LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
    }
    st = reader->Lookup(offset_tensor_name, &part_offset_tensor);
    if (!st.ok()) {
      LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
    }
    auto part_offset_flat = part_offset_tensor.flat<int32>();

    for (size_t i = 0; i < loaded_parts.size(); i++) {
      int subpart_id = loaded_parts[i];
      int subpart_offset = part_offset_flat(subpart_id);

      size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
      int64 tot_key_num = part_offset_flat(subpart_id + 1) - subpart_offset;
      int64 key_part_offset = subpart_offset * sizeof(K);
      int64 value_part_offset = subpart_offset *  value_unit_bytes;
  
      VLOG(1) << "dynamically load ev : " << name_string
              << ", subpartid:" << loaded_parts[i]
              << ", subpart_offset:" << subpart_offset
              << ", partition_id:" << partition_id
              << ", partition_num:" << partition_num
              << ", keynum:" << tot_key_num;

      int64 tot_key_bytes_read(0);
      int64 tot_value_bytes_read(0);
      size_t key_bytes_read = 0;
      size_t value_bytes_read = 0;
      while(tot_key_num > 0) {
        size_t read_key_num = std::min(std::min(buffer_size / sizeof(K),
              buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
        read_key_num = std::min((int64)read_key_num, tot_key_num);
        reader->LookupSegmentOffset(tensor_key,
            key_part_offset + tot_key_bytes_read, read_key_num * sizeof(K),
            restore_buff.key_buffer, key_bytes_read);

        reader->LookupSegmentOffset(tensor_value,
            value_part_offset + tot_value_bytes_read,
            read_key_num * value_unit_bytes, restore_buff.value_buffer,
            value_bytes_read);

        if (key_bytes_read > 0) {
          read_key_num = key_bytes_read / sizeof(K);
          VLOG(2) << "restore, read_key_num:" << read_key_num;
          st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
              partition_id, partition_num, false, stream);
          if (!st.ok()) {
            LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
          }
        }
        tot_key_num -= read_key_num;
        tot_key_bytes_read += key_bytes_read;
        tot_value_bytes_read += value_bytes_read;
        
      }
    }
  }
  return Status::OK();
}
#endif // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_
