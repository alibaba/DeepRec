/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_RENDEZVOUS_MGR_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "tensorflow/core/distributed_runtime/data_worker_rendezvous_mgr_interface.h"
#include "tensorflow/core/framework/data_worker_rendezvous.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"

namespace tensorflow {
class GenericDataWorkerRendezvous;
class DataWorkerRecvTensorThread;

class DataWorkerRendezvousMgr: public DataWorkerRendezvousMgrInterface{
 public:
  struct DataWorkerRendezvousMgrOptions{
    int queue_size = 100;
    int num_recv_threads = 1;
    int num_send_threads = 4;
    string protocol = "grpc";
    bool fuse_recv = false;
  };

  explicit DataWorkerRendezvousMgr(const DataWorkerRendezvousMgrOptions& options);
  ~DataWorkerRendezvousMgr();

  void RecvLocalAsync(const DataWorkerRendezvous::ParsedKey& key,
                      DataWorkerRendezvous::DoneCallback done) override;

  void FuseRecvLocalAsync(const std::vector<DataWorkerRendezvous::ParsedKey>& keys,
                          DataWorkerRendezvous::FuseDoneCallback done) override;

  void RegisterDataWorker(const string& task_name, const string& host_port) override;

  void SetTensorNames(const std::vector<string>& tensor_names) override;

  DataWorkerRendezvous* Find();

 private:
  mutex mu_;
  const int queue_size_;
  const int num_recv_threads_;
  const int num_send_threads_;
  const string protocol_;
  const bool fuse_recv_;
  GenericDataWorkerRendezvous* rdwr_ GUARDED_BY(mu_); 
  GenericDataWorkerRendezvous* FindOrCreate();
};

// GenericDataWorkerRendezvous supports both grpc and grpc++ as the underlying
// communication protocol. It also supports transferring data from the local 
// data worker directly.
class GenericDataWorkerRendezvous: public DataWorkerRendezvous {
 public:
  GenericDataWorkerRendezvous(const int& queue_size,
                              const int& num_recv_threads,
                              const int& num_send_threads,
                              const string& protocol,
                              const bool& fuse_recv);
  ~GenericDataWorkerRendezvous();
  
  Status Initialize(WorkerSession* session) override;
  void StartAbort(const Status& status) override;
  void SetTensorNames(const std::vector<string>& tensor_names) override;
  Status SetRecvAttrs(const DataWorkerRendezvous::ParsedKey& key,
                      const AllocatorAttributes& alloc_attrs,
                      const string& device) override;
  void DataWorkerSendAsync(const DataWorkerRendezvous::ParsedKey& key,
                           const Tensor& val,
                           const DataWorkerRendezvous::Args& send_args,
                           DataWorkerRendezvous::DoneCallback done) override;
  Status LocalDataWorkerSend(const DataWorkerRendezvous::ParsedKey& key,
                             const string& tensor_name,
                             const Tensor& val,
                             const DataWorkerRendezvous::Args& send_args) override;
  void RecvLocalAsync(const DataWorkerRendezvous::ParsedKey& key, DataWorkerRendezvous::DoneCallback done) override;
  void FuseRecvLocalAsync(const std::vector<DataWorkerRendezvous::ParsedKey>& keys,
                          DataWorkerRendezvous::FuseDoneCallback done) override;
  void DataWorkerRecvAsync(const DataWorkerRendezvous::ParsedKey& key,
                           const DataWorkerRendezvous::Args& recv_args,
                           DataWorkerRendezvous::DoneCallback done) override;
  void DataWorkerFuseRecvAsync(const DataWorkerRendezvous::Args& recv_args,
                               DataWorkerRendezvous::FuseDoneCallback done) override;
  void RegisterDataWorker(const string& task_name, const string& host_port);

 private:
  void RecvAsync(const DataWorkerRendezvous::ParsedKey& key,
                 const DataWorkerRendezvous::Args& recv_args,
                 DataWorkerRendezvous::DoneCallback done);
  void EnqueueRecvItems(std::vector<Item*>& items);
  void EnqueueFuseRecvItem(FuseItem* item);
  void SameWorkerRecvDone(const DataWorkerRendezvous::ParsedKey& parsed,
                          const DataWorkerRendezvous::Args& send_args,
                          const DataWorkerRendezvous::Args& recv_args,
                          const Tensor& in, Tensor* out, StatusCallback done);

  static uint64 KeyHash(const StringPiece& k) {
    return Hash64(k.data(), k.size());
  }
  
  const string protocol_;
  const bool fuse_recv_;
  mutex attrs_mu_;
  std::unordered_map<uint64, std::pair<AllocatorAttributes, Device*>> recv_nodes_attrs_ GUARDED_BY(attrs_mu_);
  const int num_recv_threads_;
  std::vector<std::unique_ptr<DataWorkerRecvTensorThread>> recv_threads_;
  std::unique_ptr<thread::ThreadPool> send_threads_;

  typedef std::deque<Item*> ItemQueue;
  typedef std::deque<FuseItem*> FuseItemQueue;
  typedef gtl::FlatMap<uint64, ItemQueue> Table;

  std::mutex mu_;
  std::mutex local_tmp_mu_;
  std::condition_variable cv_; 
  Status status_ GUARDED_BY(mu_);
  WorkerSession* session_ GUARDED_BY(mu_);
  
  // Table is used for both data workers and training workers for storing the items to enable async execution:
  // Data workers put the produced tensors in the Table and wait for the training workers
  // to fetch them. Training workers put the fetched tensors in their local Table.
  Table table_ GUARDED_BY(mu_);
  FuseItemQueue fuse_queue_ GUARDED_BY(mu_);
  std::vector<Item*> local_tmp_ GUARDED_BY(local_tmp_mu_);
  std::vector<string> tensor_names_;
  const int queue_size_;

  friend class DataWorkerRecvTensorThread;
  friend class GrpcDataWorkerRecvTensorThread;
  friend class StarDataWorkerRecvTensorThread;
  TF_DISALLOW_COPY_AND_ASSIGN(GenericDataWorkerRendezvous);
};


}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_RENDEZVOUS_MGR_H_