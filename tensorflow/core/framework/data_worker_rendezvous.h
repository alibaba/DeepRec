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

#ifndef TENSORFLOW_FRAMEWORK_DATA_WORKER_RENDEZVOUS_H_
#define TENSORFLOW_FRAMEWORK_DATA_WORKER_RENDEZVOUS_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <deque>

#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
struct WorkerSession;

class DataWorkerRendezvous : public core::RefCounted {
 public:
  explicit DataWorkerRendezvous() {}
  struct Args {
    AllocatorAttributes alloc_attrs;
    DeviceContext* device_context = nullptr;
    CancellationManager* cancellation_manager = nullptr;  // not owned.
    bool local = false;
    string device_type;
    string device;
  };

  struct DataWorkerInfo {
    string task_name;
    string host_port;
    string tensor_name;
    DataWorkerInfo() {}
    DataWorkerInfo(string task_name, string host_port, string tensor_name)
     : task_name(task_name), host_port(host_port), tensor_name(tensor_name) {}
  };

  virtual ~DataWorkerRendezvous() {}

  // Constructs a data worker rendezvous key for the tensor of "name".
  static string CreateKey(const string& name);

  // Parses the key constructed by CreateKey and parse src/dst device
  // names into structures respectively.
  // (TODO) Reserved for future design of multi-edge data worker send/recv op
  struct ParsedKey {
    StringPiece tensor_name;
    ParsedKey() {}
    ParsedKey(const ParsedKey& b) { *this = b; }

    ParsedKey& operator=(const ParsedKey& b);
    StringPiece FullKey() const { return buf_; }

   private:
    friend class DataWorkerRendezvous;
    friend class BaseDataWorkerSendOp;
    friend class DataWorkerSendOp;
    friend class LocalDataWorkerSendOp;
    friend class DataWorkerRecvOp;
    friend class DataWorkerFuseRecvOp;
    string buf_;
  };
  static Status ParseKey(StringPiece key, ParsedKey* out);

  // Synchronous wrapper for RecvAsync.
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead, int64 timeout_ms);
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead);

  virtual Status Initialize(WorkerSession* session) = 0;
  virtual void StartAbort(const Status& status) = 0;
  virtual void SetTensorNames(const std::vector<string>& tensor_names) = 0;
  // DataWorkerRendezvous receives tensors asynchronously and cannot 
  // rely on the RecvOp to set the attrs. We therefore expose this
  // interface to set the attrs in advance.
  virtual Status SetRecvAttrs(const ParsedKey& key, const AllocatorAttributes& alloc_attrs, const string& device) = 0;
  
  typedef std::function<void(const Status&, const Args&, const Args&, const Tensor&, const bool)>
    DoneCallback;
  typedef std::function<void(const Status&,
                             const std::vector<DataWorkerRendezvous::Args>&,
                             const Args&,
                             const std::vector<Tensor>&)>
                             // is_dead omitted
    FuseDoneCallback;
  struct Item {
    DoneCallback waiter = nullptr;
    DataWorkerInfo data_worker_info;
    Tensor value;
    bool is_dead = false;
    ParsedKey key;
    Args send_args;
    Args recv_args;
    CancellationToken cancellation_token;
    // Returns true iff this item represents a value being sent.
    bool IsSendValue() const { return this->waiter == nullptr; }

    ~Item() {
      if (send_args.device_context) {
        send_args.device_context->Unref();
      }
      if (recv_args.device_context) {
        recv_args.device_context->Unref();
      }
    }
  };
  struct FuseItem {
    FuseDoneCallback waiter = nullptr;
    DataWorkerInfo data_worker_info;
    std::vector<Tensor> values;
    std::vector<ParsedKey> keys;
    std::vector<Args> send_args;
    Args recv_args;
    CancellationToken cancellation_token;
    // Returns true iff this item represents a value being sent.
    bool IsSendValue() const { return this->waiter == nullptr; }

    ~FuseItem() {
      for (Args args : send_args) {
        if (args.device_context) {
          args.device_context->Unref();
        }
      }
      if (recv_args.device_context) {
        recv_args.device_context->Unref();
      }
    }
  };
  
  // We make the send operation an async method because it may block due to the queue size limit.
  virtual void DataWorkerSendAsync(const ParsedKey& key,
                                   const Tensor& val,
                                   const Args& send_args,
                                   DoneCallback done) = 0;
  virtual Status LocalDataWorkerSend(const ParsedKey& key,
                                     const string& tensor_name,
                                     const Tensor& val,
                                     const Args& send_args) = 0;
  virtual void RecvLocalAsync(const ParsedKey& key, DoneCallback done) = 0;
  virtual void FuseRecvLocalAsync(const std::vector<ParsedKey>& keys, FuseDoneCallback done) = 0;
  virtual void DataWorkerRecvAsync(const ParsedKey& key, const Args& args, DoneCallback done) = 0;
  virtual void DataWorkerFuseRecvAsync(const Args& recv_args,
                                       FuseDoneCallback done) = 0;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_DATA_WORKER_RENDEZVOUS_H_