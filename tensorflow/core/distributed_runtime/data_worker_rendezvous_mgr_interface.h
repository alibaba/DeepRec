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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_RENDEZVOUS_MGR_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_RENDEZVOUS_MGR_INTERFACE_H_

#include <string>
#include <vector>

#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/data_worker_rendezvous.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class DataWorkerRendezvousMgrInterface {
 public:
  DataWorkerRendezvousMgrInterface() {}
  virtual ~DataWorkerRendezvousMgrInterface() {}

  virtual DataWorkerRendezvous* Find() = 0;

  virtual void RecvLocalAsync(const DataWorkerRendezvous::ParsedKey& key,
                              DataWorkerRendezvous::DoneCallback done) = 0;

  virtual void FuseRecvLocalAsync(const std::vector<DataWorkerRendezvous::ParsedKey>& keys,
                                  DataWorkerRendezvous::FuseDoneCallback done) = 0;

  virtual void RegisterDataWorker(const string& task_name, const string& host_port) = 0;

  virtual void SetTensorNames(const std::vector<string>& tensor_names) = 0;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_RENDEZVOUS_MGR_INTERFACE_H_
