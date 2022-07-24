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

#ifndef TENSORFLOW_CORE_GRAPH_DATA_WORKER_GRAPH_PARTITION_H_
#define TENSORFLOW_CORE_GRAPH_DATA_WORKER_GRAPH_PARTITION_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"


namespace tensorflow {

struct PartitionForDataWorkerOptions {
  // A function that returns a unique graph node name with the given
  // prefix.
  typedef std::function<string(const string&)> NewNameFunc;
  NewNameFunc new_name = nullptr;

  // If specified, flib_def defines a function library that should be
  // partitioned and replicated into each resulting partition graphs.
  const FunctionLibraryDefinition* flib_def = nullptr;

  std::unordered_set<string> targets;
  bool use_default_split_points = true;
  bool extend_default_split = false;
  bool fuse_recv = false;
};

Status PartitionForDataWorker(
  const PartitionForDataWorkerOptions& opts,
  Graph* g,
  std::unordered_map<string, std::shared_ptr<GraphDef>>& data_worker_graphs,
  std::unordered_map<string, std::vector<string>>& node_names,
  std::unordered_map<string, std::vector<string>>& tensor_names);

} // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPH_DATA_WORKER_GRAPH_PARTITION_H_