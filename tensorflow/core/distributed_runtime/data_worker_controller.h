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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_CONTROLLER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DATA_WORKER_CONTROLLER_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/data_worker_graph_partition.h"

namespace tensorflow {

// Maintains and dispatches subgraphs to data workers.
class DataWorkerController {
private:
  // A data-processing graph partitioned from each *training*
  // worker task to be dispatched to data workers.
  struct TaskDataWorkerGraph {
    string task_name;
    std::vector<std::pair<string /* dw name */, string /* dw host_port */>>
      registered_data_workers;
    std::shared_ptr<GraphDef> g;
    // Names of the DataWorkerSend ops that should be run
    // on the data worker clients.
    std::vector<string> node_names;
    // Names of the tensors to be sent from data workers.
    std::vector<string> tensor_names;
    int num_registered() const { return registered_data_workers.size(); }
    void RegisterDataWorker(const string& name, const string& host_port) {
      registered_data_workers.emplace_back(name, host_port);
    }

    TaskDataWorkerGraph(const string& task_name,
                        std::shared_ptr<GraphDef> g,
                        const std::vector<string>& node_names,
                        const std::vector<string>& tensor_names)
      : task_name(task_name), g(g), node_names(node_names), tensor_names(tensor_names) {}
    ~TaskDataWorkerGraph() {}
  };
  
  mutex mu_;
  std::vector<TaskDataWorkerGraph> graphs_ GUARDED_BY(mu_);
  bool use_default_split_points_ = true;
  bool extend_default_split_ = false;
  bool fuse_recv_ = false;
  // Used for sequencing DataWorkerSend/Recv nodes.
  int64 next_node_id_ GUARDED_BY(mu_) = 0;
  
  // Returns the graph that has been allocated to the least number of data workers.
  TaskDataWorkerGraph& GetGraphForNewDataWorker();
  // Resets the device names to the target data worker.
  void ResetDeviceNamesForGraph(GraphDef* const g, const string& dw_name);
  void ResetDeviceNameForNode(NodeDef* node, const string& dw_name);

public:
  DataWorkerController() {}
  DataWorkerController(bool use_default_split_points, bool extend_default_split, bool fuse_recv);
  ~DataWorkerController() {}
  Status Partition(Graph* g, PartitionForDataWorkerOptions& popts);
  Status RegisterDataWorker(GraphDef* dst_graph,
                            const string& name,
                            const string& host_port,
                            string& training_worker_name,
                            std::vector<string>& node_names);
  const std::vector<string>* GetTensorNames(const string& task_name);
};

} // namespace tensorflow


#endif
