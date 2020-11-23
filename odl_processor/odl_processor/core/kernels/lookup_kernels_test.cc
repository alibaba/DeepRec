/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "gtest/gtest.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "odl_processor/core/util/utils.h"
#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/core/util/utils.h"

namespace tensorflow {
namespace processor {


TEST(KernelsTest, KvImportTest) {
// TODO: can NOT expose OSSkey and OSSid in the test.
// Manual test OSS and local file passed.
#if 0
  NodeDef version_def;
  Tensor version_value(DT_STRING, TensorShape({1}));
  version_value.flat<std::string>()(0) = "v1";
  TF_CHECK_OK(NodeDefBuilder("version", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", version_value)
                  .Finalize(&version_def));

  NodeDef prefix_def;
  Tensor prefix_value(DT_STRING, TensorShape({1}));
  prefix_value.flat<std::string>()(0) =
      // Local test
      //"/local/path/workspace/tmp/DeepFM/ev/1598442950_pai/variables/variables";
      // OSS test
      "oss://bucket-name\x01id=id\x02key=key\x02host=host/jktest/mm/odl_test_files/saved_model/variables";
  TF_CHECK_OK(NodeDefBuilder("prefix", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", prefix_value)
                  .Finalize(&prefix_def));

  NodeDef tensor_name_def;
  Tensor tensor_name_value(DT_STRING, TensorShape({1}));
  tensor_name_value.flat<std::string>()(0) =
      "input_from_feature_columns/fm_10169_embedding/weights";
  TF_CHECK_OK(NodeDefBuilder("tensor_name", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", tensor_name_value)
                  .Finalize(&tensor_name_def));

  NodeDef kv_import_def;
  TF_CHECK_OK(NodeDefBuilder("kv_import", "KvImport")
                  .Input("version", 0, DT_STRING)
                  .Input("prefix", 0, DT_STRING)
                  .Input("tensor_name", 0, DT_STRING)
                  .Attr("var_name", "XXX")
                  .Attr("dim_len", 1)
                  .Attr("Tkeys", DT_INT64)
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&kv_import_def));

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));

  Status status;
  std::unique_ptr<OpKernel> kv_import_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     kv_import_def, TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  // Create inputs
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back({nullptr, &version_value});
  inputs.push_back({nullptr, &prefix_value});
  inputs.push_back({nullptr, &tensor_name_value});
  params.inputs = &inputs;
  params.op_kernel = kv_import_op.get();

  std::unique_ptr<OpKernelContext> kv_import_context(
      new OpKernelContext(&params, 0));

  auto done = []() {
    LOG(INFO) << "I'm done.";
  };

  AsyncOpKernel* real_kv_import_op = (AsyncOpKernel*)(kv_import_op.get());
  real_kv_import_op->ComputeAsync(kv_import_context.get(), std::move(done));
  TF_CHECK_OK(kv_import_context->status());
#endif

  EXPECT_TRUE(1);
}


} // namespace processor
} // namespace tensorflow
