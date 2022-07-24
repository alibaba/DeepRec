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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("_DataWorkerSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from data worker to training worker.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
)doc");

REGISTER_OP("_LocalDataWorkerSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from the local data worker 
(the data worker subgraph that still resides
in the training worker) to training worker.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
)doc");

REGISTER_OP("_DataWorkerRecv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from data worker.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
recv_device: The name of the device receiving the tensor.
)doc");

REGISTER_OP("_HostDataWorkerSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from data worker to training worker.

_HostDataWorkerSend requires its input on host memory whereas _DataWorkerSend requires its
input on device memory.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
)doc");

REGISTER_OP("_HostLocalDataWorkerSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from the local data worker 
(the data worker subgraph that still resides
in the training worker) to training worker.

_HostLocalDataWorkerSend requires its input on host memory whereas _LocalDataWorkerSend requires its
input on device memory.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
)doc");

REGISTER_OP("_HostDataWorkerRecv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from data worker.

_HostDataWorkerRecv produces its output on host memory whereas _DataWorkerRecv produces its
output on device memory.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
recv_device: The name of the device receiving the tensor.
)doc");

REGISTER_OP("_DataWorkerFuseRecv")
    .Output("tensors: tensor_types")
    .Attr("tensor_types: list(type)")
    .Attr("tensor_names: list(string)")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("_HostDataWorkerFuseRecv")
    .Output("tensors: tensor_types")
    .Attr("tensor_types: list(type)")
    .Attr("tensor_names: list(string)")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // end namespace tensorflow
