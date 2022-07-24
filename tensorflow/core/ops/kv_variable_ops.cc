// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

namespace {

Status ValidateVariableResourceHandle(InferenceContext* c,
                                      ShapeAndType* shape_and_type) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    shape_and_type->shape = c->UnknownShape();
    shape_and_type->dtype = DT_INVALID;
  } else {
    *shape_and_type = (*handle_data)[0];
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (shape_and_type->dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->dtype), " got ",
          DataTypeString(value_dtype));
    }
  }
  return Status::OK();
}

Status ReadVariableShapeFn(InferenceContext* c) {
  ShapeAndType shape_and_type;
  TF_RETURN_IF_ERROR(ValidateVariableResourceHandle(c, &shape_and_type));
  c->set_output(0, shape_and_type.shape);
  return Status::OK();
}

Status CreateAssignShapeFn(InferenceContext* c) {
  ShapeAndType handle_shape_and_type;
  TF_RETURN_IF_ERROR(ValidateVariableResourceHandle(c, &handle_shape_and_type));

  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(handle_shape_and_type.shape, value_shape, &unused));
  return Status::OK();
}

}  // namespace

// KvVar
REGISTER_OP("KvVarHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("Tkeys: {int64,int32,string}")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0,
                                            std::vector<ShapeAndType>{{s, t}});

      return Status::OK();
    })
    .Doc(R"(
Creates a handle to a Variable resource.

container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
dtype: the type of this variable. Must agree with the dtypes
  of all ops using this variable.
shape: The (possibly partially specified) shape of this variable.
)");

REGISTER_OP("ReadKvVariableOp")
    .Input("resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64,int32,string}")
    .SetShapeFn(ReadVariableShapeFn)
    .Doc(R"(
Reads the value of a variable.

The tensor returned by this operation is immutable.

The value returned by this operation is guaranteed to be influenced by all the
writes on which this operation depends directly or indirectly, and to not be
influenced by any of the writes which depend directly or indirectly on this
operation.

resource: handle to the resource in which to store the variable.
dtype: the dtype of the value.
)");

REGISTER_OP("InitializeKvVariableOp")
    .Input("resource_self: resource")
    .Input("resource_primary: resource")
    .Input("value: dtype")
    .Input("empty_key: Tkeys")
    .Attr("slot_num: int = 0")
    .Attr("Tkeys: {int64,int32,string}")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")	
    .Attr("steps_to_live: int = 0")
    .Attr("ht_type: string = ''")
    .Attr("emb_index: int = 0")
    .Attr("block_num: int = 1")
    .Attr("slot_index: int = 0")
    .Attr("ht_partition_num: int = 1000")
    .Attr("filter_freq: int = 0")
    .Attr("max_freq: int = 999999")
    .Attr("max_element_size: int  = 0")
    .Attr("counter_type: type")
    .Attr("false_positive_probability: float = -1.0")
    .Attr("l2_weight_threshold: float =-1.0")
    .Attr("layout: string = 'normal'")
    .Attr("storage_type: int = 1")
    .Attr("storage_path: string = '.'")
    .Attr("storage_size: list(int) = []")
    .Attr("default_value_dim: int = 4096")
    .Attr("record_freq: bool = false")
    .Attr("record_version: bool = false")
    .SetShapeFn([](InferenceContext* c) { 
      return Status::OK();
    })
    .Doc(R"(
Assigns a new value to a variable.

Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.

resource_self: handle to the resource in which to store the variable.
resource_primary: handle to the resource in which to store the variable.
value: the value to set the new tensor to use.
dtype: the dtype of the value.
)");

REGISTER_OP("KvVarIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .Attr("Tkeys: {int64,int32,string}")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a resource handle-based variable has been initialized.

resource: the input resource handle.
is_initialized: a scalar boolean which is true if the variable has been
initialized.
)doc");

Status KvVariableShapeShapeFn(InferenceContext* c) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    return errors::InvalidArgument("Handle doesn't have shape information.");
  }
  c->set_output(0, (*handle_data)[0].shape);
  return Status::OK();
}

REGISTER_OP("KvVariableShape")
    .Input("input: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .Attr("Tkeys: {int64,int32,string}")
    .SetShapeFn(KvVariableShapeShapeFn)
    .Doc(R"doc(
Returns the shape of the variable pointed to by `resource`.

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

)doc");

REGISTER_OP("DestroyKvResourceOp")
    .Input("resource: resource")
    .Attr("ignore_lookup_error: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"(
Deletes the resource specified by the handle.

All subsequent operations using the resource will result in a NotFound
error status.

resource: handle to the resource to delete.
ignore_lookup_error: whether to ignore the error when the resource
  doesn't exist.
)");

REGISTER_OP("KvResourceGatherV1")
    .Input("resource: resource")
    .Input("indices: Tkeys")
    .Input("default_value: dtype")
    .Input("counts: counts_type")
    .Attr("validate_indices: bool = true")
    .Attr("is_use_default_value_tensor: bool = false")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64,int32,string}")
    .Attr("counts_type: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(handle_shape_and_type.shape, 1, &unused));
      ShapeHandle params_subshape;
	  params_subshape = handle_shape_and_type.shape;
      //TF_RETURN_IF_ERROR(
      //    c->Subshape(handle_shape_and_type.shape, 1, &params_subshape));
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Gather slices from the variable pointed to by `resource` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

)doc");

REGISTER_OP("KvResourceGather")
    .Input("resource: resource")
    .Input("indices: Tkeys")
    .Input("default_value: dtype")
    .Attr("is_use_default_value_tensor: bool = false")
    .Attr("validate_indices: bool = true")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64,int32,string}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(handle_shape_and_type.shape, 1, &unused));
      ShapeHandle params_subshape;
	  params_subshape = handle_shape_and_type.shape;
      //TF_RETURN_IF_ERROR(
      //    c->Subshape(handle_shape_and_type.shape, 1, &params_subshape));
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Gather slices from the variable pointed to by `resource` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

)doc");

REGISTER_OP("KvResourceScatterAdd")
    .Input("resource: resource")
    .Input("indices: Tkeys")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tkeys: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      //TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
	  var_subshape = var_shape;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
	  // TODO
      //TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    })
    .Doc(R"doc(
Adds sparse updates to the variable referenced by `resource`.

This operation computes

    # Scalar indices
    ref[indices, ...] += updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
</div>

resource: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of updated values to add to `ref`.
)doc");

REGISTER_OP("KvResourceImport")
    .Input("resource_handle: resource")
    .Input("value: dtype")
    .Input("empty_key: Tkeys")
    .Input("keys: Tkeys")
    .Input("values: dtype")
    .Input("versions: int64")
    .Attr("shape: shape")
    .Attr("Tkeys: {int64,int32,string}")
    .Attr("dtype: type")
    .Attr("steps_to_live: int = 0")
    .Attr("ht_type: string = ''")
    .Attr("ht_partition_num: int = 1000")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // TODO(dingchen): Validate keys and values shape.
      return Status::OK();
    })
    .Doc(R"doc(
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

resource_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("KvResourceImportV2")
    .Input("prefix: string")
    .Input("resource_self: resource")
    .Input("resource_primary: resource")
    .Input("value: dtype")
    .Input("tensor_names: string")
    .Input("empty_key: Tkeys")
    .Attr("slot_num: int = 0")
    .Attr("shape: shape")
    .Attr("Tkeys: {int64,int32}")
    .Attr("dtype: type")
    .Attr("emb_index: int = 0")
    .Attr("slot_index: int = 0")
    .Attr("block_num: int = 1")
    .Attr("steps_to_live: int = 0")
    .Attr("partition_id: int = 0")
    .Attr("partition_num: int = 1")
    .Attr("ht_type: string = ''")
    .Attr("filter_freq: int = 0")
    .Attr("ht_partition_num: int = 1000")
    .Attr("max_element_size: int  = 0")
    .Attr("counter_type: type")
    .Attr("false_positive_probability: float = -1.0")
    .Attr("l2_weight_threshold: float =-1.0")
    .Attr("layout: string = 'normal'")
    .Attr("max_freq: int = 999999")
    .Attr("storage_type: int = 1")
    .Attr("storage_path: string = '.'")
    .Attr("storage_size: list(int) = []")
    .Attr("default_value_dim: int = 4096")
    .Attr("record_freq: bool = false")
    .Attr("record_version: bool = false")
    .SetShapeFn([](InferenceContext* c) {
          ShapeHandle handle;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
          return Status::OK();
     })
    .Doc(R"doc()doc");

REGISTER_OP("KvResourceIncrImport")
    .Input("prefix: string")
    .Input("resource_handle: resource")
    .Input("tensor_names: string")
    .Input("empty_key: Tkeys")
    .Input("value: dtype")
    .Attr("Tkeys: {int64,int32}")
    .Attr("dtype: type")
    .Attr("partition_id: int = 0")
    .Attr("partition_num: int = 1")
    .SetShapeFn([](InferenceContext* c) {
          ShapeHandle handle;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
          return Status::OK();
     })
    .Doc(R"doc()doc");

REGISTER_OP("KvResourceExport")
    .Input("resource_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Output("versions: int64")
    .Output("freqs: int64")
    .Attr("Tkeys: {int64,int32,string}")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 2, &values));
      ShapeHandle keys = c->UnknownShapeOfRank(1);
      ShapeHandle versions = c->UnknownShapeOfRank(1);
      ShapeHandle freqs = c->UnknownShapeOfRank(1);
      c->set_output(0, keys);
      c->set_output(1, values);
      c->set_output(2, versions);
      c->set_output(3, freqs);
      return Status::OK();
    })
    .Doc(R"doc(
Outputs all keys and values in the kv resource.

resource_handle: Handle to the kvResource.
keys: Vector of all keys present in the table.
values: Tensor of all values in the table. Indexed in parallel with `keys`.
versions: Vector of all versions present in the table.
freqs: Vector of all freqs present in the table.
)doc");

}  // namespace tensorflow
