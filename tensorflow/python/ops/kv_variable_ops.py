# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ops to use variables as resources."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat

__all__ = ["EmbeddingVariable"]



class EmbeddingVariable(resource_variable_ops.ResourceVariable):
  """Embedding Variable based on resource variable.

  See the ${variables} documentation for more details.

  A `EmbeddingVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `EmbeddingVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the parted variable. After construction, the type and embedding
  dim shape of the variable are fixed. The first demension of the embedding variable
  is mutable. The shape can be changed using read_sparse methods.

  Unlike tf.ResourceVariable, a tf.EmbeddingVariable is mutable. the shape of the
  EmbeddingVariable means the embedding dim, user can use the APIs(sparse_read()) to
  change the whole shape of the EmbeddingVariable. When read_sparse(index=i, ...) is
  called, if the i-th embedding value doesn't exist, it will be initialized and return,
   else it will return the i-th existing embedding value, when the embedding variable
  is updated by back propagation, the i-th embedding value will be updated or removed.

  For example:

   ```python
    a = tf.EmbeddingVariable([1.0, 3.0, 5.0])
    a.initializer.run()

    b = a.sparse_read([2])

    tf.Print(b, [b]).run()  # Will print 1.0, 3.0, 5.0
  ```

  """

  def __init__(self,
               initial_value=None,
               initializer=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               dtype=None,
               variable_def=None,
               import_scope=None,
               constraint=None,
               invalid_key=None,
               evconfig=variables.EmbeddingVariableConfig(),
               ht_partition_num=1000):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: Ignored. Provided for compatibility with tf.Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
        a Tensor) or float32 will be used (if it is a Python object convertible
        to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `EmbeddingVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        EmbeddingVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is None, which signifies that this Variable will not be added to any
    collections.
    @end_compatibility
    """
    if context.executing_eagerly():
      raise ValueError("Creating EmbeddingVariable"
                       " only supported in GRAPH mode.")
    if variable_def:
      if initial_value is not None:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      evconfig.reveal()
      self._init_from_args(
          initial_value=initial_value,
          initializer=initializer,
          trainable=trainable,
          collections=collections,
          validate_shape=validate_shape,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          constraint=constraint,
          invalid_key=invalid_key,
          evconfig=evconfig,
          ht_partition_num=ht_partition_num)

  def __repr__(self):
    return "<tf.EmbeddingVariable '%s' embedding dim=%s dtype=%s>" % (self.name,
                                                                      self.shape,
                                                                      self.dtype.name)

  # LINT.IfChange
  # _VariableFromResource inherits from EmbeddingVariable but
  # doesn't call the constructor, so changes here might need to be reflected
  # there.
  # pylint: disable=unused-argument
  def _init_from_args(self,
                      initial_value=None,
                      initializer=None,
                      trainable=True,
                      collections=None,
                      validate_shape=True,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      constraint=None,
                      invalid_key=-1,
                      evconfig=variables.EmbeddingVariableConfig(),
                      ht_partition_num=1000):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: Ignored. Provided for compatibility with tf.Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
       a Tensor) or float32 will be used (if it is a Python object convertible
       to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the GLOBAL_VARIABLES or TRAINABLE_VARIABLES
    collections, and the `collections` argument is ignored.
    @end_compatibility
    """
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to Variable constructor must be a list, tuple, "
          "or set. Got %s of type %s" % (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    self._trainable = trainable
    self._initializer = initializer
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    if ops.GraphKeys.EMBEDDING_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.EMBEDDING_VARIABLES]
    self._save_slice_info = None
    self._in_graph_mode = not context.executing_eagerly()
    self._steps_to_live = evconfig.steps_to_live
    self._init_data_source = evconfig.init_data_source 
    self._emb_index = evconfig.emb_index
    self._slot_index = evconfig.slot_index
    self._block_num = evconfig.block_num
    self._block_handle_name = None
    self._primary = evconfig.primary
    self._ht_type = evconfig.ht_type
    self._ht_partition_num = ht_partition_num
    self._is_sparse=False
    self.importer=None
    if evconfig.filter_strategy != None:
      if isinstance(evconfig.filter_strategy, variables.CounterFilter):
        self._filter_freq = evconfig.filter_strategy.filter_freq
        self._max_element_size = 0
        self._false_positive_probability = -1.0
        self._counter_type = dtypes.uint64
      elif isinstance(evconfig.filter_strategy, variables.CBFFilter):
        self._filter_freq = evconfig.filter_strategy.filter_freq
        self._max_element_size = evconfig.filter_strategy.max_element_size
        self._false_positive_probability = evconfig.filter_strategy.false_positive_probability
        self._counter_type = evconfig.filter_strategy.counter_type
    else:
      self._filter_freq = 0
      self._max_element_size = 0
      self._false_positive_probability = -1.0
      self._counter_type = dtypes.uint64

    multi_level_list = [config_pb2.StorageType.LEVELDB, config_pb2.StorageType.SSDHASH,
                        config_pb2.StorageType.DRAM_PMEM, config_pb2.StorageType.DRAM_LEVELDB,
                        config_pb2.StorageType.DRAM_SSDHASH, config_pb2.StorageType.HBM_DRAM,
                        config_pb2.StorageType.DRAM_PMEM_SSDHASH, config_pb2.StorageType.HBM_DRAM_SSDHASH]
      
    self._record_freq = (os.environ.get("TF_RECORD_FREQ", "0") == "1")
    self._record_version = (os.environ.get("TF_RECORD_VERSION", "0") == "1")
    self._l2_weight_threshold = evconfig.l2_weight_threshold
    self._storage_type = evconfig.storage_type
    self._storage_path = evconfig.storage_path
    self._storage_size = evconfig.storage_size
    self._default_value_dim = evconfig.default_value_dim
    if (isinstance(evconfig.filter_strategy, variables.CounterFilter)  and self._filter_freq != 0) or \
       self._steps_to_live not in [0, None] or self._record_version or \
       self._storage_type in multi_level_list or self._record_freq:
      if self._block_num not in [1, None] and self._storage_type in multi_level_list:
        raise ValueError("Dynamic-dimension Embedding and Multi-level EV can't be enabled together") 
      if self._block_num not in [1, None] or \
          (self._filter_freq != 0 and self._storage_type not in multi_level_list):
        self._layout = "normal"
      else:
        self._layout = "normal_contiguous"
    else:
      self._layout = "light"

    if self._primary is None:
      self._is_primary = True
    else:
      self._is_primary = False
    with ops.control_dependencies(None):
      with ops.name_scope(name, "Variable", []
                          if init_from_fn else [initial_value]) as name:
        # pylint: disable=protected-access
        self._invalid_key = invalid_key
        self._invalid_key_type = ops.convert_to_tensor(invalid_key, name="invalid_key").dtype.base_dtype
        handle_name = ops.name_from_scope_name(name)
        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          attr = attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(
                  s=[compat.as_bytes("loc:@%s" % handle_name)]))
          with ops.get_default_graph()._attr_scope({"_class": attr}):
            with ops.name_scope("Initializer"):
              initial_value = ops.convert_to_tensor(
                  initial_value(), name="initial_value", dtype=dtype)
            rank = initial_value.get_shape().rank - 1

            self._handle = self._embedding_variable_handle(
                shape=initial_value.get_shape()[rank:],
                dtype=initial_value.dtype.base_dtype,
                shared_name=handle_name,
                name=name,
                graph_mode=self._in_graph_mode)
            if self._primary is None:
              self._primary = self
            self._primary_handle = self._primary._handle
            self._handle_device = (
                self._handle.device if self._in_graph_mode else
                context.get_default_context().device_name)
            self._graph_shape = initial_value.get_shape()[rank:]
        # pylint: enable=protected-access

        # Or get the initial value from a Tensor or Python object.
        else:
          with ops.name_scope("Initializer"):
            initial_value = ops.convert_to_tensor(
                initial_value, name="initial_value", dtype=dtype)
          rank = 0
          self._default_value_dim = 1
          # pylint: disable=protected-access
          if (self._in_graph_mode and initial_value is not None and
              initial_value.op._get_control_flow_context() is not None):
            raise ValueError(
                "Initializer for variable %s is from inside a control-flow "
                "construct, such as a loop or conditional. When creating a "
                "variable inside a loop or conditional, use a lambda as the "
                "initializer." % name)
          # pylint: enable=protected-access
          self._handle = self._embedding_variable_handle(
              shape=initial_value.get_shape(),
              dtype=initial_value.dtype.base_dtype,
              shared_name=handle_name,
              name=name,
              graph_mode=self._in_graph_mode)
          if self._primary is None:
            self._primary = self
          self._primary_handle = self._primary._handle
          self._handle_device = (self._handle.device if self._in_graph_mode else
                                 context.get_default_context().device_name)
          self._graph_shape = initial_value.get_shape()

        self._initial_value = initial_value if self._in_graph_mode else None
        self._handle_name = handle_name + ":0"
        self._dtype = initial_value.dtype.base_dtype
        self._constraint = constraint
        if self._is_primary:
          self._slot_num = 0 
        else:
          self._slot_num = evconfig.slot_num

        with ops.name_scope("IsInitialized"):
          self._is_initialized_op = (
              gen_kv_variable_ops.kv_var_is_initialized_op(self._handle,
                                                           Tkeys=self._invalid_key_type))
        if initial_value is not None:
          with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
            with ops.control_dependencies(None if self._is_primary else [self._primary.initializer]):
              self._initializer_op = (
                gen_kv_variable_ops.initialize_kv_variable_op(
                    self._handle,
                    self._primary._handle,
                    variables._try_guard_against_uninitialized_dependencies(name, initial_value),
                    ops.convert_to_tensor(invalid_key),
                    slot_num = self._slot_num,
                    shape=initial_value.get_shape()[rank:],
                    steps_to_live=self._steps_to_live,
                    emb_index=self._emb_index, block_num=self.block_num,
                    slot_index=self._slot_index,
                    ht_type=self._ht_type,
                    ht_partition_num=self._ht_partition_num,
                    filter_freq = self._filter_freq,
                    l2_weight_threshold = self._l2_weight_threshold,
                    max_element_size = self._max_element_size,
                    false_positive_probability = self._false_positive_probability,
                    counter_type = self._counter_type,
                    max_freq = 99999,
                    layout = self._layout,
                    storage_type = self._storage_type,
                    storage_path = self._storage_path,
                    storage_size = self._storage_size,
                    default_value_dim = self._default_value_dim,
                    record_freq = self._record_freq,
                    record_version = self._record_version,
                    name=n))
        self._graph_element = self._handle
        self._cached_value = None
        if not context.executing_eagerly():
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)

  def export(self):
    return gen_kv_variable_ops.kv_resource_export(self._handle, Tkeys=self._invalid_key_type)

  def _init_from_proto(self, variable_def, import_scope=None):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as EmbeddingVariable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.variable_name, import_scope=import_scope))
    self._graph_shape = tensor_shape.TensorShape(
        self._handle.op.get_attr("shape"))
    self._handle_device = self._handle.device
    self._handle_name = self._handle.name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.initializer_name, import_scope=import_scope))
    self._trainable = getattr(variable_def, "trainable", True)
    if variable_def.snapshot_name:
      self._cached_value = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.snapshot_name, import_scope=import_scope))
    else:
      self._cached_value = None
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._slot_index = self._initializer_op.get_attr("slot_index")
    self._block_num = self._initializer_op.get_attr("block_num")
    primary_name = ""
    primary_name_list = variable_def.variable_name.split("/")
    if self._block_num > 1:
      if self._slot_index == 0:
        if primary_name_list[-1] != "block0":
          for val in primary_name_list[:-1]:
            primary_name += val + "/"
          primary_name = primary_name + "block0:0"
          self._primary_handle = g.as_graph_element(
            ops.prepend_name_scope(
                primary_name, import_scope=import_scope))
        else:
          self._primary_handle = self._handle
      else:
        for val in primary_name_list[:-2]:
          primary_name += val + "/"
        primary_name = primary_name + "block0:0"
        self._primary_handle = g.as_graph_element(
            ops.prepend_name_scope(
                primary_name, import_scope=import_scope))
    else:
      if self._slot_index is 0:
        self._primary_handle = self._handle
      else:
        for val in primary_name_list[:-1]:
          primary_name += val + "/"
        primary_name = primary_name[:-1] + ":0"
        self._primary_handle = g.as_graph_element(
            ops.prepend_name_scope(
                primary_name, import_scope=import_scope))
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._invalid_key = -1
    self._steps_to_live = self._initializer_op.get_attr("steps_to_live")
    self._ht_type = self._initializer_op.get_attr("ht_type")
    self._ht_partition_num = self._initializer_op.get_attr("ht_partition_num")
    self._init_data_source = None
    self._initial_value = ops.convert_to_tensor(
                              [0], name="initial_value", dtype=self._dtype)
    self._invalid_key_type = dtypes.as_dtype(self._handle.op.get_attr("Tkeys"))
    self._graph_element = self._handle
    self._constraint = None
    self._is_sparse=False
    self._layout = self._initializer_op.get_attr("layout")
    self._slot_num = self._initializer_op.get_attr("slot_num")
    self._emb_index = self._initializer_op.get_attr("emb_index")
    self._filter_freq = self._initializer_op.get_attr("filter_freq")
    self._l2_weight_threshold = self._initializer_op.get_attr("l2_weight_threshold")
    self._max_element_size = self._initializer_op.get_attr("max_element_size")
    self._false_positive_probability = self._initializer_op.get_attr("false_positive_probability")
    self._counter_type = self._initializer_op.get_attr("counter_type")
    self._storage_type = self._initializer_op.get_attr("storage_type")
    self._storage_path = self._initializer_op.get_attr("storage_path")
    self._storage_size = self._initializer_op.get_attr("storage_size")
    self._default_value_dim = self._initializer_op.get_attr("default_value_dim")
    self._record_freq = self._initializer_op.get_attr("record_freq")
    self._record_version = self._initializer_op.get_attr("record_version")

  # LINT.ThenChange(//tensorflow/python/eager/graph_callable.py)

  def set_init_data_source_initializer(self, init_data_source):
    import pkgutil
    try:
      # pylint: disable=unused-import
      from paiio.python.ops.io_ops import feature_store_initializer 
      # pylint: enable=unused-import
    except BaseException as error:
      import sys, traceback
      print('You need paiio to use feature store', file=sys.stderr)
      raise
    steps_to_live_hybrid = self._steps_to_live 
    if not self._trainable:
      steps_to_live_hybrid = -214 
    with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
      rank = initial_value.get_shape().rank -1
      kv_init_op = gen_kv_variable_ops.initialize_kv_variable_op(
          self._handle,
          variables._try_guard_against_uninitialized_dependencies(self.name, self._initial_value),
          ops.convert_to_tensor(self._invalid_key),
          shape=self._initial_value.get_shape()[rank:],
          steps_to_live=steps_to_live_hybrid,
          name=n)
      with ops.control_dependencies([kv_init_op]):
        is_partitioned_ev = self._save_slice_info is not None
        partition_id = self._save_slice_info.var_offset[0] if is_partitioned_ev else 0
        partition_num = self._save_slice_info.full_shape[0] if is_partitioned_ev else 1
        self._initializer_op = (
           feature_store_initializer(
               self._init_data_source, 
               self._handle, 
               variables._try_guard_against_uninitialized_dependencies(self.name, self._initial_value),
               ops.convert_to_tensor(self._invalid_key),
               self._initial_value.get_shape()[rank:], 
               self._steps_to_live, partition_id, partition_num)
        )

  def recover_from_init_data_source(self, init_data_source, partition_id, partition_num):
    import pkgutil
    try:
      # pylint: disable=unused-import
      from paiio.python.ops.io_ops import feature_store_initializer 
      # pylint: enable=unused-import
    except BaseException as error:
      import sys, traceback
      print('You need paiio to use feature store', file=sys.stderr)
      raise
    steps_to_live_hybrid = self._steps_to_live 
    if not self._trainable:
      steps_to_live_hybrid = -214 
    with ops.name_scope("RecoverAssign") as n, ops.colocate_with(self._handle):
      rank = initial_value.get_shape().rank -1 
      kv_init_op = gen_kv_variable_ops.initialize_kv_variable_op(
          self._handle,
          variables._try_guard_against_uninitialized_dependencies(self.name, self._initial_value),
          ops.convert_to_tensor(self._invalid_key),
          shape=self._initial_value.get_shape()[rank:],
          steps_to_live=steps_to_live_hybrid,
          name=n)
      with ops.control_dependencies([kv_init_op]):
        return (
           feature_store_initializer(
               self._init_data_source, 
               self._handle, 
               variables._try_guard_against_uninitialized_dependencies(self.name, self._initial_value),
               ops.convert_to_tensor(self._invalid_key),
               self._initial_value.get_shape()[rank:], 
               self._steps_to_live, partition_id, partition_num)
        )

  def reconstruct_initialize_op(self):
    pass

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

  @property
  def device(self):
    """The device this variable is on."""
    return self._handle_device

  @property
  def graph(self):
    """The `Graph` of this variable."""
    return self._handle.graph

  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._handle_name

  @property
  def shape(self):
    """The embedding dim of this variable."""
    return self._graph_shape

  def total_count(self):
    """The shape of this variable."""
    return gen_kv_variable_ops.kv_variable_shape(self._handle,
		    Tkeys=self._invalid_key_type)

  def export(self):
    return gen_kv_variable_ops.kv_resource_export(self._handle,
		    self._invalid_key_type, self.dtype)

  @property
  def steps_to_live(self):
    return self._steps_to_live
  
  @property
  def storage_type(self):
    return self._storage_type

  @property
  def block_num(self):
    if self._block_num is None:
      return 1
    else:
      return self._block_num

  @property
  def create(self):
    """The op responsible for initializing this variable."""
    if not self._in_graph_mode:
      raise RuntimeError("Calling create in EAGER mode not supported.")
    return self._initializer_op

  @property
  def handle(self):
    """The handle by which this variable can be accessed."""
    return self._handle

  @property
  def invalid_key(self):
    return self._invalid_key

  def value(self):
    """A cached operation which reads the value of this variable."""
    raise NotImplementedError("EV value")

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._graph_element

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initializer_op

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable."""
    if context.executing_eagerly():
      raise RuntimeError("initial_value not supported in EAGER mode.")
    return self._initial_value

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    return self._constraint

  @property
  def op(self):
    """The op for this variable."""
    return self._handle.op

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    if context.in_eager_mode():
      raise RuntimeError("Trying to eval in EAGER mode")
    return self._graph_element.eval(session=session)

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `EmbeddingVariable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  def _read_variable_op(self):
    raise NotImplementedError("_read_variable_op")

  def read_value(self):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
     the read operation.
    """
    raise NotImplementedError("EmbeddingVariable does not implement read_value()")

  def sparse_read(self, indices, name=None, ev_init_value=None, counts=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      if self._trainable:
        tape.variable_accessed(self)
      if ev_init_value is not None:
        default_value = ev_init_value
        is_use_default_value_tensor = True
      else:
        default_value = ops.convert_to_tensor(1.0)
        is_use_default_value_tensor = False
      if counts != None:
        value = gen_kv_variable_ops.kv_resource_gather_v1(self._handle,
              indices,
              default_value,
              counts, name=name)
      else:
        value = gen_kv_variable_ops.kv_resource_gather(self._handle,
              indices,
              default_value,
              is_use_default_value_tensor,
              name=name)
    return array_ops.identity(value)

  def to_proto(self, export_scope=None):
    """Converts a `EmbeddingVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if context.executing_eagerly():
      raise RuntimeError("to_proto not supported in EAGER mode.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      if self._initial_value is not None:
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name,
                                                     export_scope)
      var_def.is_resource = True
      var_def.is_embedding_var = True
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("from_proto not supported in EAGER mode.")
    return EmbeddingVariable(
        variable_def=variable_def, import_scope=import_scope)

  def _AsTensor(self):
    return self.value()

  def _ref(self):
    """Unsupported."""
    raise NotImplementedError("EmbeddingVariable does not implement _ref()")

  __array_priority__ = 100

  def assign_sub(self, delta, use_locking=None, name=None):
    raise NotImplementedError("EmbeddingVariable does not implement assign_sub()")

  def assign_add(self, delta, use_locking=None, name=None):
    raise NotImplementedError("EmbeddingVariable does not implement assign_add()")

  def assign(self, value, use_locking=None, name=None):
    raise NotImplementedError("EmbeddingVariable does not implement assign()")

  def _embedding_variable_handle(self, shape, dtype, shared_name, name, graph_mode):
    """Creates a variable handle with information to do shape inference."""
    container = ops.get_default_graph()._container  # pylint: disable=protected-access
    if container is None:
      container = ""
    return gen_kv_variable_ops.kv_var_handle_op(shape=shape, dtype=dtype,
                                                shared_name=shared_name,
                                                name=name, 
                                                Tkeys=self._invalid_key_type,
                                                container=container)

  def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask,
                            end_mask, ellipsis_mask, new_axis_mask,
                            shrink_axis_mask):
    with ops.control_dependencies([
        gen_array_ops.resource_strided_slice_assign(
            ref=self.handle,
            begin=begin,
            end=end,
            strides=strides,
            value=value,
            name=name,
            begin_mask=begin_mask,
            end_mask=end_mask,
            ellipsis_mask=ellipsis_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask)
    ]):
      return self.value()

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    del name
    if dtype is not None and dtype != self.dtype:
      print("trying to switch the dtype to ", dtype, " from ", self.dtype)
      return NotImplemented
    if as_ref:
      return self.read_value().op.inputs[0]
    else:
      return self.value()


class MultiHashVariable(resource_variable_ops.ResourceVariable):
  def __init__(self, name, val_list, mhvconfig):
    if not val_list:
      raise ValueError("val list must not be empty")
    self._val_list = val_list
    self._name = name
    self._mhvconfig = mhvconfig
  @property
  def val_list(self):
    return self._val_list
  @property
  def mhvconfig(self):
    return self._mhvconfig

class DynamicEmbeddingVariable(resource_variable_ops.ResourceVariable):
  def __init__(self, name, ev_list):
    if not ev_list:
      raise ValueError("ev list must not be empty")
    self._ev_list = ev_list
    self._head_ev = ev_list[0]
    self._name = name
    self._graph_element = ev_list[0]._graph_element
    self._dtype = ev_list[0]._dtype
  
  def sparse_read(self, indices, blocknums, name=None):
    evnum = len(self._ev_list)
    embs =[]
    for i in range(evnum):
      evids = array_ops.boolean_mask(indices, math_ops.greater_equal(blocknums, i + 1))
      gathered_emb = self._ev_list[i].sparse_read(evids, name=name) 
      embs.append(gathered_emb)
    return embs
  def mainev(self):
    return self._ev_list[0]
  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._name
  
  def _get_save_slice_info(self):
    return self._ev_list[0]._save_slice_info
  @property
  def shape(self):
    """The embedding dim of this variable."""
    return tensor_shape.TensorShape(self.mainev().shape[0] * len(self._ev_list))
  def blocknum(self):
    return len(self._ev_list)


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access

def identity(var):
  if "GPU" in var.device:
    with ops.device(var.device):
      keys, values, versions, freqs =  gen_kv_variable_ops.kv_resource_export(var._handle, Tkeys=var._invalid_key_type, Tvalues=var.dtype)
      return [keys, values, versions, freqs]
  else:
    return var.handle


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.

# Note: registering for Variable after EmbeddingVariable because inheritance will
# otherwise lead to the wrong behavior.
ops.register_tensor_conversion_function(EmbeddingVariable, _dense_var_to_tensor)
ops.register_tensor_conversion_function(
    variables.Variable, variables.Variable._TensorConversionFunction)  # pylint: disable=protected-access

# pylint: disable=protected-access
EmbeddingVariable._OverloadAllOperators()
ops.register_dense_tensor_like_type(EmbeddingVariable)


@ops.RegisterGradient("ReadKvVariableOp")
def _ReadGrad(_, grad):
  """Gradient for read op."""
  return grad


@ops.RegisterGradient("KvResourceGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  # Walk graph back until the original handle is found.
  # TODO(apassos): more robust way of getting the shape.
  # TODO(apassos): implement this for EAGER mode.
  handle = op.inputs[0]
  while handle.op.type != "KvVarHandleOp":
    handle = handle.op.inputs[0]
  params_shape = ops.convert_to_tensor(
      tensor_shape.TensorShape(handle.op.get_attr("shape")))
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[0:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None, None]

@ops.RegisterGradient("KvResourceGatherV1")
def _GatherV1Grad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  # Walk graph back until the original handle is found.
  # TODO(apassos): more robust way of getting the shape.
  # TODO(apassos): implement this for EAGER mode.
  handle = op.inputs[0]
  while handle.op.type != "KvVarHandleOp":
    handle = handle.op.inputs[0]
  params_shape = ops.convert_to_tensor(
      tensor_shape.TensorShape(handle.op.get_attr("shape")))
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[0:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None, None, None]

