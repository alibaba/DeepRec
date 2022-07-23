# Copyright (c) 2017, Alibaba Inc.
# All right reserved.
#
# Author: Chen Ding <cnady.dc@alibaba-inc.com>
# Created: 2018/03/26
# Description:
# ==============================================================================

"""Tests for tensorflow.ops.embedding_variable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.check_ops import assert_equal
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import sparse_tensor
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.training import ftrl
from tensorflow.python.training import adam
from tensorflow.python.training import adam_async
from tensorflow.python.training import adagrad
from tensorflow.python.training import adagrad_decay
from tensorflow.python.training import adagrad_decay_v2
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.contrib.layers.python.layers import embedding_ops as emb_ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader
from tensorflow.core.protobuf import config_pb2 as config_pb3
import time
import random

class EmbeddingVariableTest(test_util.TensorFlowTestCase):
  def testSaveVersionWithGlobalStepEviction(self):
    print("testSaveVersionWithGlobalStepEviction")
    checkpoint_directory = self.get_temp_dir()
    var = variable_scope.get_embedding_variable("var_1",
                                        embedding_dim=6,
                                        initializer=init_ops.ones_initializer,
                                        steps_to_live = 5)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([5], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver()
    model_path = os.path.join(checkpoint_directory, "model.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      r, _, _ = sess.run([emb, train_op,loss])
      r, _, _ = sess.run([emb, train_op,loss])
      saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 1)

  def testDynamicDimensionEmbeddingVariable(self):
    print("testDynamicDimensionEmbeddingVariable")
    def runTestAdagradDecay(self, var, g):
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      else:
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64), blocknums=[2,2,2,2,2,2])
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            initializer=init_ops.ones_initializer(dtypes.float32),
            embedding_dim = 8,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb1 = runTestAdagradDecay(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var =  variable_scope.get_dynamic_dimension_embedding_variable("var_dist",
                                                                    embedding_block_dimension=4,
                                                                    embedding_block_num=2,
                                                                    initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagradDecay(self, var, g)

    for i in range(0, 6):
      for j in range(0, 8):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testDynamicEmbeddingVariableForInitFromProto(self):
    print("testDynamicEmbeddingVariableForInitFromProto")
    embedding = variable_scope.get_dynamic_dimension_embedding_variable("var_dist",
                                                                    embedding_block_dimension=4,
                                                                    embedding_block_num=2,
                                                                    initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(embedding, math_ops.cast([0,1,2,5,6,7], dtypes.int64), blocknums=[2,2,2,2,2,2])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    graph = ops.get_default_graph()
    meta_graph_def = saver_module.export_meta_graph()
    ops.reset_default_graph()
    with self.test_session() as sess:
      res = saver_module.import_meta_graph(meta_graph_def)

  def testEmbeddingVariableForInitFromProto(self):
    print("testEmbeddingVariableForInitFromProto")
    embedding = variable_scope.get_embedding_variable("var_dist",
                                          embedding_dim=6,
                                          initializer=init_ops.ones_initializer,
                                          steps_to_live = 4,
                                          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(embedding, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    graph = ops.get_default_graph()
    meta_graph_def = saver_module.export_meta_graph()
    ops.reset_default_graph()
    with self.test_session() as sess:
      res = saver_module.import_meta_graph(meta_graph_def)

  def testEmbeddingVariableForLookupInt64(self):
    print("testEmbeddingVariableForLookupInt64")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForLookupInt32(self):
    print("testEmbeddingVariableForLookupInt32")
    checkpoint_directory = self.get_temp_dir()
    var = variable_scope.get_embedding_variable("var_1",
                                                embedding_dim = 3,
                                                key_dtype=dtypes.int32,
                                                initializer=init_ops.ones_initializer(dtypes.float32),
                                                partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int32))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adam.AdamOptimizer(0.01)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op])
      model_path = os.path.join(checkpoint_directory, "model.ckpt")
      save_path = saver.save(sess, model_path, global_step=12345)
      saver.restore(sess, save_path)

  def testEmbeddingVariableForExport(self):
    print("testEmbeddingVariableForExport")
    with ops.device('/cpu:0'):
      ev_config = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=1))
      var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
              initializer=init_ops.ones_initializer(dtypes.float32), steps_to_live=10000, ev_option=ev_config)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      init = variables.global_variables_initializer()
      keys, values, versions, freqs = var.export()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run(emb)
        sess.run(emb)
        sess.run(emb)
        fetches = sess.run([keys, values, versions, freqs])
        print(fetches)
        self.assertAllEqual([0, 1, 2, 5, 6, 7], fetches[0])
        self.assertAllEqual([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]], fetches[1])
        self.assertAllEqual([0, 0, 0, 0, 0, 0], fetches[2])
        self.assertAllEqual([1, 1, 1, 1, 1, 1], fetches[3])

  def testEmbeddingVariableForGetShape(self):
    print("testEmbeddingVariableForGetShape")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    shape = var.total_count()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([emb])
      self.assertAllEqual([6, 3], sess.run(shape))

  def testEmbeddingVariableForMultiHashAdd(self):
    print("testEmbeddingVariableForMultiHashAdd")
    var1 = variable_scope.get_variable("var_1", shape=[5,6],
                                    initializer=init_ops.ones_initializer(dtypes.float32),
                                    partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    var2 = variable_scope.get_variable("var_2", shape=[3,6],
                                    initializer=init_ops.ones_initializer(dtypes.float32),
                                    partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
    ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
    emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
    emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
    emb = math_ops.add(emb1, emb2)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)

    ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
    var_multi = variable_scope.get_multihash_variable("var_multi",
                                        [[5,6],[3,6]],
                                        complementary_strategy="Q-R",
                                        initializer=init_ops.ones_initializer,
                                        partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2)
                                      )
    emb_multi =  embedding_ops.embedding_lookup(var_multi, ids)
    fun_m = math_ops.multiply(emb_multi, 2.0, name='multiply')
    loss_m = math_ops.reduce_sum(fun_m, name='reduce_sum')
    gs_m = training_util.get_or_create_global_step()
    opt_m = adagrad_decay.AdagradDecayOptimizer(0.1, gs_m)
    g_v_m = opt_m.compute_gradients(loss_m)
    train_op_m = opt_m.apply_gradients(g_v_m)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op, train_op_m])
      val_list = sess.run([emb, emb_multi])
      for i in range(ids.shape.as_list()[0]):
        self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddingVariableForMultiHashMul(self):
    print("testEmbeddingVariableForMultiHashMul")
    var1 = variable_scope.get_variable("var_1", shape=[5,6],
                                    initializer=init_ops.ones_initializer(dtypes.float32),
                                    partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    var2 = variable_scope.get_variable("var_2", shape=[3,6],
                                    initializer=init_ops.ones_initializer(dtypes.float32),
                                    partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
    ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
    emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
    emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
    emb = math_ops.multiply(emb1, emb2)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)

    ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
    var_multi = variable_scope.get_multihash_variable("var_multi",
                                        [[5,6],[3,6]],
                                        complementary_strategy="Q-R",
                                        operation="mul",
                                        initializer=init_ops.ones_initializer,
                                        partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2)
                                      )
    emb_multi =  embedding_ops.embedding_lookup(var_multi, ids)
    fun_m = math_ops.multiply(emb_multi, 2.0, name='multiply')
    loss_m = math_ops.reduce_sum(fun_m, name='reduce_sum')
    gs_m = training_util.get_or_create_global_step()
    opt_m = adagrad_decay.AdagradDecayOptimizer(0.1, gs_m)
    g_v_m = opt_m.compute_gradients(loss_m)
    train_op_m = opt_m.apply_gradients(g_v_m)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op, train_op_m])
      val_list = sess.run([emb, emb_multi])
      for i in range(ids.shape.as_list()[0]):
        self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddingVariableForMultiHashConcat(self):
    print("testEmbeddingVariableForMultiHashConcat")
    var1 = variable_scope.get_variable("var_1", shape=[5,6],
                                    initializer=init_ops.ones_initializer(dtypes.float32),
                                    partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    var2 = variable_scope.get_variable("var_2", shape=[3,6],
                                    initializer=init_ops.ones_initializer(dtypes.float32),
                                    partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
    ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
    emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
    emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
    emb = array_ops.concat([emb1, emb2], 1)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)

    ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
    var_multi = variable_scope.get_multihash_variable("var_multi",
                                        [[5,6],[3,6]],
                                        complementary_strategy="Q-R",
                                        operation="concat",
                                        initializer=init_ops.ones_initializer,
                                        partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2)
                                      )
    emb_multi =  embedding_ops.embedding_lookup(var_multi, ids)
    fun_m = math_ops.multiply(emb_multi, 2.0, name='multiply')
    loss_m = math_ops.reduce_sum(fun_m, name='reduce_sum')
    gs_m = training_util.get_or_create_global_step()
    opt_m = adagrad_decay.AdagradDecayOptimizer(0.1, gs_m)
    g_v_m = opt_m.compute_gradients(loss_m)
    train_op_m = opt_m.apply_gradients(g_v_m)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op, train_op_m])
      val_list = sess.run([emb, emb_multi])
      for i in range(ids.shape.as_list()[0]):
        self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddingVariableForSaveAndRestore(self):
    print("testEmbeddingVariableForSaveAndRestore")
    checkpoint_directory = self.get_temp_dir()
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                                                                                                 storage_path='/tmp/leveldb/')))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run(emb)
      emb_ori = sess.run(emb)
      emb_ori = sess.run(emb)
      emb_ori = sess.run(emb)
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
      print(save_path)
      for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
        print('loading... ', name, shape)

    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
      self.assertAllEqual(emb_ori, sess.run(emb))

  def testEmbeddingVariableForL2FeatureEvictionFromContribFeatureColumn(self):
    print("testEmbeddingVariableForL2FeatureEvictionFromContribFeatureColumn")
    checkpoint_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    columns = feature_column.sparse_column_with_embedding(
                                        column_name="col_emb",
                                        dtype=dtypes.int64,
                                        ev_option = variables.EmbeddingVariableOption(evict_option=evict))
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            combiner="mean")
    ids = {}
    ids["col_emb"] = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
    emb= feature_column_ops.input_from_feature_columns(
           columns_to_tensors=ids, feature_columns=[W])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run([emb, train_op])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model1.ckpt"), global_step=12345)
    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model1.ckpt-12345"))
      emb_right = [[0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.7927219, 0.7927219, 0.7927219],
                   [0.7927219, 0.7927219, 0.7927219],
                   [1.0, 1.0, 1.0]]
      emb_ori = sess.run(emb)
      for i in range(6):
        for j in range(3):
          self.assertAlmostEqual(emb_ori[i][j], emb_right[i][j])

  def testEmbeddingVariableForL2FeatureEviction(self):
    print("testEmbeddingVariableForL2FeatureEviction")
    checkpoint_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(evict_option=evict))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,0,0,1,1,2], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run([emb, train_op])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model1.ckpt"), global_step=12345)
      #for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
      #  print('loading... ', name, shape)
    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model1.ckpt-12345"))
      emb_right = [[0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.7927219, 0.7927219, 0.7927219],
                   [0.7927219, 0.7927219, 0.7927219],
                   [1.0, 1.0, 1.0]]
      emb_ori = sess.run(emb)
      for i in range(6):
        for j in range(3):
          self.assertAlmostEqual(emb_ori[i][j], emb_right[i][j])

  def testEmbeddingVariableForSparseColumnSharedEmbeddingCol(self):
    columns_list=[]
    columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string))
    W = feature_column.shared_embedding_columns(sparse_id_columns=columns_list,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            shared_embedding_name="xxxxx_shared")

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0]], values=["aaaa","bbbbb","ccc","4nn","5b"], dense_shape=[5, 5])
    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=W)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForFeatureFilterFromContribFeatureColumn(self):
    print("testEmbeddingVariableForFeatureFilterFromContribFeatureColumn")
    columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.int64,
                                                          ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)))
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0]], values=math_ops.cast([1,1,1,1,2,2,2,3,3,4], dtypes.int64), dense_shape=[10, 1])
    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')

    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val1 in emb1.tolist():
        for val in val1:
          self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for index, val1 in enumerate(emb1.tolist()):
        if index < 7:
          for val in val1:
            self.assertNotEqual(val, 1.0)
        else:
          for val in val1:
            self.assertEqual(val, 1.0)

  def testEmbeddingVariableForSparseColumnEmbeddingCol(self):
    columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.int64)
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32))

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=math_ops.cast([1,2,3,4,5], dtypes.int64), dense_shape=[5, 4])

    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForShrinkNone(self):
      print("testEmbeddingVariableForShrink")
      var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            steps_to_live = 5,
            initializer=init_ops.ones_initializer(dtypes.float32))
      ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      gs = training_util.get_or_create_global_step()
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        for i in range(10):
          print(sess.run([emb, train_op,loss], feed_dict={'ids:0': 2*i}))

  def testEmbeddingVariableForWeightedSumFromFeatureColumn(self):
    print("testEmbeddingVariableForWeightedSumFromFeatureColumn")
    columns_list=[]
    columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string))
    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0]], values=["aaaa","bbbbb","ccc","4nn","5b"], dense_shape=[5, 5])

    emb, _, _ = feature_column_ops.weighted_sum_from_feature_columns(columns_to_tensors=ids, feature_columns=columns_list, num_outputs=2)

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForBloomFilterInt64(self):
    print("testEmbeddingVariableForBloomFilterInt64")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01)),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForBloomFilterInt32(self):
    print("testEmbeddingVariableForBloomFilterInt32")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01,
                                      counter_type = dtypes.uint32
                                    )),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForBloomFilterInt8(self):
    print("testEmbeddingVariableForBloomFilterInt8")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01,
                                      counter_type = dtypes.uint8
                                    )),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForBloomFilterInt16(self):
    print("testEmbeddingVariableForBloomFilterInt16")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01,
                                      counter_type = dtypes.uint16
                                    )),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdagradDecayFilter(self):
    print("testEmbeddingVariableForAdagradDecayFilter")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForFtrlFilter(self):
    print("testEmbeddingVariableForFtrlFilter")
    with ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      #var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb1, top, l = sess.run([emb, train_op, loss])
        for val in emb1.tolist()[0]:
          self.assertEqual(val, 1.0)
        emb1, top, l = sess.run([emb, train_op, loss])
        for val in emb1.tolist()[0]:
          self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdamAsyncFilter(self):
    print("testEmbeddingVariableForAdamAsynsFilter")
    with ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam_async.AdamAsyncOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb1, top, l = sess.run([emb, train_op, loss])
        for val in emb1.tolist()[0]:
          self.assertEqual(val, 1.0)
        emb1, top, l = sess.run([emb, train_op, loss])
        for val in emb1.tolist()[0]:
          self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForGradientDescentFilter(self):
    print("testEmbeddingVariableForGradientDescentFilter")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdagradDecayV2Filter(self):
    print("testEmbeddingVariableForAdagradDecayV2Filter")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdamFilter(self):
    print("testEmbeddingVariableForAdamFilter")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adam.AdamOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForGradientDescent(self):
    print("testEmbeddingVariableForGradientDescent")
    def runTestGradientDescent(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = gradient_descent.GradientDescentOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    emb_var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3,
          initializer=init_ops.ones_initializer(dtypes.float32),
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
    emb1 = runTestGradientDescent(self, emb_var)
    emb2 = runTestGradientDescent(self, var)

    for i in range(0, 6):
      for j in range(0, 3):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagrad(self):
    print("testEmbeddingVariableForAdagrad")
    def runTestAdagrad(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var)
      emb2 = runTestAdagrad(self, var)

    for i in range(0, 6):
      for j in range(0, 3):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagradDecay(self):
    print("testEmbeddingVariableForAdagradDecay")
    with ops.device('/cpu:0'):
      def runTestAdagradDecay(self, var):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session() as sess:
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
          sess.run([init])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            initializer=init_ops.ones_initializer(dtypes.float32),
            embedding_dim = 3,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagradDecay(self, emb_var)
      emb2 = runTestAdagradDecay(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagradDecayV2(self):
    print("testEmbeddingVariableForAdagradDecayV2")
    with ops.device('/cpu:0'):
      def runTestAdagradDecayV2(self, var):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session() as sess:
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
          sess.run([init])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagradDecayV2(self, emb_var)
      emb2 = runTestAdagradDecayV2(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdam(self):
    print("testEmbeddingVariableForAdam")
    def runTestAdam(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam.AdamOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    emb_var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3,
          initializer=init_ops.ones_initializer(dtypes.float32),
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    var = variable_scope.get_variable("var_2", shape=[8, 3], initializer=init_ops.ones_initializer(dtypes.float32))
    emb1 = runTestAdam(self, emb_var)
    emb2 = runTestAdam(self, var)

    print(emb1.tolist())
    print(emb2.tolist())
    for i in range(0, 6):
      for j in range(0, 3):
        self.assertAlmostEqual(emb1.tolist()[i][j], emb2.tolist()[i][j], delta=1e-05)

  def testEmbeddingVariableForAdamAsync(self):
    print("testEmbeddingVariableForAdamAsync")
    def runTestAdamAsync(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam_async.AdamAsyncOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    emb_var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3,
          initializer=init_ops.ones_initializer(dtypes.float32),
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
    var = variable_scope.get_variable("var_2", shape=[8, 3],
          initializer=init_ops.ones_initializer(dtypes.float32))
    emb1 = runTestAdamAsync(self, emb_var)
    emb2 = runTestAdamAsync(self, var)

    for i in range(0, 6):
      for j in range(0, 3):
        self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])


  def testEmbeddingVariableForFtrl(self):
    print("testEmbeddingVariableForFtrl")
    def runTestAdam(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    emb_var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
          initializer=init_ops.ones_initializer(dtypes.float32),
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
    emb1 = runTestAdam(self, emb_var)
    emb2 = runTestAdam(self, var)

      #for i in range(0, 6):
      #  for j in range(0, 3):
      #    self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagradDecayStep(self):
    print("testEmbeddingVariableForAdagradDecayStep")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs, accumulator_decay_step=2)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      self.assertEqual(36.0, sess.run([emb, train_op, loss])[2])
      self.assertAlmostEqual(32.444176, sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(29.847788, sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(27.74195 , sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(25.852505, sess.run([emb, train_op, loss])[2], delta=1e-05)

  def testEmbeddingVariableRestoreSavedModel(self):
    with ops.device("/cpu:0"):
      checkpoint_directory = self.get_temp_dir() + "/save_model"
      print("testEmbeddingVariableRestoreSavedModel")
      # build graph
      columns_list=[]
      columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string))
      W = feature_column.shared_embedding_columns(sparse_id_columns=columns_list,
              dimension=3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              shared_embedding_name="xxxxx_shared")

      ids={}
      ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0]], values=["aaaa","bbbbb","ccc"], dense_shape=[3, 5])
      emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=W)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      gs = training_util.get_or_create_global_step()
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run(init)
        builder = saved_model_builder.SavedModelBuilder(checkpoint_directory)
        builder.add_meta_graph_and_variables(sess, ['tag_string'])
        builder.save()
      # load savedmodel
      with self.test_session() as sess:
        loader.load(sess, ['tag_string'], checkpoint_directory)

  def testEmbeddingVariableForGeneralConstInitializer(self):
    print("testEmbeddingVariableForGeneralConstInitializer")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,6], dtypes.int64))
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb_val = sess.run(emb)
        for i in xrange(2):
          for j in xrange(3):
            self.assertAlmostEqual(1.0, emb_val[i][j], delta=1e-05)

  def testEmbeddingVariableForGeneralRandomInitializer(self):
    print("testEmbeddingVariableForGeneralRandomInitializer")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              #initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,6], dtypes.int64))
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb_val = sess.run(emb)
        print(emb_val)
        for i in xrange(3):
          self.assertNotEqual(emb_val[0][i], emb_val[1][i])
          self.assertNotEqual(emb_val[0][i], emb_val[1][i])
          self.assertNotEqual(emb_val[0][i], emb_val[1][i])

  def testEmbeddingVariableForHTPartitionNum(self):
    print("testEmbeddingVariableForHTPartitionNum")
    ev_option = variables.EmbeddingVariableOption(ht_partition_num=20)
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option=ev_option)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForLayout(self):
    print("testEmbeddingVariableForLayout")
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5)
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=5)))
      emb1 = runTestAdagrad(self, emb_var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], 1.0)

  def testEVInitializerWithKeyFetch(self):
    print("testEVInitializerWithKeyFetch")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var", shape=[8,3],
                                        initializer=init_ops.glorot_uniform_initializer(seed = 1))
      init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                         default_value_dim=8)
      ev_option = variables.EmbeddingVariableOption(init_option=init_opt)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                       ev_option=ev_option)
      var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
      emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([0,1,2,5,6,7,8,9,10], dtypes.int64))
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1 = sess.run(var_emb)
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[0], emb2.tolist()[0])
        self.assertListEqual(emb1.tolist()[1], emb2.tolist()[1])
        self.assertListEqual(emb1.tolist()[2], emb2.tolist()[2])
        self.assertListEqual(emb1.tolist()[5], emb2.tolist()[3])
        self.assertListEqual(emb1.tolist()[6], emb2.tolist()[4])
        self.assertListEqual(emb1.tolist()[7], emb2.tolist()[5])
        self.assertListEqual(emb1.tolist()[0], emb2.tolist()[6])
        self.assertListEqual(emb1.tolist()[1], emb2.tolist()[7])
        self.assertListEqual(emb1.tolist()[2], emb2.tolist()[8])

  def testEVInitializerWithCounterFeatureFilter(self):
    def testembedding(emb1, emb2):
      is_match = 0
      for i in range(8):
        for j in range(3):
          if emb1.tolist()[i][j] != emb2.tolist()[3][j]:
            break
        if j == 2:
          is_match = 1
      return is_match

    print("testEVInitializerWithCounterFeatureFilter")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var", shape=[8,3],
                                        initializer=init_ops.glorot_uniform_initializer(seed = 1))
      counter_filter_option=variables.CounterFilter(filter_freq=3)
      init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                         default_value_dim=8)
      ev_option = variables.EmbeddingVariableOption(init_option=init_opt, filter_option=counter_filter_option)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                       ev_option=ev_option)
      var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
      emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([3], dtypes.int64))
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1 = sess.run(var_emb)
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])

  def testEVInitializerWithBloomFeatureFilter(self):
    def testembedding(emb1, emb2):
      is_match = 0
      for i in range(8):
        for j in range(3):
          if emb1.tolist()[i][j] != emb2.tolist()[0][j]:
            break
        if j == 2:
          is_match = 1
      return is_match

    print("testEVInitializerWithBloomFeatureFilter")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var", shape=[8,3],
                                        initializer=init_ops.glorot_uniform_initializer(seed = 1))
      bloom_filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 10,
                                      false_positive_probability = 0.01)
      init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                         default_value_dim=8)
      ev_option = variables.EmbeddingVariableOption(init_option=init_opt, filter_option=bloom_filter_option)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                       ev_option=ev_option)
      var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
      emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([3], dtypes.int64))
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1 = sess.run(var_emb)
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])

  def testEVInitializer(self):
    def runTest(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        return r
    print("testEVInitializer")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      init = variables.InitializerOption(default_value_dim=8192)
      ev_option = variables.EmbeddingVariableOption(init_option = init)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim = 6,
                                        initializer=init_ops.glorot_uniform_initializer(seed = 3),
                                        ev_option = ev_option)

      var = variable_scope.get_variable("var", shape=[8192, 6],
                                   initializer=init_ops.glorot_uniform_initializer(seed = 3))
      emb1 = runTest(self, emb_var, g)
      emb2 = runTest(self, var, g)

      for i in range(0, 6):
        for j in range(0, 6):
          self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForDRAM(self):
    print("testEmbeddingVariableForDRAM")
    def runTestAdagrad(self, var, g):
      search_list=[]
      for i in range(0, 1024 * 2):
        search_list.append(i)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast(search_list, dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      #print(ops.get_default_graph().as_graph_def())
      #config = config_pb3.ConfigProto(log_device_placement=True)
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r = sess.run([emb])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 128,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM)))
      var = variable_scope.get_variable("var_2", shape=[1024 * 512, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      time_start = time.time()
      emb1 = runTestAdagrad(self, emb_var, g)
      print(emb1)
      time_end = time.time()
      time_c = time_end - time_start
      print('time cost', time_c, 's')
      #emb2 = runTestAdagrad(self, var, g)
      #print(emb2)

      #for i in range(0, 6):
        #for j in range(0, 3):
          #self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForLEVELDBWithAdagrad(self):
    print("testEmbeddingVariableForLEVELDBWithAdagrad")
    db_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        print(r)
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForLEVELDBwithFilter(self):
    print("testEmbeddingVariableForLEVELDBwithFilter")
    db_directory = self.get_temp_dir()
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3),
                               storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                      storage_path=db_directory)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb1, top, l = sess.run([emb, train_op, loss])

  def testEmbeddingVariableForLEVELDBWithGradientDescent(self):
    print("testEmbeddingVariableForLEVELDBWithGradientDescent")
    db_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = gradient_descent.GradientDescentOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForLEVELDBWithAdam(self):
    print("testEmbeddingVariableForLEVELDBWithAdam")
    db_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam.AdamOptimizer(0.01)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForLEVELDBWithAdamAsync(self):
    print("testEmbeddingVariableForLEVELDBWithAdamAsync")
    db_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam_async.AdamAsyncOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForLEVELDBWithAdagradDecay(self):
    print("testEmbeddingVariableForLEVELDBWithAdagradDecay")
    db_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForLEVELDBWithAdagradDecayV2(self):
    print("testEmbeddingVariableForLEVELDBWithAdagradDecayV2")
    db_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testLevelDBCheckpoint(self):
    db_directory = self.get_temp_dir()
    checkpoint_directory = self.get_temp_dir()
    emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.LEVELDB,
                                                                                                 storage_path=db_directory)))
    emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver()
    model_path = os.path.join(checkpoint_directory, "model.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      r, _, _ = sess.run([emb, train_op,loss])
      r, _, _ = sess.run([emb, train_op,loss])
      saver.save(sess, model_path)
      r, _ = sess.run([emb, loss])
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-values":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          for j in range(0, 3):
            self.assertEqual(ckpt_value.tolist()[0][j], r[0][j])
            self.assertEqual(ckpt_value.tolist()[1][j], r[3][j])
            self.assertEqual(ckpt_value.tolist()[2][j], r[5][j])
        if name == "var_1/AdagradDecay-values":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          slot = [[72.1, 72.1, 72.1], [32.1, 32.1, 32.1], [8.1, 8.1, 8.1]]
          for j in range(0, 3):
            self.assertAlmostEqual(ckpt_value.tolist()[0][j], slot[0][j], delta=1e-5)
            self.assertAlmostEqual(ckpt_value.tolist()[1][j], slot[1][j], delta=1e-5)
            self.assertAlmostEqual(ckpt_value.tolist()[2][j], slot[2][j], delta=1e-5)
    with self.test_session() as sess:
      saver.restore(sess, model_path)
      r1, _, _ = sess.run([emb, train_op,loss])
      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(r[i][j], r1.tolist()[i][j])

  def testEmbeddingVariableForSaveFreq(self):
    db_directory = self.get_temp_dir()
    checkpoint_directory = self.get_temp_dir()
    emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                                                                                                 storage_path=db_directory)))
    emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver()
    model_path = os.path.join(checkpoint_directory, "model.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      r, _, _ = sess.run([emb, train_op,loss])
      r, _, _ = sess.run([emb, train_op,loss])
      saver.save(sess, model_path)
      r, _ = sess.run([emb, loss])
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
          self.assertEqual(ckpt_value.tolist()[2], 2)


  def testEmbeddingVariableForL2FeatureEvictionDRAM(self):
    print("testEmbeddingVariableForL2FeatureEvictionDRAM")
    checkpoint_directory = self.get_temp_dir()
    db_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    storage_option = variables.StorageOption(storage_type=config_pb2.StorageType.DRAM)
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(evict_option=evict, storage_option=storage_option))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,0,0,1,1,2], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run([emb, train_op])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model1.ckpt"), global_step=12345)
      #for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
      #  print('loading... ', name, shape)
    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model1.ckpt-12345"))
      emb_right = [[0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.7927219, 0.7927219, 0.7927219],
                   [0.7927219, 0.7927219, 0.7927219],
                   [1.0, 1.0, 1.0]]
      emb_ori = sess.run(emb)
      for i in range(6):
        for j in range(3):
          self.assertAlmostEqual(emb_ori[i][j], emb_right[i][j])

  def testEmbeddingVariableForDRAMAndLEVELDB(self):
    print("testEmbeddingVariableForDRAMAndLEVELDB")
    def runTestAdagrad(self, var, g):
      #ids = array_ops.placeholder(dtypes.int64, name="ids")
      #emb = embedding_ops.embedding_lookup(var, ids)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 2, 3, 4, 5, 6, 7, 8, 9], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        for i in xrange(60):
          r, _, _ = sess.run([emb, train_op, loss])
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      db_directory = self.get_temp_dir()
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                                                                                                 storage_path=db_directory,
                                                                                                 storage_size=[4096])))
      emb1 = runTestAdagrad(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2", shape=[100, 30], initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 9):
        for j in range(0, 30):
          self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForDRAMAndSSD(self):
    print("testEmbeddingVariableForDRAMAndSSD")
    def runTestAdagrad(self, var, g):
      #ids = array_ops.placeholder(dtypes.int64, name="ids")
      #emb = embedding_ops.embedding_lookup(var, ids)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 2, 3, 4, 5, 6, 7, 8, 9], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        for i in xrange(60):
          r, _, _ = sess.run([emb, train_op, loss])
        r = sess.run(emb)
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      db_directory = self.get_temp_dir()
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                                                                                                 storage_path="/tmp/ssd_utpy",
                                                                                                 storage_size=[5120])))
      emb1 = runTestAdagrad(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2", shape=[100, 30], initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagrad(self, var, g)

    for i in range(0, 9):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForRecordFreq(self):
    print("testEmbeddingVariableForRecordFreq")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_RECORD_FREQ"] = "1"
    os.environ["TF_RECORD_VERSION"] = "1"
    emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(emb_var,
            math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
          self.assertEqual(ckpt_value.tolist()[2], 2)
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 1)
          self.assertEqual(ckpt_value.tolist()[1], 1)
          self.assertEqual(ckpt_value.tolist()[2], 1)
    os.environ["TF_RECORD_FREQ"] = "0"
    os.environ["TF_RECORD_VERSION"] = "0"

  def testEmbeddingVariableForRecordFreqWithCounterFilter(self):
    print("testEmbeddingVariableForRecordFreqWithCounterFilter")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_RECORD_FREQ"] = "1"
    os.environ["TF_RECORD_VERSION"] = "1"
    emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)))
    emb = embedding_ops.embedding_lookup(emb_var,  math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 1)
          self.assertEqual(ckpt_value.tolist()[1], 1)
    os.environ["TF_RECORD_FREQ"] = "0"
    os.environ["TF_RECORD_VERSION"] = "0"

'''
  @test_util.run_gpu_only
  def testEmbeddingVariableForHBMandDRAM(self):
    print("testEmbeddingVariableForHBMandDRAM")
    def runTestAdagrad(self, var, g):
      search_list = []
      for i in range(0, 10 * 8):
        search_list.append(i)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast(search_list, dtypes.int64))

      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)

      init = variables.global_variables_initializer()
      config = config_pb3.ConfigProto(log_device_placement=True)
      with self.test_session(graph=g, config=config, force_gpu=True) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r = sess.run(emb)
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 128,
          initializer=init_ops.ones_initializer(dtypes.float32),
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
          #steps_to_live=5,
          ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM_DRAM)))
      var = variable_scope.get_variable("var_2", shape=[1024, 128], initializer=init_ops.ones_initializer(dtypes.float32))

      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)
      print(emb1)
      print(emb2)

    for i in range(0, 6):
      for j in range(0, 3):
        self.assertEqual(emb1[i][j], emb2[i][j])
'''

if __name__ == "__main__":
  googletest.main()
