# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Iterator`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


class IteratorTest(test_base.DatasetTestBase, parameterized.TestCase):

  @test_util.deprecated_graph_mode_only
  def testNoGradients(self):
    component = constant_op.constant([1.])
    side = constant_op.constant(0.)
    add = lambda x: x + side
    dataset = dataset_ops.Dataset.from_tensor_slices(component).map(add)
    value = dataset_ops.make_one_shot_iterator(dataset).get_next()
    self.assertIsNone(gradients_impl.gradients(value, component)[0])
    self.assertIsNone(gradients_impl.gradients(value, side)[0])
    self.assertIsNone(gradients_impl.gradients(value, [component, side])[0])

  @test_util.deprecated_graph_mode_only
  def testCapturingStateInOneShotRaisesException(self):
    var = variables.Variable(37.0, name="myvar")
    dataset = (
        dataset_ops.Dataset.from_tensor_slices([0.0, 1.0, 2.0])
        .map(lambda x: x + var))
    with self.assertRaisesRegexp(
        ValueError, r"`Dataset.make_one_shot_iterator\(\)` does not support "
        "datasets that capture stateful objects.+myvar"):
      dataset_ops.make_one_shot_iterator(dataset)

  @test_util.deprecated_graph_mode_only
  def testOneShotIterator(self):
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = dataset_ops.make_one_shot_iterator(
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(14))
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      for _ in range(14):
        for i in range(7):
          result = sess.run(get_next)
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @test_util.deprecated_graph_mode_only
  def testOneShotIteratorCaptureByValue(self):
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    tensor_components = tuple([ops.convert_to_tensor(c) for c in components])

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = dataset_ops.make_one_shot_iterator(
        dataset_ops.Dataset.from_tensor_slices(tensor_components)
        .map(_map_fn).repeat(14))
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      for _ in range(14):
        for i in range(7):
          result = sess.run(get_next)
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testOneShotIteratorInsideContainer(self):
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def within_container():

      def _map_fn(x, y, z):
        return math_ops.square(x), math_ops.square(y), math_ops.square(z)

      iterator = dataset_ops.make_one_shot_iterator(
          dataset_ops.Dataset.from_tensor_slices(components)
          .map(_map_fn).repeat(14))
      return iterator.get_next()

    server = server_lib.Server.create_local_server()

    # Create two iterators within unique containers, and run them to
    # make sure that the resources aren't shared.
    #
    # The test below would fail if cname were the same across both
    # sessions.
    for j in range(2):
      with session.Session(server.target) as sess:
        cname = "iteration%d" % j
        with ops.container(cname):
          get_next = within_container()

        for _ in range(14):
          for i in range(7):
            result = sess.run(get_next)
            for component, result_component in zip(components, result):
              self.assertAllEqual(component[i]**2, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @test_util.deprecated_graph_mode_only
  def testOneShotIteratorNonBlocking(self):
    dataset = dataset_ops.Dataset.from_tensors([1, 2, 3]).map(lambda x: x * x)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()

    # Create a session with a single thread to ensure that the
    # one-shot iterator initializer does not deadlock.
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=1, use_per_session_threads=True)
    with session.Session(config=config) as sess:
      self.assertAllEqual([1, 4, 9], sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

    # Test with multiple threads invoking the one-shot iterator concurrently.
    with session.Session(config=config) as sess:
      results = []

      def consumer_thread():
        try:
          results.append(sess.run(next_element))
        except errors.OutOfRangeError:
          results.append(None)

      num_threads = 8
      threads = [
          self.checkedThread(consumer_thread) for _ in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      self.assertEqual(num_threads, len(results))
      self.assertEqual(num_threads - 1,
                       len([None for r in results if r is None]))
      self.assertAllEqual([[1, 4, 9]], [r for r in results if r is not None])

  @test_util.deprecated_graph_mode_only
  def testOneShotIteratorInitializerFails(self):
    # Define a dataset whose initialization will always fail.
    dataset = dataset_ops.Dataset.from_tensors(array_ops.gather([0], [4]))
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.InvalidArgumentError, ""):
        sess.run(next_element)

      # Test that subsequent attempts to use the iterator also fail.
      with self.assertRaisesRegexp(errors.InvalidArgumentError, ""):
        sess.run(next_element)

    with self.cached_session() as sess:

      def consumer_thread():
        with self.assertRaisesRegexp(errors.InvalidArgumentError, ""):
          sess.run(next_element)

      num_threads = 8
      threads = [
          self.checkedThread(consumer_thread) for _ in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

  @test_util.deprecated_graph_mode_only
  def testSimpleSharedResource(self):
    components = (np.array(1, dtype=np.int64),
                  np.array([1, 2, 3], dtype=np.int64),
                  np.array(37.0, dtype=np.float64))

    server = server_lib.Server.create_local_server()

    # Create two non-overlapping sessions that share the same iterator
    # resource on the same server, and verify that an action of the
    # first session (initializing the iterator) is visible in the
    # second session.
    with ops.Graph().as_default():
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.from_tensors(
              components).map(lambda x, y, z: (x, y, z)),
          shared_name="shared_iterator")
      init_op = iterator.initializer
      get_next = iterator.get_next()

      with session.Session(server.target) as sess:
        sess.run(init_op)
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

        # Re-initialize the iterator in the first session.
        sess.run(init_op)

    with ops.Graph().as_default():
      # Re-define the iterator manually, without defining any of the
      # functions in this graph, to ensure that we are not
      # accidentally redefining functions with the same names in the
      # new graph.
      iterator = iterator_ops.Iterator.from_structure(
          shared_name="shared_iterator",
          output_types=(dtypes.int64, dtypes.int64, dtypes.float64),
          output_shapes=([], [3], []))
      get_next = iterator.get_next()

      with session.Session(server.target) as sess:
        # Use the iterator without re-initializing in the second session.
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @test_util.deprecated_graph_mode_only
  def testNotInitializedError(self):
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_tensors(components))
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.FailedPreconditionError,
                                   "iterator has not been initialized"):
        sess.run(get_next)

  @test_util.deprecated_graph_mode_only
  def testReinitializableIterator(self):
    dataset_3 = dataset_ops.Dataset.from_tensors(
        constant_op.constant([1, 2, 3]))
    dataset_4 = dataset_ops.Dataset.from_tensors(
        constant_op.constant([4, 5, 6, 7]))
    iterator = iterator_ops.Iterator.from_structure(
        dataset_ops.get_legacy_output_types(dataset_3), [None])

    dataset_3_init_op = iterator.make_initializer(dataset_3)
    dataset_4_init_op = iterator.make_initializer(dataset_4)
    get_next = iterator.get_next()

    self.assertEqual(
        dataset_ops.get_legacy_output_types(dataset_3),
        dataset_ops.get_legacy_output_types(iterator))
    self.assertEqual(
        dataset_ops.get_legacy_output_types(dataset_4),
        dataset_ops.get_legacy_output_types(iterator))
    self.assertEqual(
        [None], dataset_ops.get_legacy_output_shapes(iterator).as_list())

    with self.cached_session() as sess:
      # The iterator is initially uninitialized.
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(get_next)

      # Initialize with one dataset.
      sess.run(dataset_3_init_op)
      self.assertAllEqual([1, 2, 3], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Initialize with a different dataset.
      sess.run(dataset_4_init_op)
      self.assertAllEqual([4, 5, 6, 7], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Reinitialize with the first dataset.
      sess.run(dataset_3_init_op)
      self.assertAllEqual([1, 2, 3], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @test_util.deprecated_graph_mode_only
  def testReinitializableIteratorWithFunctions(self):

    def g():
      for i in range(10):
        yield i

    iterator = iterator_ops.Iterator.from_structure(dtypes.int64, [])
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      dataset_1 = dataset_ops.Dataset.from_generator(
          g, output_types=dtypes.int64)
      sess.run(iterator.make_initializer(dataset_1))
      for expected in range(10):
        self.assertEqual(expected, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      dataset_2 = dataset_ops.Dataset.from_generator(
          g, output_types=dtypes.int64)
      sess.run(iterator.make_initializer(dataset_2))
      for expected in range(10):
        self.assertEqual(expected, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testReinitializableIteratorStaticErrors(self):
    # Non-matching structure for types and shapes.
    with self.assertRaises(TypeError):
      iterator = iterator_ops.Iterator.from_structure(
          (dtypes.int64, dtypes.float64), [None])

    # Test validation of dataset argument.
    iterator = iterator_ops.Iterator.from_structure((dtypes.int64,
                                                     dtypes.float64))

    # Incompatible structure.
    with self.assertRaises(ValueError):
      iterator.make_initializer(
          dataset_ops.Dataset.from_tensors(((constant_op.constant(
              [1, 2, 3], dtype=dtypes.int64),), (constant_op.constant(
                  [4., 5., 6., 7.], dtype=dtypes.float64),))))

    # Incompatible types.
    with self.assertRaises(TypeError):
      iterator.make_initializer(
          dataset_ops.Dataset.from_tensors(
              (constant_op.constant([1, 2, 3], dtype=dtypes.int32),
               constant_op.constant([4., 5., 6., 7.], dtype=dtypes.float32))))

    # Incompatible shapes.
    iterator = iterator_ops.Iterator.from_structure(
        (dtypes.int64, dtypes.float64), ([None], []))
    with self.assertRaises(TypeError):
      iterator.make_initializer(
          dataset_ops.Dataset.from_tensors(
              (constant_op.constant([1, 2, 3], dtype=dtypes.int64),
               constant_op.constant([4., 5., 6., 7.], dtype=dtypes.float64))))

  @test_util.deprecated_graph_mode_only
  def testIteratorStringHandle(self):
    dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    dataset_4 = dataset_ops.Dataset.from_tensor_slices([10, 20, 30, 40])

    iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
    iterator_4 = dataset_ops.make_one_shot_iterator(dataset_4)

    handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    feedable_iterator = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dataset_ops.get_legacy_output_types(dataset_3),
        dataset_ops.get_legacy_output_shapes(dataset_3))
    next_element = feedable_iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(dataset_3),
            dataset_ops.get_structure(feedable_iterator)))

    with self.cached_session() as sess:
      iterator_3_handle = sess.run(iterator_3.string_handle())
      iterator_4_handle = sess.run(iterator_4.string_handle())

      self.assertEqual(10,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      self.assertEqual(1,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_3_handle}))
      self.assertEqual(20,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      self.assertEqual(2,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_3_handle}))
      self.assertEqual(30,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      self.assertEqual(3,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_3_handle}))
      self.assertEqual(40,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            next_element, feed_dict={handle_placeholder: iterator_3_handle})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            next_element, feed_dict={handle_placeholder: iterator_4_handle})

  @test_util.deprecated_graph_mode_only
  def testIteratorStringHandleFuture(self):
    with forward_compat.forward_compatibility_horizon(2018, 8, 4):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      dataset_4 = dataset_ops.Dataset.from_tensor_slices([10, 20, 30, 40])

      iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
      iterator_4 = dataset_ops.make_one_shot_iterator(dataset_4)

      handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      feedable_iterator = iterator_ops.Iterator.from_string_handle(
          handle_placeholder, dataset_ops.get_legacy_output_types(dataset_3),
          dataset_ops.get_legacy_output_shapes(dataset_3))
      next_element = feedable_iterator.get_next()

      self.assertTrue(
          structure.are_compatible(
              dataset_ops.get_structure(dataset_3),
              dataset_ops.get_structure(feedable_iterator)))

      with self.cached_session() as sess:
        iterator_3_handle = sess.run(iterator_3.string_handle())
        iterator_4_handle = sess.run(iterator_4.string_handle())

        self.assertEqual(
            10,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        self.assertEqual(
            1,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_3_handle}))
        self.assertEqual(
            20,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        self.assertEqual(
            2,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_3_handle}))
        self.assertEqual(
            30,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        self.assertEqual(
            3,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_3_handle}))
        self.assertEqual(
            40,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(
              next_element, feed_dict={handle_placeholder: iterator_3_handle})
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(
              next_element, feed_dict={handle_placeholder: iterator_4_handle})

  @test_util.deprecated_graph_mode_only
  def testIteratorStringHandleReuseTensorObject(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    one_shot_iterator = dataset_ops.make_one_shot_iterator(dataset)
    initializable_iterator = dataset_ops.make_initializable_iterator(dataset)
    structure_iterator = iterator_ops.Iterator.from_structure(
        dataset_ops.get_legacy_output_types(dataset))

    created_ops = len(ops.get_default_graph().get_operations())

    self.assertIs(one_shot_iterator.string_handle(),
                  one_shot_iterator.string_handle())
    self.assertIs(initializable_iterator.string_handle(),
                  initializable_iterator.string_handle())
    self.assertIs(structure_iterator.string_handle(),
                  structure_iterator.string_handle())

    # Assert that getting the (default) string handle creates no ops.
    self.assertEqual(created_ops, len(ops.get_default_graph().get_operations()))

    # Specifying an explicit name will create a new op.
    handle_with_name = one_shot_iterator.string_handle(name="foo")
    self.assertEqual("foo", handle_with_name.op.name)
    self.assertIsNot(one_shot_iterator.string_handle(), handle_with_name)

    handle_with_same_name = one_shot_iterator.string_handle(name="foo")
    self.assertEqual("foo_1", handle_with_same_name.op.name)
    self.assertIsNot(handle_with_name, handle_with_same_name)

  @test_util.deprecated_graph_mode_only
  def testIteratorStringHandleError(self):
    dataset_int_scalar = (
        dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).repeat())
    dataset_float_vector = (dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0]))

    handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])

    feedable_int_scalar = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dtypes.int32, [])
    feedable_int_vector = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dtypes.int32, [None])
    feedable_int_any = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dtypes.int32)

    with self.cached_session() as sess:
      handle_int_scalar = sess.run(dataset_ops.make_one_shot_iterator(
          dataset_int_scalar).string_handle())
      handle_float_vector = sess.run(dataset_ops.make_one_shot_iterator(
          dataset_float_vector).string_handle())

      self.assertEqual(1,
                       sess.run(
                           feedable_int_scalar.get_next(),
                           feed_dict={handle_placeholder: handle_int_scalar}))

      self.assertEqual(2,
                       sess.run(
                           feedable_int_any.get_next(),
                           feed_dict={handle_placeholder: handle_int_scalar}))

      with self.assertRaises(errors.InvalidArgumentError):
        print(sess.run(
            feedable_int_vector.get_next(),
            feed_dict={handle_placeholder: handle_int_scalar}))

      with self.assertRaises(errors.InvalidArgumentError):
        print(sess.run(
            feedable_int_vector.get_next(),
            feed_dict={handle_placeholder: handle_float_vector}))

  @test_util.deprecated_graph_mode_only
  def testRemoteIteratorUsingRemoteCallOpDirectSession(self):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 3

    with ops.device("/job:localhost/replica:0/task:0/cpu:1"):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
      iterator_3_handle = iterator_3.string_handle()

    @function.Defun(dtypes.string)
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, dataset_ops.get_legacy_output_types(dataset_3),
          dataset_ops.get_legacy_output_shapes(dataset_3))
      return remote_iterator.get_next()

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      remote_op = functional_ops.remote_call(
          args=[iterator_3_handle],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target=target_placeholder)

    with self.session(config=worker_config) as sess:
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
          })
      self.assertEqual(elem, [1])
      # Fails when target is cpu:2 where the resource is not located.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(
            remote_op,
            feed_dict={
                target_placeholder: "/job:localhost/replica:0/task:0/cpu:2"
            })
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
          })
      self.assertEqual(elem, [2])
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
          })
      self.assertEqual(elem, [3])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            remote_op,
            feed_dict={
                target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
            })

  @test_util.deprecated_graph_mode_only
  def testRemoteIteratorUsingRemoteCallOpMultiWorkers(self):
    s1 = server_lib.Server.create_local_server()
    s2 = server_lib.Server.create_local_server()
    s3 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    workers = cluster_def.job.add()
    workers.name = "worker"
    workers.tasks[0] = s1.target[len("grpc://"):]
    workers.tasks[1] = s2.target[len("grpc://"):]
    client = cluster_def.job.add()
    client.name = "client"
    client.tasks[0] = s3.target[len("grpc://"):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    worker_devices = [
        "/job:worker/replica:0/task:%d/cpu:0" % i for i in range(2)
    ]
    itr_handles = []
    for device in worker_devices:
      with ops.device(device):
        src = dataset_ops.Dataset.from_tensor_slices([device])
        itr = dataset_ops.make_one_shot_iterator(src)
        itr_handles.append(itr.string_handle())

    targets = dataset_ops.Dataset.from_tensor_slices(worker_devices)
    handles = dataset_ops.Dataset.from_tensor_slices(itr_handles)

    @function.Defun(dtypes.string)
    def loading_func(h):
      remote_itr = iterator_ops.Iterator.from_string_handle(
          h, dataset_ops.get_legacy_output_types(itr),
          dataset_ops.get_legacy_output_shapes(itr))
      return remote_itr.get_next()

    def map_fn(target, handle):
      return functional_ops.remote_call(
          args=[handle], Tout=[dtypes.string], f=loading_func, target=target)

    with ops.device("/job:client"):
      client_dataset = dataset_ops.Dataset.zip((targets, handles)).map(map_fn)
      itr = dataset_ops.make_initializable_iterator(client_dataset)
      n = itr.get_next()

    with session.Session(s3.target, config=config) as sess:
      sess.run(itr.initializer)
      expected_values = worker_devices
      for expected in expected_values:
        self.assertEqual((compat.as_bytes(expected),), sess.run(n))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(n)

  @test_util.deprecated_graph_mode_only
  def testRemoteIteratorUsingRemoteCallOpDirectSessionGPUCPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
      iterator_3_handle = iterator_3.string_handle()

    def _encode_raw(byte_array):
      return bytes(bytearray(byte_array))

    @function.Defun(dtypes.uint8)
    def _remote_fn(h):
      handle = script_ops.py_func(_encode_raw, [h], dtypes.string)
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          handle, dataset_ops.get_legacy_output_types(dataset_3),
          dataset_ops.get_legacy_output_shapes(dataset_3))
      return remote_iterator.get_next()

    with ops.device("/job:localhost/replica:0/task:0/device:GPU:0"):
      target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      iterator_3_handle_uint8 = parsing_ops.decode_raw(
          input_bytes=iterator_3_handle, out_type=dtypes.uint8)
      remote_op = functional_ops.remote_call(
          args=[iterator_3_handle_uint8],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target=target_placeholder)

    with self.cached_session() as sess:
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
          })
      self.assertEqual(elem, [1])
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
          })
      self.assertEqual(elem, [2])
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
          })
      self.assertEqual(elem, [3])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            remote_op,
            feed_dict={
                target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
            })

  @test_util.deprecated_graph_mode_only
  def testRepeatedGetNextWarning(self):
    iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.range(10))
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      for _ in range(100):
        iterator.get_next()
    self.assertEqual(100 - iterator_ops.GET_NEXT_CALL_WARNING_THRESHOLD, len(w))
    for warning in w:
      self.assertIn(
          iterator_ops.GET_NEXT_CALL_WARNING_MESSAGE, str(warning.message))

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      ("Tensor", lambda: constant_op.constant(37.0),
       tensor_spec.TensorSpec([],
                              dtypes.float32), ops.Tensor, dtypes.float32, []),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[0]],
          values=constant_op.constant([0], dtype=dtypes.int32),
          dense_shape=[1]), sparse_tensor.SparseTensorSpec([1], dtypes.int32),
       sparse_tensor.SparseTensor, dtypes.int32, [1]),
      ("Nest", lambda: {
          "a": constant_op.constant(37.0),
          "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))
      }, {
          "a":
              tensor_spec.TensorSpec([], dtypes.float32),
          "b": (tensor_spec.TensorSpec(
              [1], dtypes.string), tensor_spec.TensorSpec([], dtypes.string))
      }, {
          "a": ops.Tensor,
          "b": (ops.Tensor, ops.Tensor)
      }, {
          "a": dtypes.float32,
          "b": (dtypes.string, dtypes.string)
      }, {
          "a": [],
          "b": ([1], [])
      }),
  )
  def testIteratorStructure(self, tf_value_fn, expected_element_structure,
                            expected_output_classes, expected_output_types,
                            expected_output_shapes):
    tf_value = tf_value_fn()
    iterator = dataset_ops.make_one_shot_iterator(
        dataset_ops.Dataset.from_tensors(tf_value))

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(iterator), expected_element_structure))
    self.assertEqual(expected_output_classes,
                     dataset_ops.get_legacy_output_classes(iterator))
    self.assertEqual(expected_output_types,
                     dataset_ops.get_legacy_output_types(iterator))
    self.assertEqual(expected_output_shapes,
                     dataset_ops.get_legacy_output_shapes(iterator))

  def testIteratorGetNextName(self):
    with ops.Graph().as_default():
      iterator = dataset_ops.make_one_shot_iterator(
          dataset_ops.Dataset.from_tensors(37.0))
      next_element = iterator.get_next(name="overridden_name")
      self.assertEqual("overridden_name", next_element.op.name)

  @parameterized.named_parameters(
      ("Async", context.ASYNC),
      ("Sync", context.SYNC),
  )
  def testIteratorEagerIteration(self, execution_mode):
    with context.eager_mode(), context.execution_mode(execution_mode):
      val = 0
      dataset = dataset_ops.Dataset.range(10)
      iterator = iter(dataset)
      for foo in iterator:
        self.assertEqual(val, foo.numpy())
        val += 1

  @test_util.run_v2_only
  def testIteratorV2Function(self):

    queue = data_flow_ops.FIFOQueue(10, dtypes.int64)

    @def_function.function
    def fn():
      dataset = dataset_ops.Dataset.range(10)
      iterator = iter(dataset)
      for _ in range(10):
        queue.enqueue(next(iterator))

    fn()

    for i in range(10):
      self.assertEqual(queue.dequeue().numpy(), i)

  @test_util.run_v2_only
  def testIteratorV2FunctionError(self):
    # In this test we verify that a function that raises an error ends up
    # properly deallocating the iterator resource.

    queue = data_flow_ops.FIFOQueue(10, dtypes.int64)
    queue.enqueue(0)

    def init_fn(n):
      return n

    def next_fn(_):
      ds = dataset_ops.Dataset.range(0)
      return next(iter(ds))

    def finalize_fn(n):
      queue.enqueue(0)
      return n

    @def_function.function
    def fn():
      dataset = dataset_ops._GeneratorDataset(1, init_fn, next_fn, finalize_fn)
      iterator = iter(dataset)
      next(iterator)

    with self.assertRaises(errors.OutOfRangeError):
      fn()

    self.assertEqual(queue.size().numpy(), 2)

  @test_util.run_v2_only
  def testLimitedRetracing(self):
    trace_count = [0]

    @def_function.function
    def f(iterator):
      trace_count[0] += 1
      counter = np.int64(0)
      for elem in iterator:
        counter += elem
      return counter

    dataset = dataset_ops.Dataset.range(5)
    dataset2 = dataset_ops.Dataset.range(10)

    for _ in range(10):
      self.assertEqual(self.evaluate(f(iter(dataset))), 10)
      self.assertEqual(self.evaluate(f(iter(dataset2))), 45)
      self.assertEqual(trace_count[0], 1)

  def assert_dataset_placement(self, host_dataset, host_iterator, host_tensor,
                               device_dataset, device_iterator, device_tensor):

    self.assertTrue(
        "cpu:0" in host_dataset._variant_tensor.device.lower() or
        host_dataset._variant_tensor.device == ""
    )
    self.assertTrue(
        "cpu:0" in host_iterator._iterator_resource.device.lower() or
        host_iterator._iterator_resource.device == ""
    )
    self.assertTrue(
        "cpu:0" in host_tensor.device.lower() or host_tensor.device == ""
    )

    self.assertIn("gpu:0", device_dataset._variant_tensor.device.lower())
    self.assertIn("gpu:0", device_iterator._iterator_resource.device.lower())
    self.assertIn("gpu:0", device_tensor.device.lower())

#  @test_util.deprecated_graph_mode_only
#  def testIteratorOnDeviceGraphModeOneShotIterator(self):
#    if not test_util.is_gpu_available():
#      self.skipTest("No GPU available")
#
#    host_dataset = dataset_ops.Dataset.range(10)
#    device_dataset = host_dataset.apply(
#        prefetching_ops.prefetch_to_device("/gpu:0"))
#
#    host_iterator = dataset_ops.make_one_shot_iterator(host_dataset)
#    device_iterator = dataset_ops.make_one_shot_iterator(device_dataset)
#
#    host_tensor = host_iterator.get_next()
#    device_tensor = device_iterator.get_next()
#
#    self.assert_dataset_placement(
#        host_dataset, host_iterator, host_tensor,
#        device_dataset, device_iterator, device_tensor
#    )
#
#  @test_util.deprecated_graph_mode_only
#  def testIteratorOnDeviceGraphModeInitializableIterator(self):
#    if not test_util.is_gpu_available():
#      self.skipTest("No GPU available")
#
#    host_dataset = dataset_ops.Dataset.range(10)
#    device_dataset = host_dataset.apply(
#        prefetching_ops.prefetch_to_device("/gpu:0"))
#
#    host_iterator = dataset_ops.make_initializable_iterator(host_dataset)
#    device_iterator = dataset_ops.make_initializable_iterator(device_dataset)
#
#    host_tensor = host_iterator.get_next()
#    device_tensor = device_iterator.get_next()
#
#    self.assert_dataset_placement(
#        host_dataset, host_iterator, host_tensor,
#        device_dataset, device_iterator, device_tensor
#    )


if __name__ == "__main__":
  test.main()
