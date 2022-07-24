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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

enum class Device { GPU };
class FusedEmbeddingSparsePostLookUpV2OpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, int num_partitions, DataType dtype,
                          const std::string& combiner, const float max_norm,
                          const bool fill_empty_row, const int default_id,
                          const bool use_sparse_weights) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_sparse_post_look_up_v2",
                                "FusedEmbeddingSparsePostLookUpV2")
                     .Attr("T", dtype)
                     .Attr("num_partitions", num_partitions)
                     .Attr("partition_axis", 0)
                     .Attr("combiner", combiner)
                     .Attr("max_norm", max_norm)
                     .Attr("fill_empty_row", fill_empty_row)
                     .Attr("default_id", default_id)
                     .Attr("use_sparse_weights", use_sparse_weights)
                     .Input(FakeInput(num_partitions, dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_BOOL))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedEmbeddingSparsePostLookUpV2OpTest,
       Partition3CombinerSqrtnMaxNorm200) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 8;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 3, DT_FLOAT, "sqrtn", 200.0, false, -1,
                     false);

  // emb_shards 0
  AddInputFromArray<float>(
      TensorShape({6, emb_vector_dim}),
      {
          8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0,
          26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0,
          28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
          38.0, 39.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
          40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
      });
  // emb_shards 1
  AddInputFromArray<float>(TensorShape({1, emb_vector_dim}),
                           {56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0});
  // emb_shards 2
  AddInputFromArray<float>(
      TensorShape({3, emb_vector_dim}),
      {96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({nnz, 2}), {0, 0, 0, 1, 0, 2, 0, 3, 0, 4,
                                                 0, 5, 1, 0, 2, 0, 2, 1, 2, 2});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // indices_before_unique
  AddInputFromArray<int64>(
      TensorShape({nnz, 2}),
      {0, 5, 0, 1, 2, 1, 1, 2, 3, 6, 1, 1, 1, 7, 2, 4, 2, 7, 3, 0});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, false, false});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz}), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {22.62741661, 24.04163170, 25.45584488,  26.87005806,  28.28427124,
         29.69848442, 31.11269951, 32.52691269,  73.90083313,  75.63288879,
         77.36493683, 79.09698486, 80.82904053,  82.56108856,  84.29314423,
         86.02519226, 92.61308289, 94.01081848,  95.40855408,  96.80628204,
         98.20401764, 99.60175323, 100.99948120, 102.39721680, 71.20205688,
         72.31395721, 73.42584991, 74.53774261,  75.64963531,  76.76153564,
         77.87342834, 78.98532867});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 3, 3, 2});
    test::ExpectTensorEqual<int>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2OpTest, Partition2SumFillEmpty) {
  const int nnz = 3;
  const int batch_size = 3;
  const int emb_vector_dim = 4;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "sum", -1.0, true, -1, false);

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0});
  // emb_shards 1
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({nnz + 1, 2}), {1, 1, 0, 0, 0, 1, 1, 0});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}), {2, 0, 0, 0, 0, 5, 1, 4});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}), {false, false, true});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 1, 2, 3});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {3.0, 3.0, 3.0, 3.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 1, 0});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2OpTest, Partition2SumFillEmptyDefault2) {
  const int nnz = 3;
  const int batch_size = 3;
  const int emb_vector_dim = 4;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "sum", -1.0, true, 2, false);

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0});
  // emb_shards 1
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({nnz + 1, 2}), {1, 1, 0, 0, 0, 1, 1, 0});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}), {2, 0, 0, 0, 0, 5, 1, 4});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}), {false, false, true});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 1, 2, 3});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {3.0, 3.0, 3.0, 3.0, 10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 1, 1});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2OpTest,
       Partition2MeanFillEmptyDefault2Unique) {
  const int nnz = 7;
  const int batch_size = 5;
  const int emb_vector_dim = 2;
  const int entries = 4;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "mean", -1.0, true, 2, false);

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {1.0, 1.0, 2.0, 2.0});
  // emb_shards 1
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {3.0, 3.0, 4.0, 4.0});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({4, 2}), {1, 0, 0, 1, 1, 1, 0, 0});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 2, 1, 0, 1, 1, 3, 0, 3, 1, 4, 0, 2, 0});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false, false});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 1, 1, 0, 2, 3, 3, 3});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  // {3 + 2, 3 + 2, 3 + 2, 3 + 2, 1, 1, 4 + 1, 4 + 1, 1, 1}
  // {2, 2, 1, 2, 1}

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 2.5, 2.5, 1.0, 1.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 2, 1, 2, 1});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2OpTest,
       SinglePartitionMeanFillEmptyDefault2Unique) {
  const int nnz = 7;
  const int batch_size = 5;
  const int emb_vector_dim = 2;
  const int entries = 4;

  MakeOpAndSetDevice(Device::GPU, 1, DT_FLOAT, "mean", -1.0, true, 2, false);

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({4, emb_vector_dim}),
                           {3.0, 3.0, 2.0, 2.0, 4.0, 4.0, 1.0, 1.0});

  // partition_permutation, whatever, will not use this
  AddInputFromArray<int>(TensorShape({1, 1}), {1});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 2, 1, 0, 1, 1, 3, 0, 3, 1, 4, 0, 2, 0});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false, false});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 1, 1, 0, 2, 3, 3, 3});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  // {3 + 2, 3 + 2, 3 + 2, 3 + 2, 1, 1, 4 + 1, 4 + 1, 1, 1}
  // {2, 2, 1, 2, 1}

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 2.5, 2.5, 1.0, 1.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 2, 1, 2, 1});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2OpTest,
       SinglePartitionMeanFillEmptyDefault2UniqueSparseWeights) {
  const int nnz = 7;
  const int batch_size = 5;
  const int emb_vector_dim = 2;
  const int entries = 4;

  MakeOpAndSetDevice(Device::GPU, 1, DT_FLOAT, "sum", -1.0, true, 2, true);

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({4, emb_vector_dim}),
                           {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  // partition_permutation, whatever, will not use this
  AddInputFromArray<int>(TensorShape({1, 1}), {1});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 2, 1, 0, 1, 1, 3, 0, 3, 1, 4, 0, 2, 0});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false, false});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 1, 1, 0, 2, 3, 3, 3});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({nnz + 1}),
                           {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {3.0, 3.0, 7.0, 7.0, 8.0, 8.0, 11.0, 11.0, 7.0, 7.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 2, 1, 2, 1});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

}  // namespace
}  // namespace tensorflow