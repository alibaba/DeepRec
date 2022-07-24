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

class FusedEmbeddingSparsePostLookUpV2GradOpTest : public OpsTestBase {
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

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_sparse_post_look_up_v2_grad",
                                "FusedEmbeddingSparsePostLookUpV2Grad")
                     .Attr("T", dtype)
                     .Attr("num_partitions", num_partitions)
                     .Attr("partition_axis", 0)
                     .Attr("combiner", combiner)
                     .Attr("max_norm", max_norm)
                     .Attr("fill_empty_row", fill_empty_row)
                     .Attr("default_id", default_id)
                     .Attr("use_sparse_weights", use_sparse_weights)
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(num_partitions, dtype))
                     .Input(FakeInput(DT_UINT64))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_BOOL))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedEmbeddingSparsePostLookUpV2GradOpTest, Partition2MeanMaxNorm100) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 8;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "mean", 100.0, false, -1, false);

  // top_grad
  AddInputFromArray<float>(
      TensorShape({batch_size, emb_vector_dim}),
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
       11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
       22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});

  // emb_shards 0
  AddInputFromArray<float>(
      TensorShape({6, emb_vector_dim}),
      {8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0, 26.0, 27.0,
       28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
       32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 32.0, 33.0, 34.0, 35.0,
       36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0});

  // make same input to dump to emb_shard_ptrs
  Tensor emb_shards_0(allocator(), DT_FLOAT, TensorShape({6, emb_vector_dim}));
  test::FillValues<float>(
      &emb_shards_0,
      {8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0, 26.0, 27.0,
       28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
       32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 32.0, 33.0, 34.0, 35.0,
       36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0});

  // emb_shards 1
  AddInputFromArray<float>(
      TensorShape({4, emb_vector_dim}),
      {56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
       96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  // make same input to dump to emb_shard_ptrs
  Tensor emb_shards_1(allocator(), DT_FLOAT, TensorShape({4, emb_vector_dim}));
  test::FillValues<float>(
      &emb_shards_1, {56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
                      96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
                      96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
                      120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  // emb_shard_ptrs
  AddInputFromArray<uint64>(TensorShape({2}),
                            {reinterpret_cast<uint64>(emb_shards_0.data()),
                             reinterpret_cast<uint64>(emb_shards_1.data())});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({10, 2}), {0, 0, 1, 0, 0, 1, 1, 1, 0, 2,
                                                1, 2, 0, 3, 1, 3, 0, 4, 0, 5});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 3, 3, 2});

  // indices_before_unique
  AddInputFromArray<int64>(
      TensorShape({nnz, 2}),
      {0, 5, 1, 7, 0, 1, 2, 4, 2, 1, 2, 7, 1, 2, 3, 0, 3, 6, 1, 1});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz}), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, false, false});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor grad_shards_1(allocator(), DT_FLOAT,
                         TensorShape({6, emb_vector_dim}));
    test::FillValues<float>(
        &grad_shards_1,
        {0.00000000,  0.50000000,  1.00000000,  1.50000000,  2.00000000,
         2.50000000,  3.00000000,  3.50000000,  0.00000000,  0.50000000,
         1.00000000,  1.50000000,  2.00000000,  2.50000000,  3.00000000,
         3.50000000,  5.33333349,  5.66666651,  6.00000000,  6.33333349,
         6.66666651,  7.00000000,  7.33333349,  7.66666651,  2.65028572,
         2.98157120,  3.31285667,  3.64414287,  3.97542834,  4.30671406,
         4.63799953,  4.96928549,  11.92628479, 12.42321396, 12.92014217,
         13.41707039, 13.91399956, 14.41092777, 14.90785599, 15.40478516,
         2.16437674,  2.43492365,  2.70547056,  2.97601795,  3.24656487,
         3.51711202,  3.78765893,  4.05820608});
    test::ExpectTensorNear<float>(grad_shards_1, *GetOutput(0), 1e-4);
  }

  {
    Tensor grad_shards_2(allocator(), DT_FLOAT,
                         TensorShape({4, emb_vector_dim}));
    test::FillValues<float>(
        &grad_shards_2,
        {1.58337951, 1.78130186, 1.97922409, 2.17714667, 2.37506914, 2.57299161,
         2.77091384, 2.96883631, 1.89459133, 2.01300311, 2.13141513, 2.24982715,
         2.36823893, 2.48665094, 2.60506320, 2.72347474, 1.89459133, 2.01300311,
         2.13141513, 2.24982715, 2.36823893, 2.48665094, 2.60506320, 2.72347474,
         3.43474555, 3.57786012, 3.72097445, 3.86408877, 4.00720310, 4.15031767,
         4.29343224, 4.43654633});
    test::ExpectTensorNear<float>(grad_shards_2, *GetOutput(1), 1e-4);
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2GradOpTest, Partition2SUMUnique) {
  const int nnz = 6;
  const int batch_size = 4;
  const int emb_vector_dim = 1;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "sum", -1.0, true, -1, false);

  // top_grad
  AddInputFromArray<float>(TensorShape({batch_size, emb_vector_dim}),
                           {1.0, 2.0, 3.0, 4.0});

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({3, emb_vector_dim}), {4.0, 5.0, 6.0});
  // make same input to dump to emb_shard_ptrs
  Tensor emb_shards_0(allocator(), DT_FLOAT, TensorShape({3, emb_vector_dim}));
  test::FillValues<float>(&emb_shards_0, {4.0, 5.0, 6.0});

  // emb_shards 1
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}), {6.0, 7.0});
  Tensor emb_shards_1(allocator(), DT_FLOAT, TensorShape({2, emb_vector_dim}));
  test::FillValues<float>(&emb_shards_1, {6.0, 7.0});

  // emb_shard_ptrs
  AddInputFromArray<uint64>(TensorShape({2}),
                            {reinterpret_cast<uint64>(emb_shards_0.data()),
                             reinterpret_cast<uint64>(emb_shards_1.data())});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({5, 2}), {1, 1, 0, 2, 0, 0, 1, 0, 0, 1});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 2, 1, 2});

  // values after fill empty: 1, 1, 2, 3, 4, 2, 0
  // after unique 1, 2, 3, 4, 0

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 3, 1, 2, 1, 3, 3, 2, 3, 6, 2, 0});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 0, 1, 2, 3, 1, 4});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    /*
      permute 2 -> unique_counts: 1, unique_offsets: 4 ->
      idx_of_input_to_unique: 3 -> batch: 1, grad: 2.0

      permute 4 -> unique_counts: 1, unique_offsets: 6 ->
      idx_of_input_to_unique: 6 -> batch: 2: grad: 0.0, because fill_empty
      row

      permute 1 -> unique_counts: 2, unique_offsets: 2
        -> idx_of_input_to_unique: 2 -> batch: 1 -> grad : 2.0
        -> idx_of_input_to_unique: 5 -> batch: 3 -> grad : 4.0
      sum grad: 6.0
    */
    Tensor grad_shards_1(allocator(), DT_FLOAT,
                         TensorShape({3, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_1, {2.0, 0.0, 6.0});
    test::ExpectTensorNear<float>(grad_shards_1, *GetOutput(0), 1e-4);
  }

  {
    /*
      permute 3 -> unique_counts: 1, unique_offsets: 5 ->
      idx_of_input_to_unique: 4 -> batch: 3 -> grad: 4.0

      permute 0 -> unique_counts: 2, unique_offsets: 0
        -> idx_of_input_to_unique 0 -> batch: 0 -> grad: 1.0
        -> idx_of_input_to_unique 1 -> batch: 0 -> grad: 1.0
      sum grad: 2.0
    */
    Tensor grad_shards_2(allocator(), DT_FLOAT,
                         TensorShape({2, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_2, {4.0, 2.0});
    test::ExpectTensorNear<float>(grad_shards_2, *GetOutput(1), 1e-4);
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2GradOpTest,
       Partition2SUMUniqueDefault4) {
  const int nnz = 6;
  const int batch_size = 4;
  const int emb_vector_dim = 1;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "sum", -1.0, true, 4, false);

  // top_grad
  AddInputFromArray<float>(TensorShape({batch_size, emb_vector_dim}),
                           {1.0, 2.0, 3.0, 4.0});

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({3, emb_vector_dim}), {4.0, 5.0, 6.0});
  // make same input to dump to emb_shard_ptrs
  Tensor emb_shards_0(allocator(), DT_FLOAT, TensorShape({3, emb_vector_dim}));
  test::FillValues<float>(&emb_shards_0, {4.0, 5.0, 6.0});

  // emb_shards 1
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}), {6.0, 7.0});
  Tensor emb_shards_1(allocator(), DT_FLOAT, TensorShape({2, emb_vector_dim}));
  test::FillValues<float>(&emb_shards_1, {6.0, 7.0});

  // emb_shard_ptrs
  AddInputFromArray<uint64>(TensorShape({2}),
                            {reinterpret_cast<uint64>(emb_shards_0.data()),
                             reinterpret_cast<uint64>(emb_shards_1.data())});

  // partition_permutation
  AddInputFromArray<int>(TensorShape({5, 2}), {1, 1, 0, 2, 0, 0, 1, 0, 0, 1});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 2, 1, 2});

  // values after fill empty: 1, 1, 2, 3, 4, 2, 0
  // after unique 1, 2, 3, 4, 0

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 3, 1, 2, 1, 3, 3, 2, 3, 6, 2, 0});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 0, 1, 2, 3, 1, 4});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    /*
      permute 2 -> unique_counts: 1, unique_offsets: 4 ->
      idx_of_input_to_unique: 3 -> batch: 1, grad: 2.0

      permute 4 -> unique_counts: 1, unique_offsets: 6 ->
      idx_of_input_to_unique: 6 -> batch: 2: grad: 3.0

      permute 1 -> unique_counts: 2, unique_offsets: 2
        -> idx_of_input_to_unique: 2 -> batch: 1 -> grad : 2.0
        -> idx_of_input_to_unique: 5 -> batch: 3 -> grad : 4.0
      sum grad: 6.0
    */
    Tensor grad_shards_1(allocator(), DT_FLOAT,
                         TensorShape({3, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_1, {2.0, 3.0, 6.0});
    test::ExpectTensorNear<float>(grad_shards_1, *GetOutput(0), 1e-4);
  }

  {
    /*
      permute 3 -> unique_counts: 1, unique_offsets: 5 ->
      idx_of_input_to_unique: 4 -> batch: 3 -> grad: 4.0

      permute 0 -> unique_counts: 2, unique_offsets: 0
        -> idx_of_input_to_unique 0 -> batch: 0 -> grad: 1.0
        -> idx_of_input_to_unique 1 -> batch: 0 -> grad: 1.0
      sum grad: 2.0
    */
    Tensor grad_shards_2(allocator(), DT_FLOAT,
                         TensorShape({2, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_2, {4.0, 2.0});
    test::ExpectTensorNear<float>(grad_shards_2, *GetOutput(1), 1e-4);
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpV2GradOpTest, SinglePartitionSUMUnique) {
  const int nnz = 6;
  const int batch_size = 4;
  const int emb_vector_dim = 1;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 1, DT_FLOAT, "sum", -1.0, true, -1, false);

  // top_grad
  AddInputFromArray<float>(TensorShape({batch_size, emb_vector_dim}),
                           {1.0, 2.0, 3.0, 4.0});

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({5, emb_vector_dim}),
                           {7.0, 6.0, 4.0, 6.0, 5.0});

  // emb_shard_ptrs, whatever, will not be used
  AddInputFromArray<uint64>(TensorShape({1}), {0});

  // partition_permutation, whatever, will not be used
  AddInputFromArray<int>(TensorShape({1, 1}), {1});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 2, 1, 2});

  // values after fill empty: 1, 1, 2, 3, 4, 2, 0
  // after unique 1, 2, 3, 4, 0

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 3, 1, 2, 1, 3, 3, 2, 3, 6, 2, 0});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 0, 1, 2, 3, 1, 4});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  Tensor grad_shards_0(allocator(), DT_FLOAT, TensorShape({5, emb_vector_dim}));
  test::FillValues<float>(&grad_shards_0, {2.0, 6.0, 2.0, 4.0, 0.0});
  test::ExpectTensorNear<float>(grad_shards_0, *GetOutput(0), 1e-4);
}

TEST_F(FusedEmbeddingSparsePostLookUpV2GradOpTest,
       SinglePartitionSUMUniqueSparseWeight) {
  const int nnz = 6;
  const int batch_size = 4;
  const int emb_vector_dim = 1;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 1, DT_FLOAT, "sum", -1.0, true, -1, true);

  // top_grad
  AddInputFromArray<float>(TensorShape({batch_size, emb_vector_dim}),
                           {1.0, 1.0, 1.0, 1.0});

  // emb_shards 0
  AddInputFromArray<float>(TensorShape({5, emb_vector_dim}),
                           {7.0, 6.0, 4.0, 6.0, 5.0});

  // emb_shard_ptrs, whatever, will not be used
  AddInputFromArray<uint64>(TensorShape({1}), {0});

  // partition_permutation, whatever, will not be used
  AddInputFromArray<int>(TensorShape({1, 1}), {1});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 2, 1, 2});

  // values after fill empty: 1, 1, 2, 3, 4, 2, 0
  // after unique 1, 2, 3, 4, 0

  // indices_before_unique
  AddInputFromArray<int64>(TensorShape({nnz + 1, 2}),
                           {0, 1, 0, 3, 1, 2, 1, 3, 3, 2, 3, 6, 2, 0});

  // unique_idxs
  AddInputFromArray<int>(TensorShape({nnz + 1}), {0, 0, 1, 2, 3, 1, 4});

  // is_row_empty
  AddInputFromArray<bool>(TensorShape({batch_size}),
                          {false, false, true, false});

  // sp_weights_values
  AddInputFromArray<float>(TensorShape({nnz + 1}),
                           {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  Tensor grad_shards_0(allocator(), DT_FLOAT, TensorShape({5, emb_vector_dim}));
  test::FillValues<float>(&grad_shards_0, {3.0, 9.0, 4.0, 5.0, 0.0});
  test::ExpectTensorNear<float>(grad_shards_0, *GetOutput(0), 1e-4);
}

}  // namespace
}  // namespace tensorflow