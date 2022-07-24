#include <string>

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

class PruneInvalidAndFillEmptyRowsOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, const bool fill_empty_row,
                          const bool prune, const int default_id,
                          const bool use_sparse_weights,
                          const bool prune_sparse_weights,
                          const float default_weight) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("prune_invalid_and_fill_empty_rows",
                                "PruneInvalidAndFillEmptyRows")
                     .Attr("fill_empty_row", fill_empty_row)
                     .Attr("prune", prune)
                     .Attr("default_id", default_id)
                     .Attr("use_sparse_weights", use_sparse_weights)
                     .Attr("prune_sparse_weights", prune_sparse_weights)
                     .Attr("default_weight", default_weight)
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(PruneInvalidAndFillEmptyRowsOpTest, NothingHappend) {
  MakeOpAndSetDevice(Device::GPU, false, false, -1, false, false, 1.0);
  // sp_values
  AddInputFromArray<int64>(TensorShape({12}),
                           {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({12, 2}),
                           {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                            15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {16, 16});

  // sp_weights_values, even shape does not match sp_values, it doesn't matter
  AddInputFromArray<float>(TensorShape({1}), {1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // sp_values_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({12}));
    test::FillValues<int64>(&expected,
                            {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // sp_indices_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({12, 2}));
    test::FillValues<int64>(&expected,
                            {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                             15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }

  // sp_weights_values_out
  {
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
    test::FillValues<float>(&expected, {1.0});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
}

TEST_F(PruneInvalidAndFillEmptyRowsOpTest, PruneWithAllValid) {
  MakeOpAndSetDevice(Device::GPU, false, true, -1, true, false, 1.0);
  // sp_values
  AddInputFromArray<int64>(TensorShape({12}),
                           {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({12, 2}),
                           {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                            15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {16, 16});
  // sp_weights_values
  AddInputFromArray<float>(TensorShape({12}), {1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // sp_values_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({12}));
    test::FillValues<int64>(&expected,
                            {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // sp_indices_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({12, 2}));
    test::FillValues<int64>(&expected,
                            {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                             15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }
  // sp_weights_values_out
  {
    Tensor expected(allocator(), DT_FLOAT, TensorShape({12}));
    test::FillValues<float>(&expected, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
}

TEST_F(PruneInvalidAndFillEmptyRowsOpTest, FillEmptyRows) {
  MakeOpAndSetDevice(Device::GPU, true, false, -1, false, false, 3.0);
  // sp_values
  AddInputFromArray<int64>(TensorShape({4}), {1, 5, 3, 6});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({4, 2}), {0, 1, 0, 2, 2, 3, 3, 1});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {4, 8});
  // sp_weights_values
  AddInputFromArray<float>(TensorShape({4}), {1.0, 1.0, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // sp_values_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({5}));
    test::FillValues<int64>(&expected, {1, 5, 3, 6, 0});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // sp_indices_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({5, 2}));
    test::FillValues<int64>(&expected, {0, 1, 0, 2, 2, 3, 3, 1, 1, 0});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }
  // sp_weights_values_out, don't care
  // {
  //   Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  //   test::FillValues<float>(&expected, {1.0, 1.0, 1.0, 1.0, 3.0});
  //   test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  // }
  // is_row_empty
  {
    Tensor expected(allocator(), DT_BOOL, TensorShape({4}));
    test::FillValues<bool>(&expected, {false, true, false, false});
    test::ExpectTensorEqual<bool>(expected, *GetOutput(3));
  }
}

TEST_F(PruneInvalidAndFillEmptyRowsOpTest, PruneAndFillEmptyRowsWithDefaultId) {
  MakeOpAndSetDevice(Device::GPU, true, true, 8, true, false, 10.0);
  // sp_values
  AddInputFromArray<int64>(TensorShape({4}), {1, 5, 3, -5});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({4, 2}), {0, 1, 0, 2, 2, 3, 3, 1});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {4, 8});
  // sp_weights_values
  AddInputFromArray<float>(TensorShape({4}), {1.0, 2.0, 3.0, 4.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // sp_values_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({5}));
    test::FillValues<int64>(&expected, {1, 5, 3, 8, 8});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // sp_indices_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({5, 2}));
    test::FillValues<int64>(&expected, {0, 1, 0, 2, 2, 3, 1, 0, 3, 0});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }
  // sp_weights_values_out
  {
    Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
    test::FillValues<float>(&expected, {1.0, 2.0, 3.0, 10.0, 10.0});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  // is_row_empty
  {
    Tensor expected(allocator(), DT_BOOL, TensorShape({4}));
    test::FillValues<bool>(&expected, {false, true, false, true});
    test::ExpectTensorEqual<bool>(expected, *GetOutput(3));
  }
}

TEST_F(PruneInvalidAndFillEmptyRowsOpTest,
       PruneAndFillEmptyRowsWithDefaultIdWithSparseWeights) {
  MakeOpAndSetDevice(Device::GPU, true, true, 8, true, true, 10.0);
  // sp_values
  AddInputFromArray<int64>(TensorShape({4}), {1, 5, 3, -5});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({4, 2}), {0, 1, 0, 2, 2, 3, 3, 1});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {4, 8});
  // sp_weights
  AddInputFromArray<float>(TensorShape({4}), {-1.0, 2.0, 3.0, 4.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // sp_values_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected, {5, 3, 8, 8});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // sp_indices_out
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({4, 2}));
    test::FillValues<int64>(&expected, {0, 2, 2, 3, 1, 0, 3, 0});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }
  // sp_weights_values_out
  {
    Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&expected, {2.0, 3.0, 10.0, 10.0});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  // is_row_empty
  {
    Tensor expected(allocator(), DT_BOOL, TensorShape({4}));
    test::FillValues<bool>(&expected, {false, true, false, true});
    test::ExpectTensorEqual<bool>(expected, *GetOutput(3));
  }
}

}  // namespace
}  // namespace tensorflow