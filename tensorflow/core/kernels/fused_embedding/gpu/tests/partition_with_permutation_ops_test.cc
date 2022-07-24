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

class PartitionWithPermutationOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, const int num_partitions,
                          const std::string& partition_strategy) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(
        NodeDefBuilder("partition_with_permutation", "PartitionWithPermutation")
            .Attr("num_partitions", num_partitions)
            .Attr("partition_axis", 0)
            .Attr("partition_strategy", partition_strategy)
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(num_partitions, DT_INT64))
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(PartitionWithPermutationOpTest, Partition3Div) {
  MakeOpAndSetDevice(Device::GPU, 3, std::string("div"));
  // sp_values
  AddInputFromArray<int64>(TensorShape({12}),
                           {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {6, 16});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {3, 16});
  // partition_shapes 2
  AddInputFromArray<int64>(TensorShape({2}), {7, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // partitioned_values 0
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({6}));
    test::FillValues<int64>(&expected, {1, 5, 3, 0, 5, 5});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // partitioned_values 1
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected, {6 - 6, 7 - 6});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }
  // partitioned_values 2
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected, {12 - 9, 14 - 9, 15 - 9, 11 - 9});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(2));
  }

  // partition_permutation
  {
    Tensor expected(allocator(), DT_INT32, TensorShape({12, 2}));
    test::FillValues<int32>(&expected, {0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1,
                                        2, 2, 0, 3, 0, 4, 0, 5, 2, 3, 1, 1});
    test::ExpectTensorEqual<int32>(expected, *GetOutput(3));
  }
}

TEST_F(PartitionWithPermutationOpTest, Partition2Mod) {
  MakeOpAndSetDevice(Device::GPU, 2, std::string("mod"));
  // sp_values
  AddInputFromArray<int64>(TensorShape({12}),
                           {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {6, 16});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {6, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // partitioned_values 0
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected, {6 / 2, 12 / 2, 14 / 2, 0 / 2});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // partitioned_values 1
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({8}));
    test::FillValues<int64>(
        &expected, {1 / 2, 5 / 2, 3 / 2, 15 / 2, 5 / 2, 5 / 2, 11 / 2, 7 / 2});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }

  // partition_permutation
  {
    Tensor expected(allocator(), DT_INT32, TensorShape({12, 2}));
    test::FillValues<int32>(&expected, {1, 0, 1, 1, 1, 2, 0, 0, 0, 1, 0, 2,
                                        1, 3, 0, 3, 1, 4, 1, 5, 1, 6, 1, 7});
    test::ExpectTensorEqual<int32>(expected, *GetOutput(2));
  }
}

TEST_F(PartitionWithPermutationOpTest, Partition2ModEV) {
  MakeOpAndSetDevice(Device::GPU, 2, std::string("mod_ev"));
  // sp_values
  AddInputFromArray<int64>(TensorShape({6}), {5, 28, 1003, 2004, 1834, 17833});
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {10000, 8});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {10000, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // partitioned_values 0
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected, {28, 2004, 1834});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }
  // partitioned_values 1
  {
    Tensor expected(allocator(), DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected, {5, 1003, 17833});
    test::ExpectTensorEqual<int64>(expected, *GetOutput(1));
  }

  // partition_permutation
  {
    Tensor expected(allocator(), DT_INT32, TensorShape({6, 2}));
    test::FillValues<int32>(&expected, {1, 0, 0, 0, 1, 1, 0, 1, 0, 2, 1, 2});
    test::ExpectTensorEqual<int32>(expected, *GetOutput(2));
  }
}

}  // namespace
}  // namespace tensorflow
