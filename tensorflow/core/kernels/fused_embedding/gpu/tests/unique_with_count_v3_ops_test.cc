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

class UniqueWithCountsV3OpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("unique_with_counts_v3", "UniqueWithCountsV3")
                     .Attr("CounterType", DT_INT32)
                     .Input(FakeInput(DT_INT64))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(UniqueWithCountsV3OpTest, UniqueWithCount) {
  MakeOpAndSetDevice(Device::GPU);
  const int input_size = 20;
  const int uniq_size = 14;
  // input
  AddInputFromArray<int64>(
      TensorShape({input_size}),
      {1, 3, 2, 1, 4, 5, 6, 5, 7, 8, 1, 9, 10, 2, 13, 15, 17, 13, 12, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  std::vector<int64> input = {1, 3, 2,  1, 4,  5,  6,  5,  7,  8,
                              1, 9, 10, 2, 13, 15, 17, 13, 12, 8};

  std::vector<int64> unique_keys;
  std::vector<int64> expected_unique_keys = {1, 2, 3,  4,  5,  6,  7,
                                             8, 9, 10, 12, 13, 15, 17};

  std::vector<int> expected_unique_counts = {3, 2, 1, 1, 2, 1, 1,
                                             2, 1, 1, 1, 2, 1, 1};

  auto unique_keys_tensor = GetOutput(0);
  auto unique_idxs_tensor = GetOutput(1);
  auto unique_counts_tenosr = GetOutput(2);

  // unique_idxs
  {
    test::internal::Expector<int64>::Equal(unique_idxs_tensor->dim_size(0),
                                           input_size);
    for (int i = 0; i < input_size; i++) {
      test::internal::Expector<int64>::Equal(
          unique_keys_tensor->flat<int64>()
              .data()[unique_idxs_tensor->flat<int>().data()[i]],
          input[i]);
    }
  }
  // unique_counts
  {
    for (int i = 0; i < uniq_size; i++) {
      const int count = expected_unique_counts[i];
      const int64 expected_key = expected_unique_keys[i];
      for (int j = 0; j < uniq_size; j++) {
        if (unique_keys_tensor->flat<int64>().data()[j] == expected_key) {
          test::internal::Expector<int64>::Equal(
              unique_counts_tenosr->flat<int>().data()[j], count);
        }
      }
    }
  }

  // test unique_keys
  {
    test::internal::Expector<int64>::Equal(unique_keys_tensor->dim_size(0),
                                           uniq_size);
    for (int i = 0; i < uniq_size; i++) {
      unique_keys.push_back(unique_keys_tensor->flat<int64>().data()[i]);
    }

    std::sort(unique_keys.begin(), unique_keys.end());

    for (int i = 0; i < uniq_size; i++) {
      test::internal::Expector<int64>::Equal(unique_keys[i],
                                             expected_unique_keys[i]);
    }
  }
}

}  // namespace
}  // namespace tensorflow