/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

// Subclass of DeviceResolverDistributed which behaves identically but
// allows access to the attr_table_.
class TestableDeviceResolverDistributed : public DeviceResolverDistributed {
 public:
  TestableDeviceResolverDistributed(const DeviceMgr* dev_mgr,
                                    WorkerCacheInterface* worker_cache,
                                    const string& task)
      : DeviceResolverDistributed(dev_mgr, worker_cache, task) {}

  absl::flat_hash_map<string, DeviceAttributes>& attr_table() {
    return attr_table_;
  }
};

// Create a fake 'Device' whose only interesting attribute is a non-default
// DeviceLocality and incarnation.
static std::unique_ptr<Device> NewDevice(const string& type, const string& name,
                                         int numa_node, uint64 incarnation) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  attr.mutable_locality()->set_numa_node(numa_node);
  attr.set_incarnation(incarnation);
  return absl::make_unique<FakeDevice>(attr);
}

// Create a fake WorkerInterface that responds to requests without RPCs,
// in this case returning the DeviceAttributes of a fake remote worker.
class FakeWorker : public TestWorkerInterface {
 public:
  FakeWorker(const string& name, DeviceMgr* dev_mgr,
             DeviceResolverDistributed* dres)
      : name_(name), device_mgr_(dev_mgr), device_resolver_(dres) {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    std::vector<DeviceAttributes> dev_attr;
    device_mgr_->ListDeviceAttributes(&dev_attr);
    for (const auto& da : dev_attr) {
      *response->add_device_attributes() = da;
    }
    done(Status::OK());
  }
  
  void GetStatusAsyncWithOptions(const GetStatusRequest* request,
                                 GetStatusResponse* response,
                                 StatusCallback done,
                                 CallOptions* call_opts) override {
    GetStatusAsync(request, response, done);
  }

 private:
  string name_;
  DeviceMgr* device_mgr_;
  DeviceResolverDistributed* device_resolver_;
};

// An implementation of WorkerCacheInterface that routes all requests
// to local FakeWorkers, implementing only the methods needed for tests.
class FakeCache : public TestWorkerCache {
 public:
  // Override the Locality methods to actually pass through to the
  // worker.
  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
    return false;
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
    string task_name;
    string dev_part;
    if (!DeviceNameUtils::SplitDeviceName(device, &task_name, &dev_part)) {
      done(errors::Internal("failed to parse device name"));
      return;
    }
    auto it = workers_.find(task_name);
    if (it == workers_.end()) {
      done(errors::Internal("failed to find worker ", task_name));
      return;
    }
    WorkerInterface* wi = it->second;
    GetStatusRequest req;
    GetStatusResponse resp;
    Status status = wi->GetStatus(&req, &resp);
    if (!status.ok()) {
      done(status);
      return;
    }
    for (const auto& it : resp.device_attributes()) {
      if (it.name() == device) {
        *locality = it.locality();
        done(Status::OK());
        return;
      }
    }
    done(errors::Internal("device not found: ", device));
  }
};

class DeviceResDistTest : public ::testing::Test {
 protected:
  DeviceResDistTest() {}

  ~DeviceResDistTest() override {
    for (DeviceMgr* dm : device_mgrs_) {
      delete dm;
    }
    for (auto it : resolvers_) {
      delete it.second;
    }
    for (FakeWorker* w : workers_) {
      delete w;
    }
  }

  void DefineWorkers(int num_workers, int num_devices,
                     const string& device_type,
                     uint64 device_incarnation_base) {
    for (int w = 0; w < num_workers; ++w) {
      string name = strings::StrCat("/job:worker/replica:0/task:", w);
      DefineWorker(name, device_type, num_devices,
                   w * num_devices + device_incarnation_base);
    }
  }

  void DefineWorker(const string& worker_name, const string& device_type,
                    int num_devices, uint64 device_incarnation_base) {
    std::vector<std::unique_ptr<Device>> devices;
    for (int i = 0; i < num_devices; ++i) {
      devices.push_back(NewDevice(
          device_type,
          strings::StrCat(worker_name, "/device:", device_type, ":", i), i,
          device_incarnation_base + i));
    }
    DeviceMgr* dev_mgr = new DeviceMgr(std::move(devices));
    TestableDeviceResolverDistributed* dev_res =
        new TestableDeviceResolverDistributed(dev_mgr, &wc_, worker_name);
    resolvers_[worker_name] = dev_res;
    device_mgrs_.push_back(dev_mgr);
    std::vector<string>* dv = &dev_by_task_[worker_name];
    dv->clear();
    for (auto* d : dev_mgr->ListDevices()) {
      dv->push_back(d->name());
    }
    FakeWorker* fw = new FakeWorker(worker_name, dev_mgr, dev_res);
    workers_.push_back(fw);
    wc_.AddWorker(worker_name, fw);
  }

  void RestartWorker(const string& worker_name, const string& device_type,
                     int num_devices, uint64 device_incarnation_base) {
    for (auto it : resolvers_) {
      it.second->ClearCache();
    }
    // `DefineWorker` creates a device resolver and a worker and adds them to
    // resolvers_ and workers_.  Recreating the worker would overwrite these map
    // entries.  We destroy the old device resolver here; all other objects are
    // cleaned up in the destructor.
    delete resolvers_[worker_name];
    DefineWorker(worker_name, device_type, num_devices,
                 device_incarnation_base);
  }

  void ResolveIncarnationsAndValidate(
      const int num_workers, const int num_devices, const string& worker_prefix,
      const string& device_type,
      const std::vector<std::vector<uint64>>& expected_incarnations) {
    for (int w = 0; w < num_workers; ++w) {
      const string worker_name = absl::StrCat(worker_prefix, w);
      auto* device_resolver = resolvers_[worker_name];
      const string device_prefix =
          absl::StrCat(worker_name, "/device:", device_type, ":");
      for (int peer_w = 0; peer_w < num_workers; ++peer_w) {
        const string peer_worker_name = absl::StrCat(worker_prefix, peer_w);
        for (int d = 0; d < num_devices; ++d) {
          const string device_name =
              absl::StrCat(peer_worker_name, "/device:", device_type, ":", d);
          DeviceNameUtils::ParsedName parsed;
          ASSERT_TRUE(DeviceNameUtils::ParseFullName(device_name, &parsed));
          // NOLINT prevents linter from suggesting absl::Notification as a
          // replacement, which is not available in OSS.
          Notification note;  // NOLINT
          Status status;
          DeviceAttributes attributes;
          device_resolver->GetDeviceAttributesAsync(
              device_name, peer_worker_name, &attributes,
              [&note, &status](const Status& s) {
                status = s;
                note.Notify();
              });
          note.WaitForNotification();
          TF_EXPECT_OK(status);
          EXPECT_EQ(attributes.incarnation(), expected_incarnations[peer_w][d]);
        }
      }
    }
  }

  FakeCache wc_;
  std::vector<DeviceMgr*> device_mgrs_;
  std::unordered_map<string, TestableDeviceResolverDistributed*> resolvers_;
  std::unordered_map<string, std::vector<string>> dev_by_task_;
  std::vector<FakeWorker*> workers_;
};

TEST_F(DeviceResDistTest, Workers3Devices4) {
  DefineWorkers(/*num_workers=*/3, /*num_devices=*/4, /*device_type=*/"CPU",
                /*device_incarnation_base=*/1);
  // Check that every device is available from every task.
  for (auto it : resolvers_) {
    DeviceResolverDistributed* dres = it.second;
    for (auto it2 : dev_by_task_) {
      const string& task_name = it2.first;
      for (const auto& dev_name : it2.second) {
        DeviceNameUtils::ParsedName parsed;
        ASSERT_TRUE(DeviceNameUtils::ParseFullName(dev_name, &parsed));
        Notification note;
        Status status;
        DeviceAttributes attributes;
        dres->GetDeviceAttributesAsync(dev_name, task_name, &attributes,
                                       [&note, &status](const Status& s) {
                                         status = s;
                                         note.Notify();
                                       });
        note.WaitForNotification();
        TF_EXPECT_OK(status);
        EXPECT_EQ(parsed.id, attributes.locality().numa_node());
      }
    }
  }
  // Clear just task 0 from all.
  const string w0_name = "/job:worker/replica:0/task:0";
  for (auto it : resolvers_) {
    if (it.first == w0_name) continue;
    TestableDeviceResolverDistributed* dres = it.second;
    EXPECT_EQ(8, it.second->attr_table().size());
    dres->ClearTask("/job:worker/replica:0/task:0");
    EXPECT_EQ(4, it.second->attr_table().size());
  }
}

TEST_F(DeviceResDistTest, DeviceIncarnationChangesOnFailure) {
  constexpr int num_workers = 3;
  constexpr int num_devices = 4;
  constexpr int failing_worker_index = 1;
  const string device_type = "CPU";
  constexpr uint64 device_incarnation_base = 100;
  DefineWorkers(num_workers, num_devices, device_type, device_incarnation_base);
  const string worker_prefix = "/job:worker/replica:0/task:";
  const string failing_worker =
      absl::StrCat(worker_prefix, failing_worker_index);

  // Check device incarnations match expected.
  std::vector<std::vector<uint64>> expected_incarnations(num_workers);
  for (int w = 0; w < num_workers; ++w) {
    expected_incarnations[w].resize(num_devices);
    for (int d = 0; d < num_devices; ++d) {
      expected_incarnations[w][d] =
          w * num_devices + d + device_incarnation_base;
    }
  }
  ResolveIncarnationsAndValidate(num_workers, num_devices, worker_prefix,
                                 device_type, expected_incarnations);

  // Restart worker `failing_worker`.
  constexpr uint64 restart_incarnation_base = 200;
  RestartWorker(failing_worker, device_type, num_devices,
                restart_incarnation_base);
  for (int d = 0; d < num_devices; ++d) {
    expected_incarnations[failing_worker_index][d] =
        d + restart_incarnation_base;
  }

  // Check incarnations have changed for `failing worker`.
  ResolveIncarnationsAndValidate(num_workers, num_devices, worker_prefix,
                                 device_type, expected_incarnations);
}

}  // namespace
}  // namespace tensorflow
