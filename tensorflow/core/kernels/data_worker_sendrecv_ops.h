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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_WORKER_SENDRECV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_WORKER_SENDRECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/data_worker_rendezvous.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class BaseDataWorkerSendOp : public AsyncOpKernel {
 public:
  explicit BaseDataWorkerSendOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 protected:
  virtual void Send(OpKernelContext* ctx,
                    const DataWorkerRendezvous::Args& args,
                    DoneCallback done) = 0;
  string tensor_name_;
  string send_device_;
  string send_device_type_;
  DataWorkerRendezvous::ParsedKey parsed_key_;
  
 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BaseDataWorkerSendOp);
};

class DataWorkerSendOp : public BaseDataWorkerSendOp {
 public:
  explicit DataWorkerSendOp(OpKernelConstruction* ctx)
   : BaseDataWorkerSendOp(ctx) {}

  void Send(OpKernelContext* ctx,
            const DataWorkerRendezvous::Args& args,
            DoneCallback done) override;
};

class LocalDataWorkerSendOp : public BaseDataWorkerSendOp {
 public:
  explicit LocalDataWorkerSendOp(OpKernelConstruction* ctx)
   : BaseDataWorkerSendOp(ctx) {}

  void Send(OpKernelContext* ctx,
            const DataWorkerRendezvous::Args& args,
            DoneCallback done) override;
};

class DataWorkerRecvOp : public AsyncOpKernel {
 public:
  explicit DataWorkerRecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  string recv_device_;
  string recv_device_type_;
  DataWorkerRendezvous::ParsedKey parsed_key_;
  bool attrs_set_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(DataWorkerRecvOp);
};

class DataWorkerFuseRecvOp : public AsyncOpKernel {
 public:
  explicit DataWorkerFuseRecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  string recv_device_;
  string recv_device_type_;
  std::vector<DataWorkerRendezvous::ParsedKey> parsed_keys_;
  bool attrs_set_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(DataWorkerFuseRecvOp);
};
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_WORKER_SENDRECV_OPS_H_
