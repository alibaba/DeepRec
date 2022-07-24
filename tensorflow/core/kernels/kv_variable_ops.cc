/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"
#if GOOGLE_CUDA
#if TF_ENABLE_GPU_EV
#include "tensorflow/core/kernels/kv_variable_ops_gpu.h"
#endif // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace {
const int64 kEmbeddingVarUseDB = -214;
const int64 kInitializableEmbeddingVarUseDB = -215;
}

#define REGISTER_KV_VAR_HANDLE(ktype, vtype)                           \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_CPU)                          \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVar<ktype, vtype>>);
REGISTER_KV_VAR_HANDLE(int32, float)
REGISTER_KV_VAR_HANDLE(int64, float)
#undef REGISTER_KV_VAR_HANDLE

template <typename T, typename TKey, typename TValue>
class KvVariableShapeOp : public OpKernel {
 public:
  explicit KvVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    TensorShape shape({ev->Size(), ev->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KV_VARIABLE_SHAPE(type, ktype, vtype)                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("KvVariableShape").Device(DEVICE_CPU)                      \
                             .TypeConstraint<type>("out_type")        \
                             .TypeConstraint<ktype>("Tkeys"),         \
                             KvVariableShapeOp<type, ktype, vtype>);
REGISTER_KV_VARIABLE_SHAPE(int32, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int32, int64, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int64, float)
#undef REGISTER_KV_VARIABLE_SHAPE

class DestroyKvResourceOp : public OpKernel {
 public:
  explicit DestroyKvResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
        ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
  }

  void Compute(OpKernelContext* ctx) override {
    const ResourceHandle& p = HandleFromInput(ctx, 0);
    Status status = DeleteResource(ctx, p);
    if (ignore_lookup_error_ && errors::IsNotFound(status)) {
      return;
    }
    OP_REQUIRES_OK(ctx, status);
  }

 private:
  bool ignore_lookup_error_;
};

REGISTER_KERNEL_BUILDER(Name("DestroyKvResourceOp").Device(DEVICE_CPU),
                        DestroyKvResourceOp);

template <typename TKey, typename TValue>
class InitializeKvVariableOp : public OpKernel {
 public:
  explicit InitializeKvVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim", &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));

    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));

    if (filter_freq_ < 0) {
      LOG(INFO) << "filter_freq < 0 is invalid, feature filter is disabled.";
      filter_freq_ = 0;
    }

    if (steps_to_live_ == kEmbeddingVarUseDB ||
        steps_to_live_ == kInitializableEmbeddingVarUseDB) {
      LOG(INFO) << "hashmap use db";
      //use_db_ = true;
    } else {
      OP_REQUIRES(c, steps_to_live_ >= 0,
          errors::InvalidArgument(
            "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    }
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    if (embedding::StorageType::LEVELDB == storage_type_) {
      ht_type_ = "leveldb_kv";
      if (layout_ != "normal_contiguous")
        LOG(WARNING)
          << "layout must be NORAML_CONTIGUOUS when storage type is LEVELDB";
      layout_ = "normal_contiguous";
    }

    if (embedding::StorageType::PMEM_LIBPMEM == storage_type_ ||
        embedding::StorageType::PMEM_MEMKIND == storage_type_){
      if (layout_ != "normal_contiguous"){
        LOG(WARNING)
          << "layout must be NORAML_CONTIGUOUS"
          << " when storage type is PMEM_LIBPMEM or PMEM_MEMKIND";
      }
      layout_ = "normal_contiguous";
    }
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& default_values = context->input(2);

    OP_REQUIRES(context, dtype_ == default_values.dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 0);
    ResourceHandle handle_primary = HandleFromInput(context, 1);
    std::string opname = handle_self.name();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    CHECK(block_num_ == 1 || layout_ != "normal_contiguous");

    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {

      OP_REQUIRES_OK(context,
          LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              auto storage_manager =
                  new embedding::StorageManager<TKey, TValue>(
                    handle_self.name(),
                    embedding::StorageConfig(
                      storage_type_, storage_path_, storage_size_, layout_));
              TF_CHECK_OK(storage_manager->Init());
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                         storage_manager,
                         EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                             emb_index_, block_num_, slot_num_,
                             opname + "-primary", steps_to_live_,
                             filter_freq_, max_freq_,
                             l2_weight_threshold_, layout_,
                             max_element_size_, false_positive_probability_,
                             counter_type_, default_value_dim_,
                             record_freq_, record_version_));
            return Status::OK();
            }));
      ev->Init(default_values, default_value_dim_);
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname,
            handle_primary](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             auto storage_manager =
               new embedding::StorageManager<TKey, TValue>(
                 handle_primary.name(), embedding::StorageConfig(storage_type_,
                     storage_path_, storage_size_, layout_));
             TF_CHECK_OK(storage_manager->Init());
             *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                        storage_manager,
                        EmbeddingConfig(
                          primary_emb_index + block_num_ * primary_slot_index,
                          primary_emb_index,
                          block_num_, slot_num_, opname + "-primary",
                          steps_to_live_, filter_freq_, max_freq_,
                          l2_weight_threshold_, layout_,
                          max_element_size_, false_positive_probability_,
                          counter_type_, 0, record_freq_, record_version_));
            // default_values is slot value, should not to initialize primary value
            return Status::OK();
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                  primary_variable->storage_manager(),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                                  emb_index_,
                                  block_num_, slot_num_, opname,
                                  steps_to_live_, filter_freq_,
                                  max_freq_, l2_weight_threshold_,
                                  layout_, max_element_size_,
                                  false_positive_probability_,
                                  counter_type_, default_value_dim_,
                                  record_freq_, record_version_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);
    if (steps_to_live_ != kEmbeddingVarUseDB) {
      ev->SetInitialized();
    }
  }

 private:
  DataType dtype_;
  DataType counter_type_;
  TensorShape shape_;
  int64 steps_to_live_;
  int64 emb_index_;
  int64 block_num_;
  int64 slot_index_;
  int64 slot_num_;
  std::string ht_type_;
  int64 ht_partition_num_;
  int64 filter_freq_;
  int64 max_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_element_size_;
  float false_positive_probability_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  bool record_freq_;
  bool record_version_;
};

#define REGISTER_KERNELS(ktype, vtype)                               \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")             \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype"),       \
                          InitializeKvVariableOp<ktype, vtype>);

#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(int32, T);     \
  REGISTER_KERNELS(int64, T);

TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceIsInitializedOp : public OpKernel {
 public:
  explicit KvResourceIsInitializedOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    bool found;
    if (LookupResource<EmbeddingVar<TKey, TValue>>(
          ctx, HandleFromInput(ctx, 0), &ev).ok()) {
      found = ev->IsInitialized();
      ev->Unref();
    } else {
      found = false;
    }

    output->flat<bool>()(0) = found;
  }
};
#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .Device(DEVICE_CPU),                     \
                          KvResourceIsInitializedOp<ktype, vtype>);
REGISTER_KERNELS(int32, float)
REGISTER_KERNELS(int64, float)
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceGatherOp : public OpKernel {
 public:
  explicit KvResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 4) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    int32* counts = nullptr;
    if (c->num_inputs() == 4)
      counts = (int32*)c->input(3).data();

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)c->input(2).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() || (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      auto do_work = [this, indices_flat,
           out_base, slice_elems, c, default_v, ev, counts] (
               int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, indices_flat(i), i, ev->GetDefaultValueDim(),
              ev->ValueLen());
          int32 count = get_count_fn_(counts, i);
          ev->LookupOrCreate(indices_flat(i),
              out_base + i * slice_elems, default_v_ptr, count);
        }
      };
      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads,
          worker_threads->workers, indices_size,
          slice_bytes, do_work);
          
      ev->storage_manager()->Schedule([ev, indices]() {
        embedding::BatchCache<TKey>* cache = ev->Cache();
        if (cache) {
          cache->add_to_rank(indices);
        }
      });
    }
  }

  private:
    bool is_use_default_value_tensor_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
};

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, int32, type);      \
  REGISTER_GATHER_FULL(dev, int64, type)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_float(REGISTER_GATHER_CPU);
TF_CALL_double(REGISTER_GATHER_CPU);
//TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGatherV1")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .HostMemory("counts")               \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, int32, type);      \
  REGISTER_GATHER_FULL(dev, int64, type)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_float(REGISTER_GATHER_CPU);
TF_CALL_double(REGISTER_GATHER_CPU);
//TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL
/*
// Op that outputs tensors of all keys and all values.
template <typename TKey, typename TValue>
class KvResourceImportOp : public OpKernel {
 public:
  explicit KvResourceImportOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", context->input(1).dtype()));
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    const Tensor& default_values = context->input(1);
    OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, HandleFromInput(context, 0), &variable,
            [this, default_values](EmbeddingVar<TKey, TValue>** ptr) {
              auto ht = KVFactory<TKey, TValue>::CreateKV(
                  ht_type_, ht_partition_num_);
              *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                       new HashMap<TKey, TValue>(
                         ht, cpu_allocator()),
                         steps_to_live_);
              return (*ptr)->Init(default_values);
            }));
    core::ScopedUnref unref_me(variable);

    HashMap<TKey, TValue>* hashmap = variable->hashmap();
    const Tensor& keys = context->input(3);
    const Tensor& values = context->input(4);
    const Tensor& versions = context->input(5);
    LOG(INFO) <<  "EV:"
              << HandleFromInput(context, 0).name()
              << ", Import Size:"
              <<  keys.dim_size(0);
    OP_REQUIRES_OK(context, hashmap->Import(keys, values, versions));
    variable->SetInitialized();
  }

 private:
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  std::string ht_type_;
  int64 ht_partition_num_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImport")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
*/

constexpr int64 DEFAULT_RESTORE_THREAD_NUM = 4;

class KvRestoreThreadPool {
 public:
  KvRestoreThreadPool() {
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_EV_RESTORE_THREAD_NUM",
          DEFAULT_RESTORE_THREAD_NUM, &thread_num_));
  }

  static thread::ThreadPool* GetInstance() {
    static thread::ThreadPool tp(Env::Default(),
        "restore_ev_threadpool", thread_num_);
    return &tp;
  }

 private:
  static int64 thread_num_;
};

int64 KvRestoreThreadPool::thread_num_ =
    DEFAULT_RESTORE_THREAD_NUM;

template <typename TKey, typename TValue>
class KvResourceImportV2Op: public AsyncOpKernel {
 public:
  explicit KvResourceImportV2Op(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ",
                    std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    //OP_REQUIRES_OK(c, c->GetAttr("restore_versions", &restore_versions_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim",
          &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));

    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EV_ASYNC_RESTORE", true,
                                   &ev_async_restore_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(4);
    const std::string name_string = name.scalar<string>()();
    const Tensor& default_values = context->input(3);
    OP_REQUIRES(context, dtype_ == default_values.dtype(),
        errors::InvalidArgument(
          "Variable and ddd value dtypes don't match; respectively, ",
          dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 1);
    ResourceHandle handle_primary = HandleFromInput(context, 2);
    std::string opname = handle_self.name();
    EmbeddingVar<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
         handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              auto storage_manager =
                new embedding::StorageManager<TKey, TValue>(
                  handle_self.name(), embedding::StorageConfig(
                    storage_type_, storage_path_, storage_size_, layout_));
              TF_CHECK_OK(storage_manager->Init());
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                         storage_manager,
                         EmbeddingConfig(
                           emb_index_ + block_num_ * slot_index_,
                           emb_index_,
                           block_num_, slot_num_, opname + "-primary",
                           steps_to_live_, filter_freq_,
                           max_freq_, l2_weight_threshold_,
                           layout_,  max_element_size_,
                           false_positive_probability_,
                           counter_type_, default_value_dim_,
                           record_freq_, record_version_));
             return Status::OK();
            }));
      ev->Init(default_values, default_value_dim_);
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname,
            handle_primary](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             auto storage_manager =
               new embedding::StorageManager<TKey, TValue>(
                 handle_primary.name(), embedding::StorageConfig(
                   storage_type_, storage_path_, storage_size_,
                   layout_));
             TF_CHECK_OK(storage_manager->Init());
             *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                 storage_manager, EmbeddingConfig(
                   primary_emb_index + block_num_ * primary_slot_index,
                   primary_emb_index, block_num_, slot_num_,
                   opname + "-primary", steps_to_live_, filter_freq_,
                   max_freq_, l2_weight_threshold_,
                   layout_,  max_element_size_,
                   false_positive_probability_,
                   counter_type_, 0, record_freq_, record_version_));
            // default_values is slot value, should not to initialize primary value
            return Status::OK();
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                  primary_variable->storage_manager(),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_, block_num_, slot_num_, opname,
                    steps_to_live_, filter_freq_, max_freq_,
                    l2_weight_threshold_, layout_, max_element_size_,
                    false_positive_probability_,
                    counter_type_, default_value_dim_,
                    record_freq_, record_version_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);

    auto do_compute = [this, context, file_name_string, ev,
         name_string, done] () {
      BundleReader reader(Env::Default(), file_name_string);
      auto s = reader.status();
      if (!s.ok()) {
        LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                   << s.ToString();
      }

      EVRestoreDynamically(
          ev, name_string, partition_id_, partition_num_, context, &reader,
          "-partition_offset", "-keys", "-values", "-versions", "-freqs");
      ev->SetInitialized();
      done();
    };

    if (ev_async_restore_) {
      auto tp = KvRestoreThreadPool::GetInstance();
      tp->Schedule(do_compute);
    } else {
      do_compute();
    }
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  DataType counter_type_;
  int64 max_element_size_;
  float false_positive_probability_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
  int64 emb_index_;
  int64 slot_index_;
  int64 block_num_;
  int64 slot_num_;
  int64 filter_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_freq_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  bool record_freq_;
  bool record_version_;
  bool ev_async_restore_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")           \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
//TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceIncrImportOp: public AsyncOpKernel {
 public:
  explicit KvResourceIncrImportOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));

    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
        errors::InvalidArgument(
          "partition_id must >= 0, ", std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
        errors::InvalidArgument(
          "partition_num must >= 1, ", std::to_string(partition_num_)));

  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<string>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context,
        LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    BundleReader reader(Env::Default(), file_name_string);
    OP_REQUIRES_OK(context, reader.status());

    LOG(INFO) << "incr import, evname:"
              << name_string
              << "partition_num:"
              << partition_num_;

    EVRestoreDynamically(
        ev, name_string, partition_id_, partition_num_, context, &reader,
        "-incr_partition_offset", "-sparse_incr_keys", "-sparse_incr_values",
        "-sparse_incr_versions", "-sparse_incr_freqs");
    ev->SetInitialized();
    done(); 
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
};


#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceIncrImport")         \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceIncrImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
//TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

// Op that outputs tensors of all keys and all values.
template<typename TKey, typename TValue>
class KvResourceExportOp : public OpKernel {
 public:
  explicit KvResourceExportOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    EmbeddingVar<TKey, TValue> *ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    std::vector<TKey> tot_key_list;
    std::vector<TValue *> tot_valueptr_list;
    std::vector<int64> tot_version_list;
    std::vector<int64> tot_freq_list;
    embedding::Iterator* it = nullptr;
    int64 total_size = ev->GetSnapshot(
        &tot_key_list, &tot_valueptr_list, &tot_version_list,
        &tot_freq_list, &it);

    // Create an output tensor
    Tensor *keys_output_tensor = NULL;
    Tensor *values_output_tensor = NULL;
    Tensor *versions_output_tensor = NULL;
    Tensor *freq_output_tensor = NULL;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          0, TensorShape({total_size}), &keys_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          1, TensorShape({total_size, ev->ValueLen()}),
          &values_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          2, TensorShape({tot_version_list.size()}),
          &versions_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          3, TensorShape({tot_freq_list.size()}),
          &freq_output_tensor));

    auto keys_output = keys_output_tensor->template flat<TKey>();
    auto val_matrix = values_output_tensor->matrix<TValue>();
    auto versions_output = versions_output_tensor->template flat<int64>();
    auto freq_output = freq_output_tensor->template flat<int64>();

    for(size_t i = 0; i < total_size; i++) {
      keys_output(i) = tot_key_list[i];
      TValue *value = tot_valueptr_list[i];
      for(int64 m = 0; m < ev->ValueLen(); m++) {
        val_matrix(i, m) = *(value + m);
      }
      if (tot_version_list.size() != 0) {
        versions_output(i) = tot_version_list[i];
      }
      if (tot_freq_list.size() != 0) {
        freq_output(i) = tot_freq_list[i];
      }
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues"), \
                          KvResourceExportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)

REGISTER_KERNELS_ALL_INDEX(float);

#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if TF_ENABLE_GPU_EV
#define REGISTER_KV_VAR_HANDLE(ktype, vtype)                           \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_GPU)                          \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVarGPU<ktype, vtype>>);
REGISTER_KV_VAR_HANDLE(int32, float)
REGISTER_KV_VAR_HANDLE(int64, float)
#undef REGISTER_KV_VAR_HANDLE

template <typename T, typename TKey, typename TValue>
class KvVariableShapeOpGPU : public OpKernel {
 public:
  explicit KvVariableShapeOpGPU(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    TensorShape shape({ev->Size(), ev->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KV_VARIABLE_SHAPE(type, ktype, vtype)                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("KvVariableShape").Device(DEVICE_GPU)                      \
                             .TypeConstraint<type>("out_type")        \
                             .TypeConstraint<ktype>("Tkeys")          \
                             .HostMemory("output"),                   \
                             KvVariableShapeOpGPU<type, ktype, vtype>);
REGISTER_KV_VARIABLE_SHAPE(int32, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int32, int64, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int64, float)
#undef REGISTER_KV_VARIABLE_SHAPE

REGISTER_KERNEL_BUILDER(Name("DestroyKvResourceOp").Device(DEVICE_GPU),
                        DestroyKvResourceOp);

template <typename TKey, typename TValue>
class InitializeKvVariableOpGPU : public OpKernel {
 public:
  explicit InitializeKvVariableOpGPU(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));

     // get ev emb_index
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
     // get ev block_num
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
     // get ev slot_index
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));

    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));

    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));

    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));

    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));

    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));

    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));

    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));

    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));

    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim", &default_value_dim_));

    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));

    if (filter_freq_ < 0) {
      LOG(INFO) << "filter_freq < 0 is invalid, feature filter is disabled.";
      filter_freq_ = 0;
    }

    if (steps_to_live_ == kEmbeddingVarUseDB ||
        steps_to_live_ == kInitializableEmbeddingVarUseDB) {
      LOG(INFO) << "hashmap use db";
      //use_db_ = true;
    } else {
      OP_REQUIRES(c, steps_to_live_ >= 0,
          errors::InvalidArgument(
            "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    }
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& default_values = context->input(2);

    OP_REQUIRES(context, dtype_ == default_values.dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 0);
    ResourceHandle handle_primary = HandleFromInput(context, 1);
    std::string opname = handle_self.name();

    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              GPUHashTable<TKey, TValue>* ht =
                new GPUHashTable<TKey, TValue>(-1,
                    context->get_allocator(AllocatorAttributes()));
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  ht, context->get_allocator(AllocatorAttributes()),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                                  emb_index_,
                                  block_num_, slot_num_,
                                  opname + "-primary",
                                  steps_to_live_, filter_freq_, max_freq_,
                                  l2_weight_threshold_, layout_,
                                  max_element_size_,
                                  false_positive_probability_,
                                  counter_type_, default_value_dim_));
            return (*ptr)->Init(default_values, default_value_dim_);
            }));
    } else {
      EmbeddingVarGPU<TKey, TValue>* primary_variable = nullptr;

      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, context,
            handle_primary](EmbeddingVarGPU<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             GPUHashTable<TKey, TValue>* ht =
               new GPUHashTable<TKey, TValue>(-1,
                   context->get_allocator(AllocatorAttributes()));
             *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_primary.name(),
                 ht, context->get_allocator(AllocatorAttributes()),
                 EmbeddingConfig(
                   primary_emb_index + block_num_ * primary_slot_index,
                   primary_emb_index,
                   block_num_, slot_num_, opname + "-primary",
                   steps_to_live_, filter_freq_, max_freq_,
                   l2_weight_threshold_, layout_,
                   max_element_size_,
                   false_positive_probability_,
                   counter_type_));
             return (*ptr)->Init(default_values, default_value_dim_);
           }));


      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  primary_variable->kv(),
                  context->get_allocator(AllocatorAttributes()),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_,
                    block_num_, slot_num_, opname,
                    steps_to_live_, 0,
                    max_freq_, l2_weight_threshold_,
                    layout_, 0, -1.0, counter_type_, default_value_dim_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);
    if (steps_to_live_ != kEmbeddingVarUseDB) {
      ev->SetInitialized();
    }
  }

 private:
  DataType dtype_;
  DataType counter_type_;
  TensorShape shape_;
  int64 steps_to_live_;
  int64 emb_index_;
  int64 block_num_;
  int64 slot_index_;
  int64 slot_num_;
  std::string ht_type_;
  int64 ht_partition_num_;
  int64 filter_freq_;
  int64 max_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_element_size_;
  float false_positive_probability_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  int64 default_value_dim_;
};

#define REGISTER_KERNELS(ktype, vtype)                               \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")             \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype"),       \
                          InitializeKvVariableOpGPU<ktype, vtype>);
#define REGISTER_GPU_KERNELS(T)        \
  REGISTER_KERNELS(int32, T);     \
  REGISTER_KERNELS(int64, T);
TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .Device(DEVICE_GPU),                     \
                          KvResourceIsInitializedOp<ktype, vtype>);
REGISTER_KERNELS(int32, float)
REGISTER_KERNELS(int64, float)
#undef REGISTER_KERNELS

template <typename Device, typename TKey, typename TValue>
class KvResourceGatherOpGPU : public OpKernel {
 public:
  explicit KvResourceGatherOpGPU(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);

    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(
          indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
            "ev's value_len should same with output's dimension(1)",
            std::to_string(slice_elems), std::to_string(ev->ValueLen())));


      const TKey* key_base = &indices_flat(0);
      const cudaStream_t& stream = c->eigen_device<Device>().stream();
      if (is_use_default_value_tensor_) {
        Tensor default_values(c->input(2));
        auto default_value_num = default_values.NumElements() / ev->ValueLen();
        auto default_values_matrix = default_values.shaped<TValue, 2>(
            {default_value_num, ev->ValueLen()});     
        TValue* default_v_base = &default_values_matrix(0, 0);
        ev->LookupOrCreate(key_base, out_base, default_v_base,
            default_value_num, is_use_default_value_tensor_,
            indices_size, stream);
      } else {
        ev->LookupOrCreate(key_base, out_base, ev->GetDefaultValuePtr(),
            ev->GetDefaultValueDim(), is_use_default_value_tensor_,
            indices_size, stream);
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
};

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOpGPU<GPUDevice, ktype, vtype>)
#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, int32, type);      \
  REGISTER_GATHER_FULL(dev, int64, type)
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)
TF_CALL_float(REGISTER_GATHER_GPU);
TF_CALL_double(REGISTER_GATHER_GPU);
#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template<typename Device, typename TKey, typename TValue>
class KvResourceExportOpGPU : public OpKernel {
 public:
  explicit KvResourceExportOpGPU(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    EmbeddingVarGPU<TKey, TValue> *ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    int64 total_size = ev->Size();

    // Create an output tensor
    Tensor *keys_output_tensor = NULL;
    Tensor *values_output_tensor = NULL;
    Tensor *versions_output_tensor = NULL;
    Tensor *freq_output_tensor = NULL;

    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, TensorShape({total_size}),
          &keys_output_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(1, TensorShape({total_size, ev->ValueLen()}),
          &values_output_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(2, TensorShape({0}),
          &versions_output_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(3, TensorShape({0}),
          &freq_output_tensor));

    auto keys_flat = keys_output_tensor->flat<TKey>();
    TKey* key_base = &keys_flat(0);
    auto values_flat = values_output_tensor->flat<TValue>();
    TValue* value_base = &values_flat(0);

    const cudaStream_t& stream = ctx->eigen_device<Device>().stream();
    ev->GetSnapshot(key_base, value_base, stream);
  }
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues") \
                            .HostMemory("keys")                \
                            .HostMemory("values")              \
                            .HostMemory("versions")            \
                            .HostMemory("freqs"),              \
                          KvResourceExportOpGPU<GPUDevice, ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)

REGISTER_KERNELS_ALL_INDEX(float);

#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceImportV2OpGPU: public OpKernel {
 public:
  explicit KvResourceImportV2OpGPU(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ",
                    std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    //OP_REQUIRES_OK(c, c->GetAttr("restore_versions", &restore_versions_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim",
          &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(4);
    const std::string name_string = name.scalar<string>()();
    const Tensor& default_values = context->input(3);
    OP_REQUIRES(context, dtype_ == default_values.dtype(),
        errors::InvalidArgument(
          "Variable and ddd value dtypes don't match; respectively, ",
          dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 1);
    ResourceHandle handle_primary = HandleFromInput(context, 2);
    std::string opname = handle_self.name();
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              GPUHashTable<TKey, TValue>* ht =
                new GPUHashTable<TKey, TValue>(-1,
                    context->get_allocator(AllocatorAttributes()));
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  ht, context->get_allocator(AllocatorAttributes()),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                                  emb_index_,
                                  block_num_, slot_num_,
                                  opname + "-primary",
                                  steps_to_live_, filter_freq_, max_freq_,
                                  l2_weight_threshold_, layout_,
                                  max_element_size_,
                                  false_positive_probability_,
                                  counter_type_, default_value_dim_));
            return (*ptr)->Init(default_values, default_value_dim_);
            }));
    } else {
      EmbeddingVarGPU<TKey, TValue>* primary_variable = nullptr;

      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, context,
            handle_primary](EmbeddingVarGPU<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             GPUHashTable<TKey, TValue>* ht =
               new GPUHashTable<TKey, TValue>(-1,
                   context->get_allocator(AllocatorAttributes()));
             *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_primary.name(),
                 ht, context->get_allocator(AllocatorAttributes()),
                 EmbeddingConfig(
                   primary_emb_index + block_num_ * primary_slot_index,
                   primary_emb_index,
                   block_num_, slot_num_, opname + "-primary",
                   steps_to_live_, filter_freq_, max_freq_,
                   l2_weight_threshold_, layout_,
                   max_element_size_,
                   false_positive_probability_,
                   counter_type_));
             return (*ptr)->Init(default_values, default_value_dim_);
           }));


      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  primary_variable->kv(),
                  context->get_allocator(AllocatorAttributes()),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_,
                    block_num_, slot_num_, opname,
                    steps_to_live_, 0,
                    max_freq_, l2_weight_threshold_,
                    layout_, 0, -1.0, counter_type_, default_value_dim_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);

    BundleReader reader(Env::Default(), file_name_string);
    auto s = reader.status();
    if (!s.ok()) {
      LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                 << s.ToString();
    }

    EVRestoreDynamicallyGPU(
        ev, name_string, partition_id_, partition_num_, context, &reader,
        "-partition_offset", "-keys", "-values", "-versions", "-freqs");
    ev->SetInitialized();
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  DataType counter_type_;
  int64 max_element_size_;
  float false_positive_probability_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
  int64 emb_index_;
  int64 slot_index_;
  int64 block_num_;
  int64 slot_num_;
  int64 filter_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_freq_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  bool record_freq_;
  bool record_version_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")           \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2OpGPU<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
//TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

