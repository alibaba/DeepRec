#include <thread>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

#include <time.h>
#include <sys/resource.h>
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace embedding {
namespace {
const int THREADNUM = 16;
const int64 max = 2147483647;

struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory() : size(0), resident(0), share(0),
                 trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    //ret.push_back(reader->key().ToString());
    ret.push_back(std::string(reader->key()));
  }
  return ret;
}

TEST(TensorBundleTest, TestEVShrinkL2) {
  int64 value_size = 3;
  int64 insert_num = 5;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 1.0));
  //float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "name", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* emb_var
    = new EmbeddingVar<int64, float>("name",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", -1, 0, 99999, 14.0));
  emb_var ->Init(value, 1);
  
  for (int64 i=0; i < insert_num; ++i) {
    ValuePtr<float>* value_ptr = nullptr;
    Status s = emb_var->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = emb_var->flat(value_ptr);
    vflat += vflat.constant((float)i);
  }

  int size = emb_var->Size();
  emb_var->Shrink();
  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size:" << emb_var->Size();

  ASSERT_EQ(emb_var->Size(), 2);
}

TEST(TensorBundleTest, TestEVShrinkLockless) {

  int64 value_size = 64;
  int64 insert_num = 30;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  int steps_to_live = 5;
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "name", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* emb_var
    = new EmbeddingVar<int64, float>("name",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", steps_to_live));
  emb_var ->Init(value, 1);


  LOG(INFO) << "size:" << emb_var->Size();


  for (int64 i=0; i < insert_num; ++i) {
    ValuePtr<float>* value_ptr = nullptr;
    Status s = emb_var->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = emb_var->flat(value_ptr);
    emb_var->UpdateVersion(value_ptr, i);
  }

  int size = emb_var->Size();
  emb_var->Shrink(insert_num);

  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size: " << emb_var->Size();

  ASSERT_EQ(size, insert_num);
  ASSERT_EQ(emb_var->Size(), steps_to_live);

}


TEST(EmbeddingVariableTest, TestEmptyEV) {
  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  {
    auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
    TF_CHECK_OK(storage_manager->Init());
    EmbeddingVar<int64, float>* variable
              = new EmbeddingVar<int64, float>("EmbeddingVar",
                  storage_manager);
    variable->Init(value, 1);

    LOG(INFO) << "size:" << variable->Size();
    Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

    BundleWriter writer(Env::Default(), Prefix("foo"));
    DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
    TF_ASSERT_OK(writer.Finish());

    {
      BundleReader reader(Env::Default(), Prefix("foo"));
      TF_ASSERT_OK(reader.status());
      EXPECT_EQ(
          AllTensorKeys(&reader),
          std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
      {
        string key = "var/part_0-keys";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read keys:" << val.DebugString();
      }
      {
        string key = "var/part_0-values";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0, value_size});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read values:" << val.DebugString();
      }
      {
        string key = "var/part_0-versions";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read versions:" << val.DebugString();
      }
    }
  }
}

TEST(EmbeddingVariableTest, TestEVExportSmallLockless) {

  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", 5));
  variable->Init(value, 1);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  for (int64 i = 0; i < 5; i++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
    vflat(i) = 5.0;
  }

  LOG(INFO) << "size:" << variable->Size();


  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5, value_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}

TEST(EmbeddingVariableTest, TestEVExportLargeLockless) {

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", 5));
  variable->Init(value, 1);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  int64 ev_size = 10048576;
  for (int64 i = 0; i < ev_size; i++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }

  LOG(INFO) << "size:" << variable->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size, value_size});
      LOG(INFO) << "read values:" << val.DebugString();
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}

void multi_insertion(EmbeddingVar<int64, float>* variable, int64 value_size){
  for (long j = 0; j < 5; j++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(j, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }
}

TEST(EmbeddingVariableTest, TestMultiInsertion) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager);

  variable->Init(value, 1);

  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(multi_insertion,variable, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  std::vector<int64> tot_key_list;
  std::vector<float* > tot_valueptr_list;
  std::vector<int64> tot_version_list;
  std::vector<int64> tot_freq_list;
  embedding::Iterator* it = nullptr;
  int64 total_size = variable->GetSnapshot(&tot_key_list, &tot_valueptr_list, &tot_version_list, &tot_freq_list, &it);

  ASSERT_EQ(variable->Size(), 5);
  ASSERT_EQ(variable->Size(), total_size);
}

void InsertAndLookup(EmbeddingVar<int64, int64>* variable, int64 *keys, long ReadLoops, int value_size){
  for (long j = 0; j < ReadLoops; j++) {
    int64 *val = (int64 *)malloc((value_size+1)*sizeof(int64));
    variable->LookupOrCreate(keys[j], val, &(keys[j]));
    variable->LookupOrCreate(keys[j], val, (&keys[j]+1));
    ASSERT_EQ(keys[j] , val[0]);
    free(val);
  }
}

void MultiBloomFilter(EmbeddingVar<int64, float>* var, int value_size, int64 i) {
  for (long j = 0; j < 1; j++) {
    float *val = (float *)malloc((value_size+1)*sizeof(float));
    var->LookupOrCreate(i+1, val, nullptr);
  }
}

TEST(EmbeddingVariableTest, TestBloomFilter) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal", 10, 0.01));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));
  float *default_value = (float *)malloc((value_size+1)*sizeof(float));
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(2, val, default_value);
  
  std::vector<int64> keylist;
  std::vector<float *> valuelist;
  std::vector<int64> version_list;
  std::vector<int64> freq_list;

  embedding::Iterator* it = nullptr;
  var->GetSnapshot(&keylist, &valuelist, &version_list, &freq_list, &it);
  ASSERT_EQ(var->Size(), keylist.size());  

}

TEST(EmbeddingVariableTest, TestBloomCounterInt64) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal", 10, 0.01, DT_UINT64));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int64* counter = (int64*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt32) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal", 10, 0.01, DT_UINT32));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int32* counter = (int32*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt16) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal_contiguous", 10, 0.01, DT_UINT16));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  //int16* counter = (int16 *)var->GetBloomCounter(); 
  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int16* counter = (int16*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt8) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal_contiguous", 10, 0.01, DT_UINT8));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int8* counter = (int8*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 
  //int8* counter = (int8 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestInsertAndLookup) {
  int64 value_size = 128;
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));
 // float* fill_v = (int64*)malloc(value_size * sizeof(int64));
  auto storage_manager = new embedding::StorageManager<int64, int64>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, int64>* variable
    = new EmbeddingVar<int64, int64>("EmbeddingVar",
       storage_manager/*, EmbeddingConfig(0, 0, 1, 0, "")*/);

  variable->Init(value, 1);

  int64 InsertLoops = 1000;
  bool* flag = (bool *)malloc(sizeof(bool)*max);
  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoops);
  long *counter = (long *)malloc(sizeof(long)*InsertLoops);

  for (long i = 0; i < max; i++) {
    flag[i] = 0;
  }

  for (long i = 0; i < InsertLoops; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < InsertLoops) {
    long j = rand() % max;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else { // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(InsertAndLookup, variable, &keys[i*InsertLoops/THREADNUM], InsertLoops/THREADNUM, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

}

void MultiFilter(EmbeddingVar<int64, float>* variable, int value_size) {
  float *val = (float *)malloc((value_size+1)*sizeof(float));
  variable->LookupOrCreate(20, val, nullptr);
}

TEST(EmbeddingVariableTest, TestFeatureFilterParallel) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 7));

  var->Init(value, 1);
  float *val = (float *)malloc((value_size+1)*sizeof(float));
  int thread_num = 5;
  std::vector<std::thread> insert_threads(thread_num);
  for (size_t i = 0 ; i < thread_num; i++) {
    insert_threads[i] = std::thread(MultiFilter, var, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  ValuePtr<float>* value_ptr = nullptr;
  var->LookupOrCreateKey(20, &value_ptr);
  ASSERT_EQ(value_ptr->GetFreq(), thread_num);
}


EmbeddingVar<int64, float>* InitEV_Lockless(int64 value_size) {
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager);

  variable->Init(value, 1);
  return variable;
}

void MultiLookup(EmbeddingVar<int64, float>* variable, int64 InsertLoop, int thread_num, int i) {
  for (int64 j = i * InsertLoop/thread_num; j < (i+1)*InsertLoop/thread_num; j++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(j, &value_ptr);
  }
}

void BM_MULTIREAD_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  float* fill_v = (float*)malloc(value_size * sizeof(float));

  for (int64 i = 0; i < InsertLoop; i++){
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }

  testing::StartTiming();
  while(iters--){
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(MultiLookup, variable, InsertLoop, thread_num, i);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }

}

void hybrid_process(EmbeddingVar<int64, float>* variable, int64* keys, int64 InsertLoop, int thread_num, int64 i, int64 value_size) {
  float *val = (float *)malloc(sizeof(float)*(value_size + 1));
  for (int64 j = i * InsertLoop/thread_num; j < (i+1) * InsertLoop/thread_num; j++) {
    variable->LookupOrCreate(keys[j], val, nullptr);
  }
}

void BM_HYBRID_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoop);

  for (int64 i = 0; i < InsertLoop; i++) {
    keys[i] =  rand() % 1000;
  }

  testing::StartTiming();
  while (iters--) {
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(hybrid_process, variable, keys, InsertLoop, thread_num, i, value_size);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }
}

BENCHMARK(BM_MULTIREAD_LOCKLESS)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

BENCHMARK(BM_HYBRID_LOCKLESS)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);


TEST(EmbeddingVariableTest, TestAllocate) {
  int value_len = 8;
  double t0 = getResident()*getpagesize()/1024.0/1024.0;
  double t1 = 0;
  LOG(INFO) << "memory t0: " << t0;
  for (int64 i = 0; i < 1000; ++i) {
    float* tensor_val = TypedAllocator::Allocate<float>(ev_allocator(), value_len, AllocationAttributes());
    t1 = getResident()*getpagesize()/1024.0/1024.0;
    memset(tensor_val, 0, sizeof(float) * value_len);
  }
  double t2 = getResident()*getpagesize()/1024.0/1024.0;
  LOG(INFO) << "memory t1-t0: " << t1-t0;
  LOG(INFO) << "memory t2-t1: " << t2-t1;
  LOG(INFO) << "memory t2-t0: " << t2-t0;
}

TEST(EmbeddingVariableTest, TestEVStorageType_DRAM) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(/*emb_index = */0, /*primary_emb_index = */0,
                          /*block_num = */1, /*slot_num = */1,
                          /*name = */"", /*steps_to_live = */0,
                          /*filter_freq = */0, /*max_freq = */999999,
                          /*l2_weight_threshold = */-1.0, /*layout = */"normal",
                          /*max_element_size = */0, /*false_positive_probability = */-1.0,
                          /*counter_type = */DT_UINT64));
  variable->Init(value, 1);

  int64 ev_size = 100;
  for (int64 i = 0; i < ev_size; i++) {
    variable->LookupOrCreate(i, fill_v, nullptr);
  }

  LOG(INFO) << "size:" << variable->Size();
}

void t1(KVInterface<int64, float>* hashmap) {
  for (int i = 0; i< 100; ++i) {
    hashmap->Insert(i, new NormalValuePtr<float>(ev_allocator(), 100));
  }
}

TEST(EmbeddingVariableTest, TestRemoveLockless) {

  KVInterface<int64, float>* hashmap = new LocklessHashMap<int64, float>();
  ASSERT_EQ(hashmap->Size(), 0);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  auto t = std::thread(t1, hashmap);
  t.join();
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  ASSERT_EQ(hashmap->Size(), 100);
  TF_CHECK_OK(hashmap->Remove(1));
  TF_CHECK_OK(hashmap->Remove(2));
  ASSERT_EQ(hashmap->Size(), 98);
  LOG(INFO) << "2 size:" << hashmap->Size();
}

TEST(EmbeddingVariableTest, TestBatchCommitofDBKV) {
  int64 value_size = 4;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  std::vector<int64> size;
  size.emplace_back(1000);
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig(embedding::LEVELDB, testing::TmpDir(), size, "normal_contiguous"));
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(/*emb_index = */0, /*primary_emb_index = */0,
                          /*block_num = */1, /*slot_num = */0,
                          /*name = */"", /*steps_to_live = */0,
                          /*filter_freq = */0, /*max_freq = */999999,
                          /*l2_weight_threshold = */-1.0, /*layout = */"normal_contiguous",
                          /*max_element_size = */0, /*false_positive_probability = */-1.0,
                          /*counter_type = */DT_UINT64));
  variable->Init(value, 1);
  std::vector<ValuePtr<float>*> value_ptr_list;
  std::vector<int64> key_list;

  for(int64 i = 0; i < 6; i++) {
    key_list.emplace_back(i);
    ValuePtr<float>* tmp = new NormalContiguousValuePtr<float>(ev_allocator(), 4);
    value_ptr_list.emplace_back(tmp);
  }

  variable->BatchCommit(key_list, value_ptr_list);
  for(int64 i = 0; i < 6; i++) {
    ValuePtr<float>* tmp = nullptr;
    Status s = variable->storage_manager()->GetOrCreate(i, &tmp, 4);
    ASSERT_EQ(s.ok(), true);
  }
}

void InsertAndCommit(KVInterface<int64, float>* hashmap) {
  for (int64 i = 0; i< 100; ++i) {
    const ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 100);
    hashmap->Insert(i, tmp);
    hashmap->Commit(i, tmp);
  }
}

TEST(EmbeddingVariableTest, TestSizeDBKV) {
  KVInterface<int64, float>* hashmap = new LevelDBKV<int64, float>(testing::TmpDir());
  hashmap->SetTotalDims(100);
  ASSERT_EQ(hashmap->Size(), 0);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  auto t = std::thread(InsertAndCommit, hashmap);
  t.join();
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  ASSERT_EQ(hashmap->Size(), 100);
  TF_CHECK_OK(hashmap->Remove(1));
  TF_CHECK_OK(hashmap->Remove(2));
  ASSERT_EQ(hashmap->Size(), 98);
  LOG(INFO) << "2 size:" << hashmap->Size();
}

TEST(EmbeddingVariableTest, TestSSDIterator) {
  std::string temp_dir = testing::TmpDir();
  Allocator* alloc = ev_allocator();
  KVInterface<int64, float>* hashmap = new SSDHashKV<int64, float>(temp_dir, alloc);
  hashmap->SetTotalDims(126);
  ASSERT_EQ(hashmap->Size(), 0);
  std::vector<ValuePtr<float>*> value_ptrs;
  for (int64 i = 0; i < 10; ++i) {
    ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(alloc, 126);
    tmp->SetValue((float)i, 126);
    value_ptrs.emplace_back(tmp);
  }
  for (int64 i = 0; i < 10; i++) {
    hashmap->Commit(i, value_ptrs[i]);
  }
  embedding::Iterator* it = hashmap->GetIterator();
  int64 index = 0;
  float val_p[126] = {0.0};
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    int64 key = -1;
    it->Key((char*)&key, sizeof(int64));
    it->Value((char*)val_p, 126 * sizeof(float), 0);
    ASSERT_EQ(key, index);
    for (int i = 0; i < 126; i++)
      ASSERT_EQ(val_p[i], key);
    index++;
  }
}

TEST(EmbeddingVariableTest, TestLevelDBIterator) {
  KVInterface<int64, float>* hashmap = new LevelDBKV<int64, float>(testing::TmpDir());
  hashmap->SetTotalDims(126);
  ASSERT_EQ(hashmap->Size(), 0);
  std::vector<ValuePtr<float>*> value_ptrs;
  for (int64 i = 0; i < 10; ++i) {
    ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 126);
    tmp->SetValue((float)i, 126);
    value_ptrs.emplace_back(tmp);
  }
  for (int64 i = 0; i < 10; i++) {
    hashmap->Commit(i, value_ptrs[i]);
  }
  embedding::Iterator* it = hashmap->GetIterator();
  int64 index = 0;
  float val_p[126] = {0.0};
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    int64 key = -1;
    it->Key((char*)&key, sizeof(int64));
    it->Value((char*)val_p, 126 * sizeof(float), 0);
    ASSERT_EQ(key, index);
    for (int i = 0; i < 126; i++)
      ASSERT_EQ(val_p[i], key);
    index++;
  }
}

TEST(EmbeddingVariableTest, TestLRUCache) {
  BatchCache<int64>* cache = new LRUCache<int64>();
  int num_ids = 30;
  int num_access = 100;
  int num_evict = 50;
  int64 ids[num_access] = {0};
  int64 evict_ids[num_evict] = {0};
  for (int i = 0; i < num_access; i++){
    ids[i] = i % num_ids;
  }
  cache->add_to_rank(ids, num_access);
  int64 size = cache->get_evic_ids(evict_ids, num_evict);
  ASSERT_EQ(size, num_ids);
  ASSERT_EQ(cache->size(), 0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(evict_ids[i], (num_access % num_ids + i) % num_ids);
  }
}

TEST(EmbeddingVariableTest, TestLFUCache) {
  BatchCache<int64>* cache = new LFUCache<int64>();
  int num_ids = 30;
  int num_access = 100;
  int num_evict = 50;
  int64 ids[num_access] = {0};
  int64 evict_ids[num_evict] = {0};
  for (int i = 0; i < num_access; i++){
    ids[i] = i % num_ids;
  }
  cache->add_to_rank(ids, num_access);
  int64 size = cache->get_evic_ids(evict_ids, num_evict);
  ASSERT_EQ(size, num_ids);
  ASSERT_EQ(cache->size(), 0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(evict_ids[i], (num_access % num_ids + i) % num_ids);
  }
}

void t1_gpu(KVInterface<int64, float>* hashmap) {
  for (int i = 0; i< 100; ++i) {
    hashmap->Insert(i, new NormalGPUValuePtr<float>(ev_allocator(), 100));
  }
}

TEST(EmbeddingVariableTest,TestRemoveLocklessCPU) {
    KVInterface<int64, float>* hashmap = new LocklessHashMapCPU<int64, float>();
    ASSERT_EQ(hashmap->Size(), 0);
    LOG(INFO) << "hashmap size: " << hashmap->Size();
    auto t = std::thread(t1, hashmap);
    t.join();
    LOG(INFO) << "hashmap size: " << hashmap->Size();
    ASSERT_EQ(hashmap->Size(), 100);
    TF_CHECK_OK(hashmap->Remove(1));
    TF_CHECK_OK(hashmap->Remove(2));
    ASSERT_EQ(hashmap->Size(), 98);
    LOG(INFO) << "2 size:" << hashmap->Size();
}

/*
void CommitGPU(KVInterface<int64, float>* hashmap) {
  for (int64 i = 0; i< 100; ++i) {
    ValuePtr<float>* tmp= new NormalGPUValuePtr<float>(ev_allocator(), 100);
    hashmap->Commit(i, tmp);
  }
}

TEST(EmbeddingVariableTest, TestCommitHashMapCPU) {
  KVInterface<int64, float>* hashmap = new LocklessHashMapCPU<int64, float>();
  hashmap->SetTotalDims(100);
  ASSERT_EQ(hashmap->Size(), 0);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  auto t = std::thread(CommitGPU, hashmap);
  t.join();
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  ASSERT_EQ(hashmap->Size(), 100);
  TF_CHECK_OK(hashmap->Remove(1));
  TF_CHECK_OK(hashmap->Remove(2));
  ASSERT_EQ(hashmap->Size(), 98);
  LOG(INFO) << "2 size:" << hashmap->Size();
}

TEST(EmbeddingVariableTest, TestGPUValuePtr) {
  int ev_list_size = 32;
  ValuePtr<float>* ptr_ = new NormalGPUValuePtr<float>(ev_allocator(), ev_list_size);
  float* address = *(float **)((char *)ptr_->GetPtr() + sizeof(FixedLengthHeader));
  float host_data[ev_list_size];
  float initial_data[ev_list_size];
  for(int i = 0;i < ev_list_size;++i){
    initial_data[i] = 10;
  }
  for(int i = 0;i < ev_list_size;++i){
    LOG(INFO) << i << " " << initial_data[i];
  }
  cudaMemcpy(address, initial_data, ev_list_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(host_data, address, ev_list_size * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0;i < ev_list_size;++i){
    LOG(INFO) << i << " " << host_data[i];
  }
}//Forbidden, due to no gpu allocator at that time

TEST(EmbeddingVariableTest, TestCommitValue) {
  int ev_list_size = 32;
  ValuePtr<float>* ptr_ = new NormalGPUValuePtr<float>(ev_allocator(),ev_list_size);
  float* address = *(float **)((char *)ptr_->GetPtr() + sizeof(FixedLengthHeader));
  float initial_data[ev_list_size];
  for(int i = 0;i < ev_list_size;++i){
    initial_data[i] = 10;
  }
  cudaMemcpy(address, initial_data, ev_list_size * sizeof(float), cudaMemcpyHostToDevice);
  KVInterface<int64, float>* hashmap = new LocklessHashMapCPU<int64, float>();
  hashmap->SetTotalDims(ev_list_size);
  hashmap->Commit(1, ptr_);
  ValuePtr<float>* check;
  hashmap->Lookup(1,&check);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  float* tmp = (float *)((char *)check->GetPtr() + sizeof(FixedLengthHeader));

  for(int i = 0;i < ev_list_size;++i){
    LOG(INFO) << i << " " << tmp[i];
    //ASSERT_EQ(tmp[i], 10);
  }//
}

TEST(EmbeddingVariableTest, TestBatchCommitofLocklessHashMapCPU) {
  KVInterface<int64, float>* hashmap = new LocklessHashMapCPU<int64, float>();
  const int EmbeddingSize = 16;
  const int BatchSize = 16;

  hashmap->SetTotalDims(EmbeddingSize);
  std::vector<ValuePtr<float>*> value_ptr_list;
  std::vector<int64> key_list;

  for(int64 i = 0; i < BatchSize; i++) {
    key_list.emplace_back(i);
    ValuePtr<float>* ptr_ = new NormalGPUValuePtr<float>(EmbeddingSize);
    float* address = *(float **)((char *)ptr_->GetPtr() + sizeof(FixedLengthHeader));
    float initial_data[EmbeddingSize];
    for(int j = 0;j < EmbeddingSize;++j){
      initial_data[j] = i;
      //LOG(INFO) << "initial[" << i << "][" << j << "]=" << initial_data[j];
    }
    cudaMemcpy(address, initial_data, EmbeddingSize * sizeof(float), cudaMemcpyHostToDevice);
    value_ptr_list.emplace_back(ptr_);
  }//initialize V on GPU

  timespec start,end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  hashmap->BatchCommit(key_list, value_ptr_list);
  clock_gettime(CLOCK_MONOTONIC, &end);
  std::cout << "time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms" << std::endl;

  for(int64 i = 0; i < BatchSize; i++) {
    ValuePtr<float>* check;
    hashmap->Lookup(i,&check);
    float* tmp = (float *)((char *)check->GetPtr() + sizeof(FixedLengthHeader));
    for(int j = 0;j < EmbeddingSize;++j){
      LOG(INFO) << "batch[" << i << "][" << j << "]=" << tmp[j];
      //ASSERT_EQ(tmp[j], i);
    }
  }//compare value after BatchCommit
}
*/

const int total_size = 1024 * 8;
const int th_num = 1;
const int malloc_size = total_size / th_num;

void malloc_use_allocator(Allocator* allocator){
  timespec start;
  timespec end;
  float* first = (float *)allocator->AllocateRaw(0, sizeof(float));

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < malloc_size; ++i) {
    int ev_list_size = 32;
    float* ptr_ = (float *)allocator->AllocateRaw(0, sizeof(float) * ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "cost time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";
}

TEST(EmbeddingVariableTest, TestEVMalloc) {
  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_use_allocator, ev_allocator());
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}

TEST(EmbeddingVariableTest, TestCPUMalloc) {
  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_use_allocator, cpu_allocator());
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}

TEST(EmbeddingVariableTest, TestGPUMalloc) {
  SessionOptions sops;
  std::unique_ptr<Device> device = DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");
  Allocator* gpu_allocator = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), TfGpuId(0), 1 << 26);

  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_use_allocator, gpu_allocator);
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}


TEST(EmbeddingVariableTest, TestCPUGPUMalloc) {
  SessionOptions sops;
  std::unique_ptr<Device> device = DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");

  Allocator* gpu_allocator = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), TfGpuId(0), 1 << 26);

  timespec start;
  timespec end;

  ValuePtr<float>* ptr_1 = new NormalGPUValuePtr<float>(gpu_allocator, 32);

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 1024 * 2; ++i) {
    int ev_list_size = 32;
    ValuePtr<float>* ptr_ = new NormalGPUValuePtr<float>(gpu_allocator, ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "cost time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";


  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 1024 * 2; ++i) {
    int ev_list_size = 32;
    ValuePtr<float>* ptr_ = new NormalValuePtr<float>(cpu_allocator(), ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "cost time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";


  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 1; ++i) {
    int ev_list_size = 32 * 1024 * 2;
    ValuePtr<float>* ptr_ = new NormalGPUValuePtr<float>(gpu_allocator, ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "cost time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";
}

void malloc_free_use_allocator(Allocator* allocator){
  timespec start;
  timespec end;
  std::vector<float*> ptrs;
  float* first = (float *)allocator->AllocateRaw(0, sizeof(float));

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < malloc_size; ++i) {
    int ev_list_size = 32;
    float* ptr_ = (float *)allocator->AllocateRaw(0, sizeof(float) * ev_list_size);
    ptrs.push_back(ptr_);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "first time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (auto iter = ptrs.begin();iter != ptrs.end();iter++) {
    allocator->DeallocateRaw(*iter);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "free time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < malloc_size; ++i) {
    int ev_list_size = 32;
    float* ptr_ = (float *)allocator->AllocateRaw(0, sizeof(float) * ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "second time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms";
}

TEST(EmbeddingVariableTest, TestEVMallocFree) {
  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_free_use_allocator, ev_allocator());
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}



} // namespace
} // namespace embedding
} // namespace tensorflow
